"""Parametric-basis fit options — port of param_basis_opts.m.

`ParamBasisOpts.power()` and `.phase()` reproduce the two default blocks in
param_basis_opts.m:96-152. Field names match the MATLAB option names so a
MATLAB config can be transcribed one-for-one.

The x axis is the SO-feature axis: SO-power (dB) for the power fit, SO-phase
(rad) for the phase fit. The y axis is always frequency (Hz). Mode parameter
rows are always ``[amp, fmean, fstd, xmean, xstd, theta]``; ``fstd`` and
``xstd`` are standard deviations, not variances (param_basis_opts.m:56-58).

The width bounds below are the historical literals divided by sqrt(2). That
factor is the sigma reparameterization: the kernels used to omit the 1/2 in
the exponent, so the fitted width was sqrt(2)*sigma and the bound literals
were written in those units. Dividing keeps each bound describing the same
*physical* window. The phase ``xstd`` (recikappa) bounds are deliberately NOT
divided — that parameter was always a true sigma.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import pi, sqrt
from typing import Literal, Sequence

import numpy as np

from pydynamo.soph.paramfit.basis import SQRT2

Criterion = Literal["max", "mindr2", "minpctr2", "kneedle"]

NAN = float("nan")


@dataclass(frozen=True)
class ParamBasisOpts:
    """Options for one parametric-basis fit (one axis)."""

    # Which axis this option set describes. Drives the handful of places the
    # two fits genuinely differ (basis function, seam wrapping, smoothing).
    kind: Literal["power", "phase"] = "power"

    # Analysis window. `feature_limits` is power_limits (dB) or phase_limits
    # (rad) depending on `kind`.
    feature_limits: tuple = (-2.0, 20.0)
    freq_limits: tuple = (2.0, 18.0)

    # Watershed seeding: [merge_thresh, dur_min, bw_min, height_min, trim_vol].
    # merge_thresh = nan means "let extracthistpeaks pick it".
    watershed_params: tuple = (NAN, 4.0, 0.25, 0.0, 0.7)
    # Gaussian smoothing std applied before the seeding watershed. Phase only;
    # None means no smoothing.
    gauss_filt_std: tuple | None = None
    # Run the seeding watershed on exp(SOPH) instead of SOPH.
    wshed_exp: bool = False

    max_peaks: int = 6
    # Modes injected ahead of / instead of the watershed seeds. (N, 6).
    prefix_modes: Sequence = field(default_factory=tuple)
    # 1 = before watershed modes, -1 = after, 0 = replace them entirely.
    prefix_modes_order: int = -1

    max_overlap: float = 0.25
    min_amp: float = 0.0
    # Frequency exclusion radius (Hz) for seed dedup and the too-close revert
    # check. 0 disables both — the phase axis passes 0.
    min_freq_diff: float = 0.5

    criterion: Criterion = "minpctr2"
    min_dr2: float = 0.01
    min_pctr2: float = 0.01
    kneedle_tol: float = 0.01

    # Per-mode bounds, same column order as a mode row. nan entries are filled
    # from the data in `resolve_bounds`.
    # Both width slots are rotGauss sigmas on the power axis, so both carry
    # the /sqrt(2) sigma rescale of the historical 2.5 / 0.1 / 30 / 2.5.
    UB_default: tuple = (NAN, NAN, 2.5 / SQRT2, NAN, 30.0 / SQRT2, 0.03)
    LB_default: tuple = (NAN, NAN, 0.1 / SQRT2, NAN, 2.5 / SQRT2, -0.03)
    constrain_freq_center: bool = True
    _constrain_feature_center: bool = True

    # LM iteration cap handed to the Rust kernel; 0 selects its default of
    # 100 * n_params, which is what scipy least_squares uses.
    max_iters: int = 0

    verbose: bool = False
    peak_assign_prob: float = 0.95

    def __post_init__(self):
        feature_name = (
            "constrain_power_center" if self.kind == "power"
            else "constrain_phase_center"
        )
        flags = {
            "constrain_freq_center": self.constrain_freq_center,
            feature_name: self._constrain_feature_center,
        }
        for name, value in flags.items():
            is_bool = isinstance(value, (bool, np.bool_))
            is_binary_number = (
                isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value)
                and value in (0, 1)
            )
            if not (is_bool or is_binary_number):
                raise ValueError(f"{name} must be a binary scalar")

        prob = self.peak_assign_prob
        is_numeric = (
            isinstance(prob, (int, float, np.integer, np.floating))
            and not isinstance(prob, (bool, np.bool_))
        )
        if not (
            is_numeric and np.isfinite(prob) and 0.0 < float(prob) < 1.0
        ):
            raise ValueError("peak_assign_prob must be in (0, 1)")

    @property
    def constrain_power_center(self):
        if self.kind != "power":
            raise AttributeError(
                "constrain_power_center is only valid for power options"
            )
        return self._constrain_feature_center

    @property
    def constrain_phase_center(self):
        if self.kind != "phase":
            raise AttributeError(
                "constrain_phase_center is only valid for phase options"
            )
        return self._constrain_feature_center

    @staticmethod
    def power(**overrides) -> "ParamBasisOpts":
        """Defaults for the SO-power fit (param_basis_opts.m:97-122)."""
        if "constrain_phase_center" in overrides:
            raise TypeError(
                "constrain_phase_center is only valid for phase options"
            )
        if "_constrain_feature_center" in overrides:
            raise TypeError("use constrain_power_center")
        if "constrain_power_center" in overrides:
            overrides["_constrain_feature_center"] = overrides.pop(
                "constrain_power_center"
            )
        return replace(ParamBasisOpts(), **overrides)

    @staticmethod
    def phase(**overrides) -> "ParamBasisOpts":
        """Defaults for the SO-phase fit (param_basis_opts.m:124-152).

        Note `min_freq_diff` is 0: the phase axis has no frequency-dedup rule
        in MATLAB, and residual_max_seed.m documents 0 as the phase-axis
        convention that disables the exclusion mask.
        """
        if "constrain_power_center" in overrides:
            raise TypeError(
                "constrain_power_center is only valid for power options"
            )
        if "_constrain_feature_center" in overrides:
            raise TypeError("use constrain_phase_center")
        if "constrain_phase_center" in overrides:
            overrides["_constrain_feature_center"] = overrides.pop(
                "constrain_phase_center"
            )
        base = ParamBasisOpts(
            kind="phase",
            feature_limits=(-pi, pi),
            freq_limits=(2.0, 18.0),
            watershed_params=(NAN, pi / 6, 2.0, 1e-4, 0.4),
            gauss_filt_std=(10.0, 5.0),
            wshed_exp=False,
            max_peaks=3,
            max_overlap=0.15,
            min_amp=1e-4,
            min_freq_diff=0.0,
            criterion="minpctr2",
            min_dr2=0.01,
            min_pctr2=0.025,
            # Slot 2 is the vmGauss frequency sigma (Hz). MATLAB's phase
            # UB_default(3) is sqrt(15) in the sqrt(2)*sigma convention, so
            # the sigma-equivalent is sqrt(15)/sqrt(2) = sqrt(7.5).
            #
            # Slot 4 is recikappa, which the von Mises factor already makes a
            # true sigma, so 2*pi and pi/5 stay exactly as they are. These two
            # slots are not symmetric -- do not "tidy" them into one rescale.
            UB_default=(NAN, NAN, sqrt(7.5), 2 * pi, 2 * pi, pi / 3),
            LB_default=(NAN, NAN, 1.0 / SQRT2, -2 * pi, pi / 5, -pi / 3),
        )
        return replace(base, **overrides)


def resolve_bounds(
    opts: ParamBasisOpts,
    amp0: np.ndarray,
    valid_feature_axis: np.ndarray,
    valid_freq_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill the nan entries of UB_default/LB_default from the data.

    Mirrors param_basis_power.m:238-275. Amplitude bounds come from the seed
    amplitudes (x10 / /10); center bounds come from the valid axis extents,
    then are released to +/-inf when the corresponding `constrain_*` flag is
    off.
    """
    ub = np.asarray(opts.UB_default, dtype=float).copy()
    lb = np.asarray(opts.LB_default, dtype=float).copy()
    amp0 = np.asarray(amp0, dtype=float).ravel()

    if np.isnan(ub[0]):
        ub[0] = np.nanmax(amp0) * 10.0
    if np.isnan(lb[0]):
        lb[0] = np.nanmin(amp0) / 10.0

    if np.isnan(ub[1]):
        ub[1] = np.max(valid_freq_axis)
    if np.isnan(lb[1]):
        lb[1] = np.min(valid_freq_axis)

    if np.isnan(ub[3]):
        ub[3] = np.max(valid_feature_axis)
    if np.isnan(lb[3]):
        lb[3] = np.min(valid_feature_axis)

    if not opts.constrain_freq_center:
        ub[1], lb[1] = np.inf, -np.inf
    constrain_feature_center = (
        opts.constrain_power_center if opts.kind == "power"
        else opts.constrain_phase_center
    )
    if not constrain_feature_center:
        ub[3], lb[3] = np.inf, -np.inf

    return lb, ub
