"""Spline-basis fitting for SO-power and SO-phase histograms.

This is the Python wrapper around ``dynamo_rs.fit_tensor_product_spline``.
It mirrors MATLAB's ``spline_basis.m`` domain filtering and knot construction;
the least-squares spline itself is evaluated by the shared Rust kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi
from typing import Literal

import numpy as np
from scipy.interpolate import BSpline

try:
    import dynamo_rs as _rs
except ImportError:  # pragma: no cover - exercised only without the extension
    _rs = None


@dataclass(frozen=True)
class SplineBasisOpts:
    """Options that affect one spline-basis fit."""

    kind: Literal["power", "phase"] = "power"
    feature_limits: tuple[float, float] = (-2.0, 20.0)
    freq_limits: tuple[float, float] = (2.0, 18.0)
    num_knots_x: int = 5
    num_knots_y: int = 18

    def __post_init__(self):
        if self.kind not in ("power", "phase"):
            raise ValueError("kind must be 'power' or 'phase'")

        for name in ("feature_limits", "freq_limits"):
            values = np.asarray(getattr(self, name), dtype=float).ravel()
            if (
                values.size != 2
                or not np.isfinite(values).all()
                or values[0] >= values[1]
            ):
                raise ValueError(f"{name} must contain two increasing finite values")
            object.__setattr__(self, name, (float(values[0]), float(values[1])))

        for name in ("num_knots_x", "num_knots_y"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))

    @staticmethod
    def power(**overrides) -> "SplineBasisOpts":
        """MATLAB ``spline_basis_opts('power')`` defaults."""
        return replace(SplineBasisOpts(), **overrides)

    @staticmethod
    def phase(**overrides) -> "SplineBasisOpts":
        """MATLAB ``spline_basis_opts('phase')`` defaults."""
        base = SplineBasisOpts(
            kind="phase",
            feature_limits=(-pi, pi),
            freq_limits=(2.0, 18.0),
            num_knots_x=5,
            num_knots_y=9,
        )
        return replace(base, **overrides)


@dataclass
class SplineObject:
    """Portable numerical equivalent of MATLAB's tensor spline object.

    ``coefs`` uses the emitted ``(frequency_basis, feature_basis)`` layout.
    Calling the object evaluates the fitted surface on feature/frequency axes.
    """

    coefs: np.ndarray
    knots_x: np.ndarray
    knots_y: np.ndarray
    order: tuple[int, int] = (4, 4)
    form: str = "B-"
    dim: int = 1

    @property
    def number(self) -> tuple[int, int]:
        return (self.coefs.shape[1], self.coefs.shape[0])

    def __call__(self, feature_bins, freq_bins) -> np.ndarray:
        feature_bins = _as_axis(feature_bins, "feature_bins")
        freq_bins = _as_axis(freq_bins, "freq_bins")
        feature_basis = BSpline.design_matrix(
            feature_bins, self.knots_x, self.order[0] - 1,
        ).toarray()
        freq_basis = BSpline.design_matrix(
            freq_bins, self.knots_y, self.order[1] - 1,
        ).toarray()
        return feature_basis @ self.coefs.T @ freq_basis.T


@dataclass
class SplineFitResult:
    """One axis's spline fit and the information needed to reconstruct it."""

    splinefit: np.ndarray
    coefs: np.ndarray
    spline_obj: SplineObject
    knots_x: np.ndarray
    knots_y: np.ndarray
    fit_SOfeature_bins: np.ndarray
    fit_freq_bins: np.ndarray
    knots_x_aug: np.ndarray
    knots_y_aug: np.ndarray


def _as_axis(values, name: str) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    if raw.ndim == 0 or raw.ndim > 2 or (raw.ndim == 2 and 1 not in raw.shape):
        raise ValueError(f"{name} must be a vector")
    axis = raw.ravel()
    if (
        axis.size == 0
        or not np.isfinite(axis).all()
        or np.any(np.diff(axis) <= 0)
    ):
        raise ValueError(f"{name} must be a finite increasing vector")
    return axis


def _orient_soph(
    soph: np.ndarray,
    feature_bins: np.ndarray,
    freq_bins: np.ndarray,
) -> np.ndarray:
    """Return SOPH as ``(n_freq, n_feature)``.

    MATLAB-layout ``(n_feature, n_freq)`` is checked first, matching
    ``spline_basis.m`` when the two dimensions happen to be equal.
    """
    if soph.shape == (feature_bins.size, freq_bins.size):
        return soph.T
    if soph.shape == (freq_bins.size, feature_bins.size):
        return soph
    raise ValueError(
        f"incompatible SOPH dimensions: soph {soph.shape}, "
        f"feature_bins {feature_bins.size}, freq_bins {freq_bins.size}"
    )


def fit_spline_basis(
    soph,
    feature_bins,
    freq_bins,
    opts: SplineBasisOpts | None = None,
    kind: Literal["power", "phase"] = "power",
) -> SplineFitResult:
    """Fit one SOPH with MATLAB-compatible filtering and cubic B-splines.

    The returned ``splinefit`` has shape ``(n_fit_feature, n_fit_freq)`` and
    ``coefs`` has shape ``(num_knots_y + 2, num_knots_x + 2)``, matching
    MATLAB's emitted arrays.
    """
    if opts is None:
        if kind not in ("power", "phase"):
            raise ValueError("kind must be 'power' or 'phase'")
        opts = (
            SplineBasisOpts.power()
            if kind == "power"
            else SplineBasisOpts.phase()
        )
    if _rs is None or not hasattr(_rs, "fit_tensor_product_spline"):
        raise ImportError(
            "dynamo_rs with fit_tensor_product_spline is required for spline "
            "basis fitting; rebuild the coordinated native extensions with the "
            "DYNAM-O_toolbox controlled bootstrap (`./bootstrap.sh --yes` or "
            "`bootstrap.ps1 -Yes`)"
        )

    feature_bins = _as_axis(feature_bins, "feature_bins")
    freq_bins = _as_axis(freq_bins, "freq_bins")
    soph = np.asarray(soph, dtype=float)
    if soph.ndim != 2 or soph.size == 0:
        raise ValueError("soph must be a nonempty matrix")
    if np.any(soph < 0):
        raise ValueError("soph must be nonnegative")
    soph = _orient_soph(soph, feature_bins, freq_bins)

    valid_mat = np.isfinite(soph)
    invalid_freq = np.all(~valid_mat, axis=1)
    valid_mat[invalid_freq, :] = True
    valid_feature = (
        (feature_bins >= opts.feature_limits[0])
        & (feature_bins <= opts.feature_limits[1])
        & np.all(valid_mat, axis=0)
    )
    valid_freq = (
        (freq_bins >= opts.freq_limits[0])
        & (freq_bins <= opts.freq_limits[1])
        & ~invalid_freq
    )
    if not valid_feature.any() or not valid_freq.any():
        raise ValueError("no finite SOPH bins remain inside the fit limits")

    fit_feature_bins = feature_bins[valid_feature]
    fit_freq_bins = freq_bins[valid_freq]
    fit_soph = soph[np.ix_(valid_freq, valid_feature)]

    knots_x = np.concatenate((
        [fit_feature_bins[0] - 0.1],
        np.linspace(
            fit_feature_bins[0],
            fit_feature_bins[-1],
            opts.num_knots_x,
        ),
        [fit_feature_bins[-1] + 0.1],
    ))
    knots_y = np.concatenate((
        [fit_freq_bins[0] - 0.1],
        np.linspace(
            fit_freq_bins[0],
            fit_freq_bins[-1],
            opts.num_knots_y,
        ),
        [fit_freq_bins[-1] + 0.1],
    ))

    raw = _rs.fit_tensor_product_spline(
        np.ascontiguousarray(fit_soph.T),
        np.ascontiguousarray(fit_feature_bins),
        np.ascontiguousarray(fit_freq_bins),
        np.ascontiguousarray(knots_x),
        np.ascontiguousarray(knots_y),
        4,
        3,
    )

    splinefit = np.ascontiguousarray(raw["splinefit"], dtype=float)
    coefs = np.ascontiguousarray(raw["coefs"], dtype=float)
    knots_x_aug = np.ascontiguousarray(raw["knots_x_aug"], dtype=float)
    knots_y_aug = np.ascontiguousarray(raw["knots_y_aug"], dtype=float)
    spline_obj = SplineObject(
        coefs=coefs,
        knots_x=knots_x_aug,
        knots_y=knots_y_aug,
    )

    return SplineFitResult(
        splinefit=splinefit,
        coefs=coefs,
        spline_obj=spline_obj,
        knots_x=knots_x,
        knots_y=knots_y,
        fit_SOfeature_bins=fit_feature_bins,
        fit_freq_bins=fit_freq_bins,
        knots_x_aug=knots_x_aug,
        knots_y_aug=knots_y_aug,
    )


__all__ = [
    "SplineBasisOpts",
    "SplineFitResult",
    "SplineObject",
    "fit_spline_basis",
]
