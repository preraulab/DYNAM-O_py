"""MATLAB-compatible parametric-fit result tables and annotations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import i0e

from pydynamo.soph.paramfit.opts import parse_peak_assign

# 3-sigma footprint cap for the density rules (background/argmax); matches
# Rust dynamo_pipeline::mode_peaks::DENSITY_ASSIGN_Q_CAP and the MATLAB
# assign_mode_peaks.m Q_CAP. Numeric contours are not capped.
_DENSITY_ASSIGN_Q_CAP = 4.5


PK_COLUMNS = (
    "PkCount",
    "PkFreq",
    "PkDuration",
    "PkBandwidth",
    "PkHeight",
    "PkVolume",
    "PkArea",
    "PkPeakiness",
    "PkSOpower",
    "PkSOphase",
)

POWER_PARAM_COLUMNS = (
    "Density",
    "FreqMean",
    "FreqStd",
    "SOpowerMean",
    "SOpowerStd",
    "Theta",
    "Volume",
    "PrefPhase",
    "Coupling",
    *PK_COLUMNS,
)

PHASE_PARAM_COLUMNS = (
    "Density",
    "FreqMean",
    "FreqStd",
    "SOphaseMean",
    "SOphaseStd",
    "Theta",
    "Volume",
    *PK_COLUMNS,
)

_PK_SOURCE_COLUMNS = (
    "PeakFrequency",
    "Duration",
    "Bandwidth",
    "Height",
    "Volume",
    "Area",
    "Peakiness",
    "SOpower",
    "SOphase",
)


def create_params_table(params, kind: str) -> pd.DataFrame:
    """Pack raw mode parameters into MATLAB's stable named schema."""
    params = np.asarray(params, dtype=float)
    if params.size == 0:
        params = np.empty((0, 6), dtype=float)
    elif params.ndim == 1:
        params = params.reshape(1, -1)
    if params.ndim != 2 or params.shape[1] != 6:
        raise ValueError("params must have shape (n_modes, 6)")

    if kind == "power":
        columns = POWER_PARAM_COLUMNS
    elif kind == "phase":
        columns = PHASE_PARAM_COLUMNS
    else:
        raise ValueError("kind must be 'power' or 'phase'")

    values = np.full((params.shape[0], len(columns)), np.nan, dtype=float)
    values[:, :6] = params

    density = params[:, 0]
    freq_std = params[:, 2]
    feature_std = params[:, 4]
    if kind == "power":
        values[:, 6] = density * (2.0 * np.pi) * feature_std * freq_std
    else:
        kappa = 1.0 / feature_std**2
        values[:, 6] = (
            density * (2.0 * np.pi) * i0e(kappa)
            * np.sqrt(2.0 * np.pi) * freq_std
        )

    values[:, columns.index("PkCount")] = 0.0
    return pd.DataFrame(values, columns=columns, dtype=np.float64)


def annotate_power_preferred_phase(
    table: pd.DataFrame,
    model_soph_phase,
    freq_bins,
    phase_bins,
) -> pd.DataFrame:
    """Fill power-mode PrefPhase/Coupling from the fitted phase surface."""
    out = table.copy()
    if out.empty or model_soph_phase is None or np.size(model_soph_phase) == 0:
        return out

    freq_bins = np.asarray(freq_bins, dtype=float).ravel()
    phase_bins = np.asarray(phase_bins, dtype=float).ravel()
    model = np.asarray(model_soph_phase, dtype=float)
    if not np.any(np.isfinite(model)):
        return out
    if model.shape == (freq_bins.size, phase_bins.size):
        pass
    elif model.shape == (phase_bins.size, freq_bins.size):
        model = model.T
    else:
        raise ValueError(
            "model_soph_phase must have shape (n_freq, n_phase) "
            "or (n_phase, n_freq)"
        )

    freq_idx = np.abs(
        freq_bins[:, None] - out["FreqMean"].to_numpy(dtype=float)[None, :]
    ).argmin(axis=0)
    profiles = model[freq_idx, :]
    phase_idx = np.argmax(profiles, axis=1)
    out["PrefPhase"] = phase_bins[phase_idx]
    out["Coupling"] = profiles[np.arange(len(out)), phase_idx]
    return out


def _mode_q(mode_params, kind, stats_table):
    """Per-peak Q for one mode, plus the valid-input mask.

    Q is the shared containment quadratic (see the Rust ``mode_quadratic``
    and MATLAB ``get_mode_peaks``). A degenerate mode (non-positive width)
    yields an all-False mask.
    """
    _, freq_mean, freq_std, feature_mean, feature_std, theta = mode_params

    frequency = stats_table["PeakFrequency"].to_numpy(dtype=float)
    feature_column = "SOphase" if kind == "phase" else "SOpower"
    feature = stats_table[feature_column].to_numpy(dtype=float)
    delta_freq = frequency - freq_mean

    if kind == "phase":
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = 1.0 / feature_std**2
            delta = (
                feature - feature_mean + delta_freq * np.sin(theta) + np.pi
            ) % (2.0 * np.pi) - np.pi
            q = (
                0.5 * (delta_freq / freq_std) ** 2
                + kappa * (1.0 - np.cos(delta))
            )
    elif kind == "power":
        delta_feature = feature - feature_mean
        u = delta_freq * np.cos(theta) + delta_feature * np.sin(theta)
        v = -delta_freq * np.sin(theta) + delta_feature * np.cos(theta)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = 0.5 * (
                (u / freq_std) ** 2 + (v / feature_std) ** 2
            )
    else:
        raise ValueError("kind must be 'power' or 'phase'")

    ok = np.isfinite(frequency) & np.isfinite(feature) & np.isfinite(q)
    if not (freq_std > 0.0 and feature_std > 0.0):
        ok = np.zeros(len(stats_table), dtype=bool)
    return q, ok


def _mode_peak_mask(mode_params, kind, stats_table, prob):
    """Numeric-contour membership (the pre-argmax primitive)."""
    if not 0.0 < prob < 1.0:
        raise ValueError("prob must be in (0, 1)")
    q, ok = _mode_q(mode_params, kind, stats_table)
    return ok & (q <= -np.log1p(-prob))


def _assign_mode_peaks(params6, kind, stats_table, assign, background=None):
    """Member masks for every mode under one assignment rule.

    Mirrors Rust ``dynamo_pipeline::mode_peaks::assign_members`` and MATLAB
    ``assign_mode_peaks.m``:

    * numeric (``p`` / ``sigma``): per-mode independent contour Q <= thr.
    * ``background`` (power axis): member iff the mode's density at the
      peak, ``amp*exp(-Q)``, is at least the background plane
      ``xxx*P + yyy*F + zzz`` evaluated at the peak; non-finite background
      fails closed. Non-exclusive.
    * ``argmax`` (power axis, exclusive): the peak goes to the first mode
      attaining the maximal density, and only when that density beats the
      background there (ties with the background go to the background).
    * Both density rules operate within each mode's 3-sigma footprint
      (``Q <= _DENSITY_ASSIGN_Q_CAP``): outside it a mode neither claims a
      peak nor blocks another mode's argmax. Where the background plane
      approaches zero, an uncapped comparison would admit arbitrarily far
      tails. Numeric contours are NOT capped.
    * The phase axis falls back to the ``0.95 p`` contour under the
      density rules: phase Density is a row-normalized empirical height
      while the stored phase background is pre-normalization.

    Returns an ``(n_peaks, n_modes)`` boolean array.
    """
    assign = parse_peak_assign(assign)
    if kind == "phase" and assign["kind"] in ("background", "argmax"):
        assign = parse_peak_assign("0.95 p")

    params6 = np.asarray(params6, dtype=float).reshape(-1, 6)
    n_modes = params6.shape[0]
    n_peaks = len(stats_table)
    members = np.zeros((n_peaks, n_modes), dtype=bool)
    if n_peaks == 0 or n_modes == 0:
        return members

    if assign["kind"] in ("p", "sigma"):
        thr = assign["thr"]
        for m in range(n_modes):
            q, ok = _mode_q(params6[m], kind, stats_table)
            members[:, m] = ok & (q <= thr)
        return members

    # Density rules (power axis only by this point).
    background = np.asarray(
        [np.nan] * 3 if background is None else background, dtype=float
    )
    frequency = stats_table["PeakFrequency"].to_numpy(dtype=float)
    feature = stats_table["SOpower"].to_numpy(dtype=float)
    density = np.zeros((n_peaks, n_modes), dtype=float)
    for m in range(n_modes):
        q, ok = _mode_q(params6[m], kind, stats_table)
        amp = params6[m, 0]
        if np.isfinite(amp) and amp > 0.0:
            in_cap = ok & (q <= _DENSITY_ASSIGN_Q_CAP)
            density[in_cap, m] = amp * np.exp(-q[in_cap])
    bg = background[0] * feature + background[1] * frequency + background[2]

    if assign["kind"] == "background":
        finite_bg = np.isfinite(bg)
        for m in range(n_modes):
            members[:, m] = (
                (density[:, m] > 0.0) & finite_bg & (density[:, m] >= bg)
            )
        return members

    # argmax: exclusive, first mode wins ties, background wins a tie.
    best = np.argmax(density, axis=1)          # first maximizer (np.argmax)
    best_d = density[np.arange(n_peaks), best]
    assigned = (best_d > 0.0) & np.isfinite(bg) & (best_d > bg)
    members[np.arange(n_peaks), best] = assigned
    return members


def annotate_modes_with_peak_stats(
    table: pd.DataFrame,
    kind: str,
    stats_table_soph,
    assign="argmax",
    background=None,
) -> pd.DataFrame:
    """Fill PkCount and nine per-mode TF-peak summaries.

    ``assign`` accepts anything ``parse_peak_assign`` does — a numeric value
    keeps this function backward compatible with the old ``prob`` argument.
    ``background`` is the fit's ``[xxx, yyy, zzz]``; only the density rules
    consult it (and only on the power axis — see ``_assign_mode_peaks``).
    """
    out = table.copy()
    n_modes = len(out)
    counts = np.zeros(n_modes, dtype=float)
    summaries = np.full((n_modes, len(_PK_SOURCE_COLUMNS)), np.nan, dtype=float)

    feature_column = "SOphase" if kind == "phase" else "SOpower"
    have_stats = (
        isinstance(stats_table_soph, pd.DataFrame)
        and not stats_table_soph.empty
        and {"PeakFrequency", feature_column}.issubset(stats_table_soph.columns)
    )
    if have_stats:
        params6 = out.iloc[:, :6].to_numpy(dtype=float)
        member_matrix = _assign_mode_peaks(
            params6, kind, stats_table_soph, assign, background
        )
        for mode_idx in range(n_modes):
            members = member_matrix[:, mode_idx]
            counts[mode_idx] = float(np.count_nonzero(members))

            for prop_idx, source_name in enumerate(_PK_SOURCE_COLUMNS):
                if source_name not in stats_table_soph.columns:
                    continue
                values = stats_table_soph[source_name].to_numpy(dtype=float)[
                    members
                ]
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                if source_name == "SOphase":
                    summaries[mode_idx, prop_idx] = np.angle(
                        np.mean(np.exp(1j * values))
                    )
                else:
                    summaries[mode_idx, prop_idx] = np.mean(values)

    out["PkCount"] = counts
    for output_name, values in zip(PK_COLUMNS[1:], summaries.T):
        out[output_name] = values
    return out
