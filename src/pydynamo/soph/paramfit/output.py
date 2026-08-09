"""MATLAB-compatible parametric-fit result tables and annotations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import i0e


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


def _mode_peak_mask(mode_params, kind, stats_table, prob):
    if not 0.0 < prob < 1.0:
        raise ValueError("prob must be in (0, 1)")

    _, freq_mean, freq_std, feature_mean, feature_std, theta = mode_params
    if not (freq_std > 0.0 and feature_std > 0.0):
        return np.zeros(len(stats_table), dtype=bool)

    frequency = stats_table["PeakFrequency"].to_numpy(dtype=float)
    feature_column = "SOphase" if kind == "phase" else "SOpower"
    feature = stats_table[feature_column].to_numpy(dtype=float)
    delta_freq = frequency - freq_mean

    if kind == "phase":
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
        q = 0.5 * (
            (u / freq_std) ** 2 + (v / feature_std) ** 2
        )
    else:
        raise ValueError("kind must be 'power' or 'phase'")

    threshold = -np.log1p(-prob)
    return (
        np.isfinite(frequency)
        & np.isfinite(feature)
        & np.isfinite(q)
        & (q <= threshold)
    )


def annotate_modes_with_peak_stats(
    table: pd.DataFrame,
    kind: str,
    stats_table_soph,
    prob: float,
) -> pd.DataFrame:
    """Fill PkCount and nine per-mode TF-peak summaries."""
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
        for mode_idx in range(n_modes):
            mode_params = out.iloc[mode_idx, :6].to_numpy(dtype=float)
            members = _mode_peak_mask(
                mode_params, kind, stats_table_soph, float(prob)
            )
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
