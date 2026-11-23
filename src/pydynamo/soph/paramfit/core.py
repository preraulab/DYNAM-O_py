"""Parametric-basis fits of a recording's SOPHs: a wrapper over the kernel.

The fits run in the shared Rust kernel (`dynamo_rs.fit_paramfits`), the
same code the DYNAM-O desktop app and command-line tool run, so a pydynamo
fit and an app fit of the same SOPHs are identical. This module only
converts between pydynamo's layout and the kernel's:

* SOPHs go in as (n_feature, n_freq), pydynamo's and the kernel's layout.
  The kernel returns its model the same way; `ParamFitResult.model_soph`
  is its transpose, (n_freq, n_feature), the layout pydynamo has always
  used for fitted surfaces.
* Options are the `paramfit` section of a DYNAM-O settings file, as a dict
  (``{"power": {...}, "phase": {...}, "peak_assign": ...}``). Omitted keys
  take the kernel's defaults; ``documents/SETTINGS_REFERENCE.md`` in the
  DesktopApp repo lists every key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pydynamo import _kernel

try:
    import dynamo_rs as _rs
except ImportError:  # pragma: no cover - exercised only without the kernel
    _rs = None


# The member-peak columns at the end of every paramfit table, in CSV order.
PK_COLUMNS = (
    "PkCount", "PkFreq", "PkDuration", "PkBandwidth", "PkHeight", "PkPower",
    "PkVolume", "PkArea", "PkPeakiness", "PkSOpower", "PkSOphase",
)

# Stats-table columns the kernel reads for the Pk* columns, by kernel key.
_PEAK_COLUMNS = {
    "freq": "PeakFrequency",
    "duration": "Duration",
    "bandwidth": "Bandwidth",
    "height": "Height",
    "volume": "Volume",
    "area": "Area",
    "peakiness": "Peakiness",
    "sopower": "SOpower",
    "sophase": "SOphase",
}


@dataclass
class ParamFitResult:
    """One axis's parametric fit, as the kernel returns it.

    `params` is (N, 6) `[Density, FreqMean, FreqStd, SOMean, SOStd, Theta]`
    as the paramfit CSV writes them (for phase, Density is the empirical
    amplitude and SOMean is wrapped to [-pi, pi]). `params_table` holds the
    full CSV table. `model_soph` is (n_freq, n_feature), zero outside the
    fit's analysis window. `unit_row` is 1 for a row-normalized (phase) fit
    and NaN otherwise, as the CSV records it.
    """
    params: np.ndarray
    background: np.ndarray
    model_soph: np.ndarray
    gof: dict
    unit_row: float
    peak_assign: str
    params_table: pd.DataFrame = field(default_factory=pd.DataFrame)


def _require_kernel():
    if _rs is None or not hasattr(_rs, "fit_paramfits"):
        _kernel.record_fallback("fit_paramfits")
        raise ImportError(
            "dynamo_rs with fit_paramfits is required for parametric basis "
            "fitting; rebuild it from DYNAM-O_rs (see README)."
        )


def _peak_dict(stats_table, peak_in_hist):
    """The kernel's `peaks` argument from a stats table, or None."""
    if stats_table is None or len(stats_table) == 0:
        return None
    peaks = {}
    for key, column in _PEAK_COLUMNS.items():
        peaks[key] = np.ascontiguousarray(stats_table[column].to_numpy(dtype=float))
    if peak_in_hist is None:
        in_hist = np.ones(len(stats_table), dtype=bool)
    else:
        in_hist = np.asarray(peak_in_hist, dtype=bool)
    peaks["in_hist"] = np.ascontiguousarray(in_hist)
    return peaks


def _axis_result(axis_out, n_freq, n_feature):
    """A ParamFitResult from one axis of the kernel's output, or None when
    the kernel declined the fit (its reason is raised as a warning by the
    caller)."""
    if axis_out["skipped"] is not None:
        return None
    table = pd.DataFrame(np.asarray(axis_out["table"]), columns=axis_out["columns"])
    table["PkCount"] = table["PkCount"].astype(int)
    # The kernel's model is (n_feature, n_freq); pydynamo keeps fitted
    # surfaces as (n_freq, n_feature).
    model = np.asarray(axis_out["model"], dtype=float)
    if model.shape != (n_feature, n_freq):
        raise ValueError(
            f"kernel model shape {model.shape}, expected {(n_feature, n_freq)}"
        )
    model = model.T
    return ParamFitResult(
        params=np.asarray(axis_out["params"], dtype=float).reshape(-1, 6),
        background=np.asarray(axis_out["background"], dtype=float),
        model_soph=model,
        gof=dict(axis_out["gof"]),
        unit_row=float(axis_out["unit_row"]),
        peak_assign=str(axis_out["peak_assign"]),
        params_table=table,
    )


def fit_paramfits(
    sopower_mat,
    sophase_mat,
    sopower_bins,
    sophase_bins,
    freq_bins,
    settings: dict | None = None,
    *,
    stats_table: pd.DataFrame | None = None,
    peak_in_hist=None,
    baseline=None,
):
    """Fit both SOPHs. Returns ``(power, phase, skipped)``.

    * ``sopower_mat`` / ``sophase_mat``: (n_feature, n_freq) SOPHs.
    * ``settings``: the settings-file ``paramfit`` section (dict), or None
      for the defaults.
    * ``stats_table`` / ``peak_in_hist``: every TF-peak and which ones the
      SOPHs counted, for the ``Pk*`` columns. Without them those columns
      hold a count of 0 and NaN.
    * ``baseline``: ``(values, freqs)``, the extraction baseline (linear
      power), for ``PkPower``.

    ``power`` / ``phase`` are `ParamFitResult`, or None when the kernel
    declined that fit on degenerate input; ``skipped`` maps the declined
    axis to the kernel's reason.
    """
    _require_kernel()
    freq_bins = np.ascontiguousarray(np.asarray(freq_bins, dtype=float).ravel())
    sopower_bins = np.ascontiguousarray(np.asarray(sopower_bins, dtype=float).ravel())
    sophase_bins = np.ascontiguousarray(np.asarray(sophase_bins, dtype=float).ravel())
    sopower_hist = np.ascontiguousarray(np.asarray(sopower_mat, dtype=float))
    sophase_hist = np.ascontiguousarray(np.asarray(sophase_mat, dtype=float))

    baseline_values = None
    baseline_freqs = None
    if baseline is not None:
        baseline_values = np.ascontiguousarray(np.asarray(baseline[0], dtype=float).ravel())
        baseline_freqs = np.ascontiguousarray(np.asarray(baseline[1], dtype=float).ravel())

    opts_json = json.dumps(settings) if settings is not None else None
    out = _rs.fit_paramfits(
        sopower_hist, sophase_hist, sopower_bins, sophase_bins, freq_bins,
        opts_json=opts_json,
        peaks=_peak_dict(stats_table, peak_in_hist),
        baseline=baseline_values,
        baseline_freqs=baseline_freqs,
    )
    skipped = {}
    for axis in ("power", "phase"):
        if out[axis]["skipped"] is not None:
            skipped[axis] = out[axis]["skipped"]
    power = _axis_result(out["power"], freq_bins.size, sopower_bins.size)
    phase = _axis_result(out["phase"], freq_bins.size, sophase_bins.size)
    return power, phase, skipped


def rebuild_phase_model(modes, background, phase_bins, freq_bins):
    """A phase fit's model surface rebuilt from its paramfit CSV values.

    ``modes`` is (N, 6) with the recorded Density in column 0. Returns
    (n_freq, n_phase), matching the fit's own model. The kernel does the
    work (`fits::rebuild_phase_model`).
    """
    _require_kernel()
    return _rs.rebuild_phase_model(
        np.ascontiguousarray(np.asarray(modes, dtype=float).reshape(-1, 6)),
        np.ascontiguousarray(np.asarray(background, dtype=float).ravel()),
        np.ascontiguousarray(np.asarray(phase_bins, dtype=float).ravel()),
        np.ascontiguousarray(np.asarray(freq_bins, dtype=float).ravel()),
    )
