"""``run_dynamo``: one recording through the whole DYNAM-O run.

The run is the kernel's (``dynamo_rs.run_dynamo``), the code the DYNAM-O
desktop app and CLI run: resampling, artifact detection, both multitaper
passes, baseline, TF-peak extraction and refinement, per-peak stage /
SO-power / SO-phase, the SOPHs, and the parametric and spline fits. Given
the same recording and settings, every number matches the app's. This
module converts the kernel's output into pydynamo's result types and,
optionally, draws the summary figure.
"""

from __future__ import annotations

import copy
import json
import time as _time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pydynamo import _kernel
from pydynamo.settings import default_settings, merge_settings
from pydynamo.soph.paramfit import ParamFitResult, paramfit_result_from_kernel
from pydynamo.soph.splinefit import SplineFitResult, spline_result_from_kernel

#: Stats-table columns, in the stats CSV's order.
STATS_COLUMNS = (
    "PeakTime", "PeakFrequency", "Duration", "Bandwidth", "Height", "Volume",
    "SegmentNum", "Area", "Peakiness", "bbox_tl_s", "bbox_tl_Hz", "PeakStage",
    "SOpower", "SOphase",
)


@dataclass
class SOPHsResult:
    """The SO-power and SO-phase histograms of a run, their time series and
    fits. ``SOpower_mat`` / ``SOphase_mat`` are ``(n_feature, n_freq)``."""
    SOpower_mat: np.ndarray
    SOphase_mat: np.ndarray
    SOpower_bins: np.ndarray
    SOphase_bins: np.ndarray
    freq_bins: np.ndarray
    SOpower_norm: np.ndarray
    SOpower_times: np.ndarray
    SOphase: np.ndarray            # wrapped to [-pi, pi]
    SOphase_times: np.ndarray
    SOpower_paramfit: ParamFitResult | None = None
    SOphase_paramfit: ParamFitResult | None = None
    SOpower_splinefit: SplineFitResult | None = None
    SOphase_splinefit: SplineFitResult | None = None


@dataclass
class DynamoOutput:
    """Everything one run computes.

    ``stats_table`` holds every TF-peak (the stats CSV's columns);
    ``peak_in_hist`` marks the peaks the SOPHs counted. ``spect`` /
    ``stimes`` / ``sfreqs`` are the summary-figure spectrogram. ``fs`` is
    the rate the run used (after any resampling) and ``t`` the sample
    times at that rate. ``settings`` is the full settings section the run
    used, ``semantics`` the kernel's computation-semantics version.
    """
    stats_table: pd.DataFrame
    peak_in_hist: np.ndarray
    spect: np.ndarray
    stimes: np.ndarray
    sfreqs: np.ndarray
    artifacts: np.ndarray
    artifact_spans: np.ndarray
    fs: float
    t: np.ndarray
    stage_times: np.ndarray
    stage_vals: np.ndarray
    baseline: np.ndarray
    baseline_freqs: np.ndarray
    SOpower_norm_native: np.ndarray
    SOpower_native_t_start: float
    SOphase_unwrapped: np.ndarray
    SOPHs: SOPHsResult
    settings: dict
    semantics: int
    skipped: dict = field(default_factory=dict)
    #: The pass-2 extraction spectrogram (``spect``, ``sfreqs``,
    #: ``stimes``), only with ``run_dynamo(..., keep_spectrogram=True)``.
    pass2_spectrogram: dict | None = None
    fig: Any = None
    timings: dict | None = None
    provenance: Any = None


def _wrap_to_pi(phase: np.ndarray) -> np.ndarray:
    """Wrap radians to [-pi, pi] (NaN stays NaN)."""
    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def _run_settings(settings, fit_param_basis: bool, fit_spline_basis: bool) -> dict:
    """The full settings section for a run: the defaults, ``settings``
    merged over them, and the fit switches applied."""
    section = default_settings()
    if settings is not None:
        section = merge_settings(section, settings)
    else:
        section = copy.deepcopy(section)
    output = section.setdefault("output", {})
    # The kernel fits an axis when its CSV/TIFF or its figure is wanted.
    if not fit_param_basis:
        output["write_paramfits"] = False
        output["write_param_png"] = False
    if not fit_spline_basis:
        output["write_splinefits"] = False
        output["write_spline_png"] = False
    return section


def run_dynamo(
    data: np.ndarray,
    fs: float,
    stage_times: np.ndarray,
    stage_vals: np.ndarray,
    *,
    settings: dict | None = None,
    time_range: tuple[float, float] | None = None,
    fit_param_basis: bool = True,
    fit_spline_basis: bool = True,
    keep_spectrogram: bool = False,
    plot: bool = True,
    verbose: bool = True,
) -> DynamoOutput:
    """Run one recording through DYNAM-O.

    Stage convention (DYNAM-O): 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake.

    ``settings`` is the ``settings`` section of a DYNAM-O settings file, as
    a dict (see :func:`pydynamo.load_settings`); keys it omits take the
    defaults, and None runs with the defaults. ``time_range`` (seconds)
    limits the run to part of the recording. ``fit_param_basis=False`` /
    ``fit_spline_basis=False`` skip those fits. ``keep_spectrogram`` also
    returns the pass-2 extraction spectrogram as ``pass2_spectrogram``
    (over a gigabyte for a full night; no progress lines are printed).
    """
    section = _run_settings(settings, fit_param_basis, fit_spline_basis)
    timings: dict = {}
    t_start = _time.time()

    progress = None
    if verbose and not keep_spectrogram:
        def progress(stage, fraction):
            print(f"[dynamo] {stage}", flush=True)

    out = _kernel.rs().run_dynamo(
        np.ascontiguousarray(np.asarray(data, dtype=float).ravel()),
        float(fs),
        np.ascontiguousarray(np.asarray(stage_times, dtype=float).ravel()),
        np.ascontiguousarray(np.asarray(stage_vals, dtype=float).ravel()),
        json.dumps(section),
        None if time_range is None else (float(time_range[0]), float(time_range[1])),
        bool(keep_spectrogram),
        progress,
    )
    timings["run"] = _time.time() - t_start

    peaks = out["peaks"]
    stats = pd.DataFrame({name: np.asarray(peaks[name]) for name in STATS_COLUMNS})
    stats["SegmentNum"] = stats["SegmentNum"].astype(int)
    stats["PeakStage"] = stats["PeakStage"].astype(int)

    freq_bins = np.asarray(out["freq_bins"])
    sopower_bins = np.asarray(out["sopower_bins"])
    sophase_bins = np.asarray(out["sophase_bins"])
    skipped = {}
    paramfits = {}
    # The kernel fits when the paramfit CSVs or their figures are wanted; a
    # missing fit is "skipped" (declined on degenerate input) only then.
    output = section["output"]
    paramfits_wanted = bool(output.get("write_paramfits") or output.get("write_param_png"))
    for axis, n_feature in (("power", sopower_bins.size), ("phase", sophase_bins.size)):
        axis_out = out["paramfit"][axis]
        paramfits[axis] = paramfit_result_from_kernel(axis_out, freq_bins.size, n_feature)
        if axis_out["skipped"] is not None and paramfits_wanted:
            skipped[f"paramfit_{axis}"] = axis_out["skipped"]
            warnings.warn(
                f"SO-{axis} parametric fit skipped: {axis_out['skipped']}",
                RuntimeWarning,
                stacklevel=2,
            )

    so_phase_unwrapped = np.asarray(out["so_phase_unwrapped"])
    sophs = SOPHsResult(
        SOpower_mat=np.asarray(out["sopower_hist"]),
        SOphase_mat=np.asarray(out["sophase_hist"]),
        SOpower_bins=sopower_bins,
        SOphase_bins=sophase_bins,
        freq_bins=freq_bins,
        SOpower_norm=np.asarray(out["so_power_norm"]),
        SOpower_times=np.asarray(out["so_power_times"]),
        SOphase=_wrap_to_pi(so_phase_unwrapped),
        SOphase_times=np.asarray(out["so_phase_times"]),
        SOpower_paramfit=paramfits["power"],
        SOphase_paramfit=paramfits["phase"],
        SOpower_splinefit=spline_result_from_kernel(
            out["splinefit"]["power"], sopower_bins, freq_bins,
        ),
        SOphase_splinefit=spline_result_from_kernel(
            out["splinefit"]["phase"], sophase_bins, freq_bins,
        ),
    )
    pass2_spectrogram = None
    if keep_spectrogram:
        pass2_spectrogram = {
            "spect": np.asarray(out["spectrogram"]),
            "sfreqs": np.asarray(out["spect_sfreqs"]),
            "stimes": np.asarray(out["spect_stimes"]),
        }

    from pydynamo.io.stamp import current_stamp

    result = DynamoOutput(
        stats_table=stats,
        peak_in_hist=np.asarray(out["peak_in_hist"], dtype=bool),
        spect=np.asarray(out["summary_spectrogram"]),
        stimes=np.asarray(out["summary_stimes"]),
        sfreqs=np.asarray(out["summary_sfreqs"]),
        artifacts=np.asarray(out["artifacts"], dtype=bool),
        artifact_spans=np.asarray(out["artifact_spans"]).reshape(-1, 2),
        fs=float(out["fs"]),
        t=np.asarray(out["t_time_range"]),
        stage_times=np.asarray(out["stage_times"]),
        stage_vals=np.asarray(out["stage_vals"]),
        baseline=np.asarray(out["baseline"]),
        baseline_freqs=np.asarray(out["baseline_freqs"]),
        SOpower_norm_native=np.asarray(out["so_power_norm_native"]),
        SOpower_native_t_start=float(out["so_power_native_t_start"]),
        SOphase_unwrapped=so_phase_unwrapped,
        SOPHs=sophs,
        settings=section,
        semantics=int(out["semantics"]),
        skipped=skipped,
        pass2_spectrogram=pass2_spectrogram,
        timings=timings,
        provenance=current_stamp(),
    )

    if plot:
        from pydynamo.plot import summary_plot
        t_plot = _time.time()
        result.fig = summary_plot(
            result.spect, result.stimes, result.sfreqs, result.artifacts, result.t,
            result.stage_times, result.stage_vals, result.stats_table, sophs,
            time_range=time_range,
            SOpower_norm_method=section["sopower"]["norm_method"],
            hist_peakidx=result.peak_in_hist,
            SOPH_stages=tuple(section["hist"]["soph_stages"]),
        )
        timings["plot_summary"] = _time.time() - t_plot
    timings["total"] = _time.time() - t_start
    if verbose:
        print(f"[dynamo] total wallclock: {timings['total']:.1f}s")
    return result
