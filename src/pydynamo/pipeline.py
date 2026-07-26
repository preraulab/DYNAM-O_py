"""Top-level pipeline: run_dynamo().

Matches DYNAM-O_dev's computeTFPeaks flow:
  1. Multitaper spectrogram pass-1 (1 s window)
  2. Artifacts + baseline subtract → pass-1 spect
  3. Extract pass-1 peaks (watershed + merge + trim), keep labels image
  4. Multitaper spectrogram pass-2 (2 s window) + baseline subtract
  5. Mask pass-2 spect to pass-1 regions
  6. Extract pass-2 peaks on masked spect
  7. Hann-window 1 Hz frequency refinement per peak
  8. Assign per-peak Stage, SO-power, SO-phase
  9. SO-power + SO-phase histograms
 10. Parametric-basis fits
 11. Summary plot
"""

from __future__ import annotations

import time as _time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


@contextmanager
def _timer(store: dict, key: str):
    """Context manager that records wallclock in `store[key]`."""
    t0 = _time.time()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + (_time.time() - t0)

from pydynamo.artifacts import detect_artifacts
from pydynamo.baseline import compute_baseline, subtract_baseline
from pydynamo.defaults import (
    BaselineOpts, DetectionOpts, SOPHOpts, derived_peak_filters,
)
from pydynamo.soph.histogram import so_power_histogram, so_phase_histogram
from pydynamo.soph.paramfit import (
    ParamBasisOpts,
    ParamFitResult,
    fit_param_basis as _fit_param_basis,
)
from pydynamo.soph.sophase import compute_so_phase
from pydynamo.soph.sopower import compute_so_power
from pydynamo.spectrogram import mtm_spectrogram
from pydynamo.tfpeaks.extract import extract_tfpeaks
from pydynamo.tfpeaks.mask import mask_spectrogram
from pydynamo.tfpeaks.refine import refine_peak_frequency


def _crop_baseline_exclude(
    baseline_exclude, *, data_size: int, time_slice: slice,
) -> np.ndarray:
    """Validate and align an explicit exclusion mask to the cropped data."""
    values = np.asarray(baseline_exclude)
    crop_start, crop_stop, crop_step = time_slice.indices(data_size)
    cropped_size = len(range(crop_start, crop_stop, crop_step))

    if values.size == 0:
        return np.zeros(cropped_size, dtype=bool)

    values = values.ravel()
    is_binary_numeric = (
        np.issubdtype(values.dtype, np.number)
        and np.all(np.isin(values, (0, 1)))
    )
    if values.dtype.kind != "b" and not is_binary_numeric:
        raise ValueError("baseline_exclude must be boolean or binary numeric")

    if values.size == data_size:
        values = values[time_slice]
    elif values.size != cropped_size:
        raise ValueError(
            "baseline_exclude must be empty, the full data length "
            f"({data_size}), or the cropped time_range length ({cropped_size})"
        )

    return values.astype(bool, copy=False)


def _resolve_baseline_range(
    baseline_trim, *, stage_times: np.ndarray, stage_vals: np.ndarray,
) -> tuple[float, float]:
    """Resolve MATLAB baseline_trim to an absolute range in seconds.

    A scalar is a symmetric buffer in minutes around the first and last
    non-wake (stages 1--4) staging times, clipped at zero and the final
    staging time. With no non-wake stages, MATLAB uses the full staging span.
    """
    values = np.asarray(baseline_trim)
    if values.size == 0:
        return (float("-inf"), float("inf"))
    if not np.issubdtype(values.dtype, np.number) or values.dtype.kind == "b":
        raise ValueError("baseline_trim must be numeric")

    values = values.astype(float, copy=False).ravel()
    if values.size == 1:
        nonwake_times = stage_times[np.isin(stage_vals, (1, 2, 3, 4))]
        if nonwake_times.size == 0:
            return (0.0, float(stage_times.max()))
        buffer_seconds = values[0] * 60.0
        return (
            float(max(nonwake_times.min() - buffer_seconds, 0.0)),
            float(min(
                nonwake_times.max() + buffer_seconds,
                stage_times.max(),
            )),
        )
    if values.size == 2:
        return (float(values[0]), float(values[1]))
    raise ValueError(
        "baseline_trim must contain zero, one, or two numeric values"
    )


@dataclass
class SOPHsResult:
    SOpower_mat: np.ndarray
    SOphase_mat: np.ndarray
    SOpower_bins: np.ndarray
    SOphase_bins: np.ndarray
    freq_bins: np.ndarray
    SOpower_TIB: np.ndarray
    SOphase_TIB: np.ndarray
    peak_at_freq_SOpower: np.ndarray
    peak_at_freq_SOphase: np.ndarray
    SOpower_norm: np.ndarray
    SOpower_times: np.ndarray
    SOphase: np.ndarray
    SOphase_times: np.ndarray
    SOpower_paramfit: ParamFitResult | None = None
    SOphase_paramfit: ParamFitResult | None = None


@dataclass
class DynamoOutput:
    stats_table: pd.DataFrame
    spect: np.ndarray              # pass-2 spectrogram (for plotting)
    stimes: np.ndarray
    sfreqs: np.ndarray
    artifacts: np.ndarray
    SOPHs: SOPHsResult
    fig: Any = None
    timings: dict | None = None    # matches MATLAB `timings` struct fields


def run_dynamo(
    data: np.ndarray,
    fs: float,
    stage_times: np.ndarray,
    stage_vals: np.ndarray,
    *,
    detection_opts: DetectionOpts | None = None,
    baseline_opts: BaselineOpts | None = None,
    soph_opts: SOPHOpts | None = None,
    param_basis_power_opts: ParamBasisOpts | None = None,
    param_basis_phase_opts: ParamBasisOpts | None = None,
    fit_param_basis: bool = True,
    time_range: tuple[float, float] | None = None,
    plot: bool = True,
    verbose: bool = True,
    # Scalar overrides, kept for backward compatibility. Each is applied on
    # top of the corresponding option object when it is not None.
    merge_thresh: float | None = None,
    trim_vol: float | None = None,
    seg_time: float | None = None,
    soph_stages: tuple[int, ...] | None = None,
    min_time_in_bin: float | None = None,
    min_peak_at_freq: int | None = None,
    double_watershed: bool | None = None,
    refinement: bool | None = None,
) -> DynamoOutput:
    """Run the pydynamo pipeline.

    Stage convention (DYNAM-O): 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake.

    Options mirror the MATLAB `runDYNAMO` option structs. Parametric fitting
    runs by default; pass `fit_param_basis=False` to return only the raw SOPH
    histograms.
    """
    det = detection_opts if detection_opts is not None else DetectionOpts()
    base = baseline_opts if baseline_opts is not None else BaselineOpts()
    soph = soph_opts if soph_opts is not None else SOPHOpts()

    if merge_thresh is not None:
        det = replace(det, merge_thresh=merge_thresh)
    if trim_vol is not None:
        det = replace(det, trim_vol=trim_vol)
    if seg_time is not None:
        det = replace(det, seg_time=seg_time)
    if double_watershed is not None:
        det = replace(det, double_watershed=double_watershed)
    if refinement is not None:
        det = replace(det, refinement=refinement)
    if soph_stages is not None:
        soph = replace(soph, SOPH_stages=tuple(soph_stages))
    if min_time_in_bin is not None:
        soph = replace(soph, SOpower_min_time_in_bin=min_time_in_bin)
    if min_peak_at_freq is not None:
        soph = replace(soph, SOphase_min_peak_at_freq=min_peak_at_freq)

    pk = derived_peak_filters(det)

    data = np.ascontiguousarray(np.asarray(data, dtype=np.float64).ravel())
    fs = float(fs)
    stage_times = np.asarray(stage_times, dtype=float).ravel()
    stage_vals = np.asarray(stage_vals, dtype=float).ravel()

    if time_range is None:
        valid_stage_inds = np.flatnonzero((stage_vals > 0) & (stage_vals < 6))
        if valid_stage_inds.size == 0:
            raise ValueError("No valid stages found.")
        time_range = (
            float(stage_times[valid_stage_inds[0]]),
            float(stage_times[valid_stage_inds[-1]]),
        )

    i0 = int(round(time_range[0] * fs))
    i1 = int(round(time_range[1] * fs))
    time_slice = slice(i0, i1 + 1)
    data_tr = data[time_slice]
    t_tr = np.arange(i0, i1 + 1) / fs
    explicit_baseline_exclude = _crop_baseline_exclude(
        base.baseline_exclude, data_size=data.size, time_slice=time_slice,
    )
    baseline_range = _resolve_baseline_range(
        base.baseline_trim, stage_times=stage_times, stage_vals=stage_vals,
    )

    if verbose:
        print(f"[dynamo] {data_tr.size} samples ({data_tr.size/fs:.0f}s) at {fs} Hz")

    timings: dict[str, float] = {}
    t_start = _time.time()

    # ---- Artifacts (once, on raw data) ----
    if verbose: print("[dynamo] artifacts...")
    with _timer(timings, "artifact"):
        artifacts = detect_artifacts(data_tr, fs)

    # Stage mask for baseline exclusion
    stage_at_data = interp1d(
        stage_times, stage_vals, kind="previous",
        bounds_error=False, fill_value=0.0,
    )(t_tr)
    stage_exclude = ~np.isin(stage_at_data, base.baseline_stages)
    baseline_exclude = (
        artifacts | stage_exclude | explicit_baseline_exclude
    )

    # ---- Pass 1: 1 s window ----
    if verbose: print("[dynamo] pass-1 spectrogram (1 s window)...")
    with _timer(timings, "spect_pass1"):
        spect1, stimes1_rel, sfreqs = mtm_spectrogram(
            data_tr, fs,
            freq_range=det.mtm_freq_range, taper_params=det.mtm_taper_params,
            window_params=(det.mtm_window_length_1, det.mtm_window_stepsize),
            dsfreqs=det.mtm_dsfreqs,
        )
    stimes1 = stimes1_rel + t_tr[0]
    with _timer(timings, "baseline_pass1"):
        baseline1 = compute_baseline(spect1, stimes1, t_tr, baseline_exclude,
                                      baseline_range=baseline_range,
                                      baseline_ptile=base.baseline_ptile)
        spect1_norm = subtract_baseline(spect1, baseline1)

    if verbose: print("[dynamo] pass-1 extract...")
    with _timer(timings, "extract_pass1"):
        stats1, labels1 = extract_tfpeaks(
            spect1_norm, stimes1, sfreqs,
            seg_time=det.seg_time,
            return_labels=True, return_raw_labels=True,
            num_tapers_for_prom=int(det.mtm_taper_params[1]),
            downsample=det.downsample_spect, merge_thresh=det.merge_thresh,
            max_merges=det.max_merges, trim_vol=det.trim_vol,
            dur_min=pk["dur_min"], dur_max=det.dur_max,
            bw_min=pk["bw_min_pass1"], bw_max=det.bw_max,
        )
    if verbose: print(f"[dynamo]   pass-1: {len(stats1)} peaks")

    # ---- Pass 2 (optional): 2 s window, masked by pass-1 regions ----
    if det.double_watershed and not stats1.empty:
        if verbose: print("[dynamo] pass-2 spectrogram (2 s window)...")
        with _timer(timings, "spect_pass2"):
            spect2, stimes2_rel, sfreqs2 = mtm_spectrogram(
                data_tr, fs,
                freq_range=det.mtm_freq_range, taper_params=det.mtm_taper_params,
                window_params=(det.mtm_window_length_2, det.mtm_window_stepsize),
                dsfreqs=det.mtm_dsfreqs,
            )
        stimes2 = stimes2_rel + t_tr[0]
        assert sfreqs2.shape == sfreqs.shape, \
            "pass-1/pass-2 sfreqs differ (nfft mismatch)"
        with _timer(timings, "baseline_pass2"):
            # computeTFPeaks.m: with reuse_baseline (the default) pass 2 keeps
            # the pass-1 baseline rather than recomputing one from the pass-2
            # spectrogram.
            baseline2 = (baseline1 if det.reuse_baseline else
                         compute_baseline(spect2, stimes2, t_tr,
                                          baseline_exclude,
                                          baseline_range=baseline_range,
                                          baseline_ptile=base.baseline_ptile))
            spect2_norm = subtract_baseline(spect2, baseline2)
            if verbose: print("[dynamo] masking pass-2 spect with pass-1 regions...")
            spect2_masked = mask_spectrogram(spect2_norm, stimes2, labels1, stimes1)

        if verbose: print("[dynamo] pass-2 extract on masked spectrogram...")
        # MATLAB-exact params (computeTFPeaks.m:342): pass-2 uses dur_min=0.5
        # from pass-1 (line 342 ignores pass-2's own dur_min return), and
        # bw_min = (TW/T) = 2/2 = 1.0 Hz (spectral resolution / 2).
        with _timer(timings, "extract_pass2"):
            stats = extract_tfpeaks(
                spect2_masked, stimes2, sfreqs2,
                seg_time=det.seg_time,
                num_tapers_for_prom=int(det.mtm_taper_params[1]),
                downsample=det.downsample_spect, merge_thresh=det.merge_thresh,
                max_merges=det.max_merges, trim_vol=det.trim_vol,
                dur_min=pk["dur_min"], dur_max=det.dur_max,
                bw_min=pk["bw_min_pass2"], bw_max=det.bw_max,
            )
        spect_for_plot = spect2
        stimes_for_plot = stimes2
        if verbose: print(f"[dynamo]   pass-2: {len(stats)} peaks")
    else:
        stats = stats1
        spect_for_plot = spect1
        stimes_for_plot = stimes1

    # ---- Hann refinement ----
    if det.refinement and not stats.empty:
        if verbose: print("[dynamo] Hann frequency refinement...")
        with _timer(timings, "refine"):
            stats = refine_peak_frequency(
                stats, data_tr, fs, t=t_tr,
                freq_range=det.mtm_freq_range,
                window_size=det.refine_window_size,
                dsfreqs=det.refine_dsfreqs,
                refine_method=det.refine_method,
                remove_edge_peaks=det.refine_remove_edge_peaks,
            )
        if verbose: print(f"[dynamo]   after refinement: {len(stats)} peaks")

    # ---- Per-peak stage ----
    with _timer(timings, "peak_stage"):
        if not stats.empty:
            stage_interp = interp1d(
                stage_times, stage_vals, kind="previous",
                bounds_error=False, fill_value=0.0,
            )
            stats["PeakStage"] = stage_interp(stats["PeakTime"].to_numpy())
            art_at_peak = interp1d(
                t_tr, artifacts.astype(float), kind="nearest",
                bounds_error=False, fill_value=0.0,
            )(stats["PeakTime"].to_numpy()) >= 0.5
            stats.loc[art_at_peak, "PeakStage"] = 6

    # ---- SO-power / SO-phase timeseries ----
    if verbose: print("[dynamo] SO-power...")
    with _timer(timings, "peak_sopower"):
        SOpower_norm, SOpower_times, SOpower_stages, _, _ = compute_so_power(
            data_tr, fs,
            stage_times=stage_times, stage_vals=stage_vals,
            eeg_times=t_tr, time_range=time_range,
            isexcluded=artifacts,
            SO_freqrange=soph.SO_freqrange, tapers=soph.SOpower_tapers,
            window_params=soph.SOpower_window_params,
            SOpower_outlier_threshold=soph.SOpower_outlier_threshold,
            norm_method=soph.SOpower_norm_method,
            retain_Fs=soph.SOpower_retain_Fs,
        )
        if not stats.empty:
            xp = np.concatenate((
                [SOpower_times[0] - 1], SOpower_times,
                [SOpower_times[-1] + 1],
            ))
            fp = np.concatenate((
                [SOpower_norm[0]], SOpower_norm, [SOpower_norm[-1]],
            ))
            stats["SOpower"] = np.interp(stats["PeakTime"].to_numpy(), xp, fp)

    if verbose: print("[dynamo] SO-phase...")
    with _timer(timings, "peak_sophase"):
        SOphase_unwrapped, SOphase_times, SOphase_stages, _ = compute_so_phase(
            data_tr, fs,
            stage_times=stage_times, stage_vals=stage_vals,
            eeg_times=t_tr, isexcluded=artifacts,
            SO_freqrange=soph.SO_freqrange,
        )
        if not stats.empty:
            peak_phase_unwrapped = np.interp(
                stats["PeakTime"].to_numpy(), SOphase_times, SOphase_unwrapped,
            )
            stats["SOphase"] = (peak_phase_unwrapped + np.pi) % (2 * np.pi) - np.pi

    # ---- SOPH histograms ----
    # Match SOpowerphaseHistogram.m: the SO-power and SO-phase grids can
    # represent the same excluded interval with different NaN footprints.
    # Propagate SO-power invalidity to the phase grid before either histogram
    # selects peaks so both histograms use the same event population.
    sopower_dt = SOpower_times[1] - SOpower_times[0]
    power_at_phase_times = np.interp(
        SOphase_times,
        np.concatenate((
            [SOpower_times[0] - sopower_dt], SOpower_times,
            [SOpower_times[-1] + sopower_dt],
        )),
        np.concatenate((
            [SOpower_norm[0]], SOpower_norm, [SOpower_norm[-1]],
        )),
        left=np.nan,
        right=np.nan,
    )
    SOphase_hist = SOphase_unwrapped.copy()
    SOphase_hist[np.isnan(power_at_phase_times)] = np.nan

    if verbose: print("[dynamo] SO-power histogram...")
    pf = stats["PeakFrequency"].to_numpy() if not stats.empty else np.array([])
    pt = stats["PeakTime"].to_numpy() if not stats.empty else np.array([])
    ps = stats["PeakStage"].to_numpy() if not stats.empty else np.array([])
    with _timer(timings, "soph_sopower_hist"):
        sopow = so_power_histogram(
            pf, pt, ps, SOpower_norm, SOpower_times, SOpower_stages,
            time_range=time_range, soph_stages=soph.SOPH_stages,
            freq_range=soph.freq_range, freq_binsizestep=soph.freq_binsizestep,
            so_range=soph.SOpower_range, so_binsizestep=soph.SOpower_binsizestep,
            min_time_in_bin=soph.SOpower_min_time_in_bin,
            compute_rate=soph.compute_rate, norm_dim=0,
        )

    if verbose: print("[dynamo] SO-phase histogram...")
    with _timer(timings, "soph_sophase_hist"):
        sopha = so_phase_histogram(
            pf, pt, ps, SOphase_hist, SOphase_times, SOphase_stages,
            time_range=time_range, soph_stages=soph.SOPH_stages,
            freq_range=soph.freq_range, freq_binsizestep=soph.freq_binsizestep,
            so_range=soph.SOphase_range,
            so_binsizestep=soph.SOphase_binsizestep,
            min_peak_at_freq=soph.SOphase_min_peak_at_freq,
            compute_rate=soph.compute_rate, norm_dim=soph.SOphase_norm_dim,
        )

    # SOpowerphaseHistogram.m requires one shared TF-peak population.
    power_peak_selection = np.asarray(
        sopow["peak_selection_inds"], dtype=bool,
    )
    phase_peak_selection = np.asarray(
        sopha["peak_selection_inds"], dtype=bool,
    )
    assert np.array_equal(power_peak_selection, phase_peak_selection), (
        "SO-power and SO-phase histograms included different TF peaks."
    )
    peak_selection_inds = power_peak_selection
    stats_table_soph = stats.loc[peak_selection_inds]

    sophs = SOPHsResult(
        SOpower_mat=sopow["c_mat"], SOphase_mat=sopha["c_mat"],
        SOpower_bins=sopow["c_cbins"], SOphase_bins=sopha["c_cbins"],
        freq_bins=sopow["freq_cbins"],
        SOpower_TIB=sopow["time_in_bin"], SOphase_TIB=sopha["time_in_bin"],
        peak_at_freq_SOpower=sopow["peak_at_freq"],
        peak_at_freq_SOphase=sopha["peak_at_freq"],
        SOpower_norm=SOpower_norm, SOpower_times=SOpower_times,
        SOphase=(SOphase_unwrapped + np.pi) % (2 * np.pi) - np.pi,
        SOphase_times=SOphase_times,
    )

    if fit_param_basis:
        if verbose:
            print("[dynamo] fitting parametric basis...")
        with _timer(timings, "fit_param_basis"):
            # Match fitParamBasis.m: phase runs first, and one failed fit does
            # not discard a successful fit on the other axis.
            try:
                sophs.SOphase_paramfit = _fit_param_basis(
                    sophs.SOphase_mat, sophs.SOphase_bins, sophs.freq_bins,
                    opts=param_basis_phase_opts, kind="phase",
                    stats_table_soph=stats_table_soph,
                )
            except Exception as exc:
                warnings.warn(
                    f"SO-phase parametric fit failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            try:
                sophs.SOpower_paramfit = _fit_param_basis(
                    sophs.SOpower_mat, sophs.SOpower_bins, sophs.freq_bins,
                    opts=param_basis_power_opts, kind="power",
                    stats_table_soph=stats_table_soph,
                    phase_model_soph=(
                        sophs.SOphase_paramfit.model_soph
                        if sophs.SOphase_paramfit is not None else None
                    ),
                    phase_bins=sophs.SOphase_bins,
                )
            except Exception as exc:
                warnings.warn(
                    f"SO-power parametric fit failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    fig = None
    if plot:
        from pydynamo.plot import summary_plot
        if verbose: print("[dynamo] plotting...")
        with _timer(timings, "plot_summary"):
            fig = summary_plot(
                spect_for_plot, stimes_for_plot, sfreqs, artifacts, t_tr,
                stage_times, stage_vals, stats, sophs,
                data=data_tr, fs=fs, time_range=time_range,
                freq_limits=(2.0, 25.0), mtm_freq_range=(2.0, 25.0),
                hist_peakidx=peak_selection_inds,
                SOPH_stages=soph.SOPH_stages,
            )
    timings["total"] = _time.time() - t_start

    if verbose:
        print(f"\n[dynamo] total wallclock: {timings['total']:.1f}s")

    return DynamoOutput(
        stats_table=stats, spect=spect_for_plot, stimes=stimes_for_plot,
        sfreqs=sfreqs, artifacts=artifacts, SOPHs=sophs, fig=fig,
        timings=timings,
    )
