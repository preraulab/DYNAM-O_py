"""Frozen default option sets mirroring the DYNAM-O 'default' preset.

Sourced from:
- toolbox/TFpeak_functions/option_sets/detection_opts.m
- toolbox/TFpeak_functions/option_sets/baseline_opts.m
- toolbox/SOpowphase_functions/SOpowerphasehist_opts.m

Only the 'default' preset is modelled here. No 'stokes_2023' or 'precision'.

These are consumed by `run_dynamo`; when a field's value is changed here it
changes what the pipeline actually does. Keep each field's value equal to the
`addOptional`/`addParameter` default in the MATLAB file named above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Sequence


@dataclass(frozen=True)
class DetectionOpts:
    # Multitaper spectrogram
    mtm_freq_range: tuple = (0.0, 30.0)
    mtm_taper_params: tuple = (2, 3)           # (time-BW, num_tapers)
    mtm_window_length_1: float = 1.0            # pass 1 (temporal resolution)
    mtm_window_length_2: float = 2.0            # pass 2 (spectral resolution)
    mtm_window_stepsize: float = 0.05
    mtm_dsfreqs: float = 0.1

    # Watershed + merge + trim
    double_watershed: bool = True
    downsample_spect: tuple = (2, 2)            # (time_factor, freq_factor)
    seg_time: float = 30.0                      # segment duration (s)
    merge_thresh: float = 11.0
    max_merges: float = float("inf")
    trim_vol: float = 0.8

    # Peak filters.
    #
    # MATLAB does not take dur_min/bw_min as settings — computeSpectrogram
    # (computeTFPeaks.m:462-466) derives them from the taper bandwidth:
    #     dur_min = window_length / 2
    #     bw_min  = (NW / window_length * 2) / 2 = NW / window_length
    # and pass 2 discards its own dur_min, reusing pass 1's
    # (computeTFPeaks.m:368 drops the 4th output). Leave these None to derive
    # the same way; set them only to deliberately override.
    dur_min: float | None = None
    dur_max: float = 5.0
    bw_min_pass1: float | None = None
    bw_min_pass2: float | None = None
    bw_max: float = 15.0

    # Reuse the pass-1 baseline for pass 2 instead of recomputing it
    # (detection_opts.m:74, default true). The two passes share the NFFT and
    # time grid; pass 2's extra temporal smoothing only shifts the sub-2 Hz
    # baseline, which bw_min >= 2 filters out anyway.
    reuse_baseline: bool = True

    # Refinement (Hann-window peak-frequency refine)
    refinement: bool = True
    refine_window_size: float = 4.0
    refine_dsfreqs: float = 0.05
    refine_method: str = "spline_interp"
    refine_remove_edge_peaks: bool = True


def derived_peak_filters(det: DetectionOpts) -> dict:
    """Resolve the taper-derived dur_min / bw_min values.

    Port of computeTFPeaks.m:462-466. Explicit values on `det` win; None means
    derive. Returns ``{"dur_min", "bw_min_pass1", "bw_min_pass2"}``.
    """
    nw = float(det.mtm_taper_params[0])
    t1 = float(det.mtm_window_length_1)
    t2 = float(det.mtm_window_length_2)
    return {
        "dur_min": t1 / 2.0 if det.dur_min is None else float(det.dur_min),
        "bw_min_pass1": (nw / t1 if det.bw_min_pass1 is None
                         else float(det.bw_min_pass1)),
        "bw_min_pass2": (nw / t2 if det.bw_min_pass2 is None
                         else float(det.bw_min_pass2)),
    }


@dataclass(frozen=True)
class BaselineOpts:
    baseline_stages: tuple = (1, 2, 3, 4, 5)
    baseline_exclude: Sequence[float] = field(default_factory=tuple)
    baseline_ptile: float = 2.0
    baseline_trim: tuple = (float("-inf"), float("inf"))


@dataclass(frozen=True)
class SOPHOpts:
    # Frequency axis (applies to both SO-power and SO-phase histograms).
    # NOTE: this is the *histogram* frequency range, not the spectrogram's —
    # DetectionOpts.mtm_freq_range stays (0, 30).
    freq_range: tuple = (2.0, 18.0)
    freq_binsizestep: tuple = (1.0, 0.2)        # (bin size, step) in Hz

    # Slow-oscillation band
    SO_freqrange: tuple = (0.3, 1.5)

    # SO-power computation
    SOpower_norm_method: str = "p2shift1234"
    SOpower_min_time_in_bin: float = 10.0       # minutes
    SOpower_outlier_threshold: float = 3.0
    SOpower_retain_Fs: bool = True
    SOpower_tapers: tuple = (5, 9)
    SOpower_window_params: tuple = (5.0, 0.5)   # (window length s, step s)

    # SO-power histogram. Fixed (not per-subject adaptive) so histograms are
    # comparable across subjects out of the box — see the comment above
    # `SOpower_range` in SOpowerphasehist_opts.m. Set either to None to opt
    # back into adaptive bins (range = min/max of normalized SO-power,
    # width = range/10, step = range/100).
    SOpower_range: tuple | None = (-5.0, 25.0)          # dB
    SOpower_binsizestep: tuple | None = (2.5, 0.25)     # (bin size, step) dB

    # SO-phase histogram
    SOphase_range: tuple = (-pi, pi)
    SOphase_binsizestep: tuple = (2 * pi / 5, 2 * pi / 100)
    SOphase_min_peak_at_freq: int = 0
    SOphase_norm_dim: int = 1

    # Whether the per-peak `SOpower` column's shift-normalization percentile is
    # restricted to the stages named in `SOpower_norm_method` (the "1234" of
    # p2shift1234), as the histogram's is.
    #
    # MATLAB says False by accident: computePeakSOpower.m calls computeSOpower
    # without stage_times/stage_vals, so SOpower_stages degrades to scalar
    # `true`, the stage mask drops out, and the percentile is taken over every
    # in-range sample. SOpowerHistogram.m *does* pass the stages. So within one
    # MATLAB run the stats_table SOpower column and the SOPH SO-power axis use
    # different normalizations — on the validation night they differ by 0.21 dB,
    # meaning a peak's reported SOpower does not match the bin it was placed in.
    #
    # True (the default here) keeps the column and the histogram consistent.
    # Set False to reproduce MATLAB's stats_table column bit-for-bit.
    SOpower_peak_shift_uses_stages: bool = True

    # Stages included in histograms (DYNAM-O convention: 1=N3 ... 5=Wake)
    SOPH_stages: tuple = (1, 2, 3)              # NREM only

    compute_rate: bool = True                   # peaks/min (not raw count)
