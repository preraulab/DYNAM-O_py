"""Percentile-baseline subtraction — port of computeTFPeaks.m:computeBaseline.

baseline_exclude is the OR of:
    - artifact mask (from detect_artifacts, interpolated onto stimes)
    - samples in stages NOT in baseline_stages
    - any explicit user-supplied baseline_exclude

The baseline is the Nth-percentile power per frequency bin, computed over
the valid (non-excluded, in-range) spectrogram columns. Zero-valued pixels
in the spectrogram are treated as missing data (NaN) to avoid them biasing
the percentile.

The MATLAB pipeline divides the spectrogram by the broadcast baseline
(equivalent to dB-domain subtraction after 10*log10).
"""

from __future__ import annotations

import numpy as np


def compute_baseline(
    spect: np.ndarray,
    stimes: np.ndarray,
    t_data: np.ndarray,
    baseline_exclude: np.ndarray,
    baseline_range: tuple[float, float] = (float("-inf"), float("inf")),
    baseline_ptile: float = 2.0,
) -> np.ndarray:
    """Compute per-frequency baseline spectrum.

    Parameters
    ----------
    spect : (F, T) array — spectrogram
    stimes : (T,) array — spectrogram window-center times (s)
    t_data : (N,) array — data sample timestamps (s), same length as
             baseline_exclude
    baseline_exclude : (N,) bool — samples to exclude from baseline
    baseline_range : (start_s, end_s) — further restriction on stimes
    baseline_ptile : percentile (0-100)

    Returns
    -------
    baseline : (F, 1) array
    """
    spect = np.asarray(spect)
    stimes = np.asarray(stimes).ravel()
    t_data = np.asarray(t_data).ravel()
    baseline_exclude = np.asarray(baseline_exclude, dtype=bool).ravel()

    # Interpolate baseline_exclude onto stimes (nearest).
    # MATLAB: interp1(t_time_range, single(baseline_exclude), stimes, 'nearest')
    # Nearest-neighbor: find index in t_data closest to each stime.
    # Since t_data is uniformly sampled, binary search works.
    idx = np.searchsorted(t_data, stimes)
    idx = np.clip(idx, 0, t_data.size - 1)
    # Adjust for nearest (searchsorted gives the right-neighbor; compare the
    # left side too)
    left = np.clip(idx - 1, 0, t_data.size - 1)
    use_left = np.abs(t_data[left] - stimes) < np.abs(t_data[idx] - stimes)
    idx = np.where(use_left, left, idx)
    exclude_stimes = baseline_exclude[idx]

    in_range = (stimes >= baseline_range[0]) & (stimes <= baseline_range[1])
    valid = (~exclude_stimes) & in_range
    if not valid.any():
        raise ValueError(
            "No valid baseline time bins remain after applying artifacts, "
            "stage filtering, and baseline_range."
        )

    spect_bl = spect[:, valid].astype(np.float64, copy=True)
    spect_bl[spect_bl == 0] = np.nan

    # MATLAB `prctile` uses Hyndman-Fan method #5 ("hazen"). numpy's default
    # is "linear" (method #7) — which gives ~0.05% mismatch vs MATLAB on the
    # 2nd percentile over ~100k samples. Use hazen to match exactly.
    baseline = np.nanpercentile(spect_bl, baseline_ptile, axis=1,
                                keepdims=True, method="hazen")
    return baseline


def subtract_baseline(spect: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Divide-in-linear == subtract-in-dB. Matches MATLAB::

        spect_baseline = spect ./ baseline  % element-wise broadcast
    """
    return spect / baseline
