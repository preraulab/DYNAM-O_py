"""Compute per-region stats (Area, BoundingBox, WeightedCentroid, etc.)
and pack into a pandas DataFrame — port of computePeakStatsTable.m.

MATLAB uses regionprops + pandas-like table. We use scikit-image's
regionprops_table with intensity image.

NOTE: DYNAM-O's MATLAB data is (F, T) with rows = freq, cols = time.
- BoundingBox(:, 1) = top-left time  → stimes axis (cols, aka x)
- BoundingBox(:, 2) = top-left freq  → sfreqs axis (rows, aka y)
- WeightedCentroid(:, 1) = time centroid (cols)
- WeightedCentroid(:, 2) = freq centroid (rows)

scikit-image's regionprops returns centroids as (row, col); remember to
swap when mapping to (time, freq).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from skimage.measure import find_contours, regionprops


def compute_peak_stats(
    labels: np.ndarray,
    data: np.ndarray,
    stimes: np.ndarray,
    sfreqs: np.ndarray,
    segment_num: int = 1,
) -> pd.DataFrame:
    """Build a stats DataFrame with one row per labelled region.

    Columns (subset of DYNAM-O):
        PeakTime, PeakFrequency, Height, Area, Duration, Bandwidth, Volume,
        Peakiness (= log10(Area * Height / Volume)),
        BoundingBox (4-tuple: (time_tl, freq_tl, width_s, height_Hz)),
        Boundaries (Nx2 ndarray: column 0 = time (s), column 1 = freq (Hz)),
        HeightData (1D ndarray of per-pixel intensities inside the region),
        SegmentNum
    """
    labels = np.asarray(labels, dtype=np.int64)
    data = np.asarray(data, dtype=np.float64)
    stimes = np.asarray(stimes, dtype=np.float64)
    sfreqs = np.asarray(sfreqs, dtype=np.float64)
    assert labels.shape == data.shape

    dt = float(stimes[1] - stimes[0])
    df = float(sfreqs[1] - sfreqs[0])
    seg_startx = float(stimes[0])
    seg_starty = float(sfreqs[0])

    # skimage regionprops with intensity image. Relabel to contiguous 1..K
    # so regionprops doesn't silently skip labels.
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    if unique_labels.size == 0:
        return pd.DataFrame(
            columns=["PeakTime", "PeakFrequency", "Height", "Area",
                     "Duration", "Bandwidth", "Volume", "Peakiness",
                     "BoundingBox", "Boundaries", "HeightData", "SegmentNum"]
        )

    # Treat NaN pixels as out-of-region. MATLAB does this explicitly.
    data_clean = np.where(np.isnan(data), 0.0, data)
    safe_labels = labels.copy()
    safe_labels[np.isnan(data)] = 0

    props = regionprops(safe_labels, intensity_image=data_clean)

    rows = []
    for p in props:
        # BoundingBox (skimage): (min_row, min_col, max_row, max_col)
        r0, c0, r1, c1 = p.bbox
        bb_time = (c0) * dt + seg_startx
        bb_freq = (r0) * df + seg_starty
        bb_w = (c1 - c0) * dt
        bb_h = (r1 - r0) * df

        # WeightedCentroid (skimage): (row, col) in pixel coords, zero-based.
        # MATLAB's WeightedCentroid is 1-based so subtracting 0.5 aligns; we
        # use 0-based but offset by the segment start identically.
        wc_r, wc_c = p.weighted_centroid
        peak_time = wc_c * dt + seg_startx
        peak_freq = wc_r * df + seg_starty

        vals = p.image_intensity[p.image]
        height_data = np.ascontiguousarray(vals, dtype=np.float64)
        area = p.area * dt * df
        volume = float(vals.sum()) * dt * df
        height = float(vals.max() - vals.min())

        # Boundaries: trace the perimeter of this region's mask and convert
        # (row, col) pixel coords to (time, freq) world coords. Pad with a
        # 1-px zero border so find_contours has a level crossing even when
        # the cropped mask is fully filled, then offset by -1 to undo the pad.
        padded = np.pad(p.image.astype(np.float64), 1)
        contours_pix = find_contours(padded, 0.5)
        if contours_pix:
            pieces = []
            for cnt in contours_pix:
                rr = cnt[:, 0] - 1 + r0
                cc = cnt[:, 1] - 1 + c0
                pieces.append(np.column_stack([cc * dt + seg_startx,
                                               rr * df + seg_starty]))
            boundaries = np.vstack(pieces)
        else:
            boundaries = np.empty((0, 2), dtype=np.float64)
        # Peakiness = log10(Area * Height / Volume). Degenerate cases:
        #   volume == 0      → ratio is +∞ or NaN; mark NaN.
        #   height == 0      → ratio is 0; log10(0) = -∞ (kept as sentinel).
        #   ratio negative   → shouldn't happen on positive spectrograms; NaN.
        if volume == 0.0:
            peakiness = float("nan")
        else:
            ratio = area * height / volume
            peakiness = math.log10(ratio) if ratio > 0.0 else (
                float("-inf") if ratio == 0.0 else float("nan")
            )

        rows.append({
            "Label": p.label,
            "PeakTime": peak_time,
            "PeakFrequency": peak_freq,
            "Height": height,
            "Area": area,
            "Duration": bb_w,
            "Bandwidth": bb_h,
            "Volume": volume,
            "Peakiness": peakiness,
            "BoundingBox": (bb_time, bb_freq, bb_w, bb_h),
            "Boundaries": boundaries,
            "HeightData": height_data,
            "SegmentNum": int(segment_num),
        })

    return pd.DataFrame(rows)
