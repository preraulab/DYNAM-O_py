"""Watershed seeding over the histogram — port of extracthistpeaks.m.

This runs the *same* watershed / merge / trim machinery as TF-peak extraction,
but over a SOPH image rather than a spectrogram, to produce initial mode
guesses for the parametric fit. The axes are (y = frequency, x = SO-feature),
so "Duration" here is an extent along the SO-power or SO-phase axis, not
seconds.

No smoothing happens here. The phase path smooths *before* calling
(param_basis_phase.m:172-191); the power path does not smooth at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure

from pydynamo.soph.paramfit.basis import SQRT2
from pydynamo.soph.paramfit.matlab_compat import prctile
from pydynamo.tfpeaks.extract import _HAS_FUSED, watershed
from pydynamo.tfpeaks.merge import (
    _compute_edge_weight, build_rag_from_watershed, merge_segment,
)
from pydynamo.tfpeaks.trim import trim_regions

# Hard-coded in extracthistpeaks.m and not settable by the caller.
_CONN_WSHED = 8
_CONN_TRIM = 8
_MAX_MERGES = float("inf")


def _run_watershed(img):
    """runWatershed.m: negate, watershed, zero the NaNs, relabel 8-connected.

    The relabel is not cosmetic — MATLAB's `bwlabel` uses 8-connectivity, so
    two basins touching only diagonally across a 1-pixel watershed line get
    fused into one region. Reproducing that needs a 3x3 structuring element.
    """
    nan_mask = ~np.isfinite(img)
    data = np.where(nan_mask, 0.0, img)

    labels = watershed(-data)
    labels = np.asarray(labels, dtype=np.int64)
    labels[nan_mask] = 0

    relabeled, _ = ndimage.label(labels != 0, structure=np.ones((3, 3), dtype=int))
    return relabeled.astype(np.int64)


def _dynamic_merge_threshold(labels, img):
    """merge_thresh = prctile(initial edge weights, 95) — mergeWshedSegment.m.

    Taken when the caller passes NaN, which is the power-axis default
    (`watershed_params[0]`). Uses MATLAB's percentile convention because the
    threshold directly decides how many seed regions survive.
    """
    rag = build_rag_from_watershed(labels)
    data_flat = np.asarray(img, dtype=float).ravel(order="C")
    weights = []
    for u, v in rag.edges():
        w = _compute_edge_weight(rag, u, v, data_flat)
        if np.isfinite(w):
            weights.append(w)
    if not weights:
        return 0.0
    return float(prctile(np.asarray(weights), 95.0))


def _region_stats(labels, img, x, y):
    """computePeakStatsTable.m, restricted to the columns paramfit consumes."""
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    props = pd.DataFrame(measure.regionprops_table(
        labels, np.where(np.isfinite(img), img, 0.0),
        properties=("label", "area", "centroid_weighted", "bbox",
                    "intensity_min", "intensity_max"),
    ))
    if props.empty:
        return props

    minr, minc = props["bbox-0"], props["bbox-1"]
    maxr, maxc = props["bbox-2"], props["bbox-3"]

    out = pd.DataFrame({
        "label": props["label"],
        # skimage's 0-based centroid equals MATLAB's (index - 1), so the
        # MATLAB `-1` offset is already applied.
        "PeakFrequency": props["centroid_weighted-0"] * dy + float(y[0]),
        "SOFeature": props["centroid_weighted-1"] * dx + float(x[0]),
        "Height": props["intensity_max"] - props["intensity_min"],
        # Extent along the SO-feature axis. Named "Duration" to match the
        # MATLAB column that param_basis_* reads; it is not seconds.
        "Duration": (maxc - minc) * dx,
        "Bandwidth": (maxr - minr) * dy,
        "Area": props["area"] * dx * dy,
    })
    return out


def _extract_hist_peaks_fused(img, x, y, merge_thresh, dur_min, bw_min,
                              height_min, trim_vol):
    """Seeding via the fused `dynamo_rs.extract_tfpeaks`.

    extracthistpeaks.m runs the same watershed → merge → trim → regionprops →
    filterStatsTable chain as TF-peak extraction, just over a histogram
    instead of a spectrogram and as a single segment with no downsampling. So
    the fused kernel does the whole job: pass the SO-feature axis where it
    expects times and the frequency axis where it expects frequencies, and its
    PeakTime/Duration outputs come back as SOFeature and the SO-feature
    extent.
    """
    from pydynamo.tfpeaks.extract import extract_tfpeaks_fused

    x = np.asarray(x, dtype=float).ravel()
    span = float(x[-1] - x[0]) + abs(float(x[1] - x[0]))
    df = extract_tfpeaks_fused(
        np.where(np.isfinite(img), img, 0.0), x, np.asarray(y, float).ravel(),
        seg_time=span * 10.0,          # one segment over the whole image
        return_labels=False,
        downsample=(1, 1),             # extracthistpeaks does not downsample
        merge_thresh=float(merge_thresh),
        max_merges=_MAX_MERGES,
        trim_vol=float(trim_vol),
        dur_min=float(dur_min), dur_max=float("inf"),
        bw_min=float(bw_min), bw_max=float("inf"),
        # filterStatsTable compares pow2db(Height) > pow2db(height_min);
        # height_min == 0 makes that -inf, i.e. Height > 0.
        prom_min=(10.0 * np.log10(height_min) if height_min > 0
                  else float("-inf")),
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()
    return pd.DataFrame({
        "label": df["label"].to_numpy(),
        "PeakFrequency": df["PeakFrequency"].to_numpy(),
        "SOFeature": df["PeakTime"].to_numpy(),
        "Height": df["Height"].to_numpy(),
        "Duration": df["Duration"].to_numpy(),
        "Bandwidth": df["Bandwidth"].to_numpy(),
        "Area": df["Area"].to_numpy(),
    })


def extract_hist_peaks(img, x, y, merge_thresh, dur_min, bw_min, height_min,
                       trim_vol, use_fused=None):
    """Seed modes from a watershed over `img`.

    `img` is (len(y), len(x)) = (n_freq, n_feature). Returns
    ``(stats, labels)`` where `stats` is a DataFrame with the columns
    param_basis_* consumes — PeakFrequency, SOFeature, Height, Duration,
    Bandwidth — and `labels` is the final label image (for plotting).

    `merge_thresh` may be NaN, which selects the 95th-percentile dynamic
    threshold that the power axis uses by default.
    """
    img = np.asarray(img, dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if img.size == 0 or not np.any(img) or not np.any(np.isfinite(img)):
        raise ValueError("image must not be empty")
    if not 0 < trim_vol <= 1:
        raise ValueError("trim_vol must be in (0, 1]")

    labels = _run_watershed(img)
    if labels.max() == 0:
        return pd.DataFrame(), labels

    if merge_thresh is None or not np.isfinite(merge_thresh):
        merge_thresh = _dynamic_merge_threshold(labels, img)

    # MATLAB merges while `max_wt > merge_thresh` (strict); both merge paths
    # here merge while `w >= merge_thresh`. That only differs on exact
    # equality, which never comes up for the TF-peak path's fixed threshold —
    # but the dynamic threshold IS one of the edge weights, so equality is hit
    # every time and the strictness decides whether the top region survives.
    # Stepping one ULP up makes `>=` behave like MATLAB's `>`.
    strict_thresh = float(np.nextafter(float(merge_thresh), np.inf))

    # Prefer the fused kernel: it is the same chain extracthistpeaks.m runs,
    # and on real histograms it recovers seed regions the Python assembly
    # misses, which is what lets the fit reach MATLAB's mode count.
    if use_fused is None:
        use_fused = _HAS_FUSED
    if use_fused:
        stats = _extract_hist_peaks_fused(img, x, y, strict_thresh, dur_min,
                                          bw_min, height_min, trim_vol)
        return stats, labels

    labels = np.asarray(
        merge_segment(labels, np.where(np.isfinite(img), img, 0.0),
                      merge_thresh=strict_thresh,
                      max_merges=_MAX_MERGES),
        dtype=np.int64,
    )
    if labels.max() == 0:
        return pd.DataFrame(), labels

    # --- Pre-trim size rejection (extracthistpeaks.m) ----------------------
    # Index-based extent, i.e. one bin narrower than the BoundingBox-derived
    # Duration/Bandwidth used in the final filter below. Both criteria apply
    # whenever either minimum is positive.
    if dur_min > 0 or bw_min > 0:
        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])
        keep_labels = []
        for lab in np.unique(labels):
            if lab == 0:
                continue
            rows, cols = np.nonzero(labels == lab)
            if ((cols.max() - cols.min()) * dx > dur_min
                    and (rows.max() - rows.min()) * dy > bw_min):
                keep_labels.append(lab)
        if not keep_labels:
            return pd.DataFrame(), labels
        labels = np.where(np.isin(labels, keep_labels), labels, 0)

    # --- Trim -------------------------------------------------------------
    if trim_vol < 1:
        shift_val = float(np.nanmin(img))
        labels, _ = trim_regions(
            labels, np.where(np.isfinite(img), img, 0.0),
            vol_thresh=float(trim_vol), conn=_CONN_TRIM, shift_val=shift_val,
        )
        labels = np.asarray(labels, dtype=np.int64)
        if labels.max() == 0:
            return pd.DataFrame(), labels

    stats = _region_stats(labels, img, x, y)
    if stats.empty:
        return stats, labels

    # --- Final filter (filterStatsTable.m; all strict) ---------------------
    keep = ((stats["Duration"] > dur_min)
            & (stats["Bandwidth"] > bw_min)
            & (stats["Height"] > height_min))
    stats = stats[keep].reset_index(drop=True)
    return stats, labels


def seeds_from_stats(stats, freq_limits, min_freq_diff, wshed_exp=False,
                     kind="power"):
    """Turn an `extract_hist_peaks` table into a (N, 6) seed stack.

    Mirrors param_basis_power.m:196-236: drop seeds outside `freq_limits`,
    sort by Height descending, greedily drop any seed within `min_freq_diff`
    of a taller one already kept, then map columns onto mode parameters.

    MATLAB divides the `Bandwidth`/`Duration` bounding-box extents by 1.96 to
    get a width. Reproduced as written, keeping numerical parity. (The divisor
    is dimensionally questionable — 1.96 is a half-width z-score but the
    extents are full widths — but that is MATLAB's choice, not ours to fix
    here.) The extra `/sqrt(2)` is the sigma reparameterization, which keeps
    the seeded *shape* identical now that the parameter is a sigma.

    `kind` matters because slot 4 is polymorphic: the SO-power Gaussian width
    for ``'power'`` (rescales) but `recikappa` for ``'phase'``, which was
    always a true sigma and must NOT be rescaled.
    """
    if stats is None or len(stats) == 0:
        return np.zeros((0, 6))

    s = stats[(stats["PeakFrequency"] >= freq_limits[0])
              & (stats["PeakFrequency"] <= freq_limits[1])]
    if s.empty:
        return np.zeros((0, 6))

    s = s.sort_values("Height", ascending=False).reset_index(drop=True)

    if min_freq_diff > 0 and len(s) > 1:
        kept_idx, kept_freqs = [], []
        for i, f in enumerate(s["PeakFrequency"].to_numpy()):
            if all(abs(kf - f) >= min_freq_diff for kf in kept_freqs):
                kept_idx.append(i)
                kept_freqs.append(f)
        s = s.iloc[kept_idx].reset_index(drop=True)

    amp0 = s["Height"].to_numpy(dtype=float)
    if wshed_exp:
        amp0 = np.log(amp0)

    # Slot 4 is a Gaussian sigma for power but recikappa for phase; only the
    # former takes the sigma rescale.
    so_width_divisor = 1.96 * SQRT2 if kind == "power" else 1.96

    return np.column_stack([
        amp0,
        s["PeakFrequency"].to_numpy(dtype=float),
        s["Bandwidth"].to_numpy(dtype=float) / (1.96 * SQRT2),
        s["SOFeature"].to_numpy(dtype=float),
        s["Duration"].to_numpy(dtype=float) / so_width_divisor,
        np.zeros(len(s)),
    ])
