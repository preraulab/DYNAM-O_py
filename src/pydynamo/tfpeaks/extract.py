"""Per-segment TF-peak extraction — follows pyDYNAM-O's `detect_tfpeaks`.

Flow per segment:
    1. (optional) downsample by (time_factor, freq_factor)
    2. watershed on -spect with watershed_line=True
    3. build RAG via expand_labels(distance=5) + skimage.graph.RAG
    4. merge_segment: iterative max-weight edge merge until w < merge_thresh
    5. expand merged labels (fill the 0-line) + (optional) resize up
    6. filter regions by bandwidth/duration/prominence BEFORE trimming (so
       we don't pay for trimming peaks we'd throw away)
    7. per-region trim to `trim_volume` fraction — keep largest subregion
    8. regionprops stats table → pandas
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2
from skimage import measure, morphology
from skimage.segmentation import expand_labels
from skimage.segmentation import watershed as _sk_watershed
from skimage.transform import resize

# Use MATLAB-matching watershed from dynamo_rs when available.
try:
    import dynamo_rs as _dynamo_rs
    _HAS_RUST_WS = True
except ImportError:
    _HAS_RUST_WS = False
    _dynamo_rs = None


def watershed(img_neg, connectivity=2, watershed_line=True):
    """Drop-in replacement matching the signature used in this module.

    Prefers dynamo_rs.matlab_watershed (pure-Rust port of MATLAB's IPT
    watershed with column-major indexing + FIFO-within-priority) for
    better agreement with MATLAB. Falls back to skimage when not installed.
    """
    if _HAS_RUST_WS:
        arr = np.ascontiguousarray(img_neg, dtype=np.float64)
        return _dynamo_rs.matlab_watershed(arr).astype(np.int64)
    return _sk_watershed(img_neg, connectivity=connectivity, watershed_line=watershed_line)

from pydynamo.tfpeaks.merge import merge_segment


def _pow2db(y: np.ndarray | float) -> np.ndarray | float:
    """Power → dB, 0 → NaN (pyDYNAM-O compatibility)."""
    if np.isscalar(y):
        if y <= 0:
            return np.nan
        return 10 * np.log10(y)
    arr = np.asarray(y, dtype=float)
    out = np.where(arr > 0, 10 * np.log10(np.where(arr > 0, arr, 1.0)), np.nan)
    return out


def _filter_label_image(labels: np.ndarray, table: pd.DataFrame) -> np.ndarray:
    """Keep only label ids represented by rows in `table`."""
    if table.empty or "label" not in table:
        return np.zeros_like(labels)
    kept = table["label"].to_numpy(dtype=np.int64, copy=False)
    return np.where(np.isin(labels, kept), labels, 0)


def min_prominence(num_tapers: int, alpha: float = 0.95) -> float:
    """Chi-squared-based minimum peak prominence (dB). Matches MATLAB
    `ht_db_min` and pyDYNAM-O's min_prominence."""
    chi2_df = 2 * num_tapers
    return -_pow2db(chi2_df / chi2.ppf(alpha / 2 + 0.5, chi2_df)) * 2


def _trim_region(
    labels_merged: np.ndarray,
    graph_data: np.ndarray,
    region_num: int,
    trim_volume: float,
) -> np.ndarray:
    """Port of pyDYNAM-O's trim_region. Returns a mask image with
    `region_num` at retained pixels, 0 elsewhere."""
    reg_idx = np.where(labels_merged == region_num)
    reg_vals = np.sort(graph_data[reg_idx])
    total = float(reg_vals.sum())
    if total <= 0 or reg_vals.size == 0:
        return np.zeros_like(graph_data)

    percent_vol = np.cumsum(reg_vals) / total
    # First index where cumulative fraction exceeds (1 - trim_volume)
    above = np.flatnonzero(percent_vol > (1.0 - trim_volume))
    if above.size == 0:
        return np.zeros_like(graph_data)
    trim_idx = int(above[0])
    trim_level = reg_vals[trim_idx]

    rplot = np.zeros(graph_data.shape, dtype=bool)
    rplot[reg_idx] = True

    img = rplot & (graph_data > trim_level)
    filled = morphology.remove_small_holes(img)
    label_img = measure.label(filled, connectivity=2)

    if label_img.max() > 1:
        props = measure.regionprops(label_img)
        max_area_label = props[int(np.argmax([p.area for p in props]))].label
        label_img = label_img == max_area_label

    return (label_img > 0).astype(np.int64) * region_num


def extract_tfpeaks_segment(
    spect: np.ndarray,                 # (F, T) baseline-subtracted segment
    stimes: np.ndarray,
    sfreqs: np.ndarray,
    segment_num: int = 1,
    downsample: Tuple[int, int] | None = (2, 2),
    merge_thresh: float = 8.0,
    max_merges: float = float("inf"),
    trim_vol: float = 0.8,
    dur_min: float = 0.5,              # default to half a window (~1s/2)
    dur_max: float = 5.0,
    bw_min: float = 2.0,               # ~ 2 * (TW / windowlen)
    bw_max: float = 15.0,
    prom_min: float | None = None,     # chi-squared-based if None
    num_tapers_for_prom: int = 3,
    return_labels: bool = False,
    return_raw_labels: bool = False,
    trim_shift_val: float | None = None,
) -> pd.DataFrame | Tuple[pd.DataFrame, np.ndarray]:
    """Extract TF-peaks from one spectrogram segment.

    When `return_labels=True`, also returns the trimmed (F, T) label image
    restricted to peaks in the returned table. Set `return_raw_labels=True`
    only when the unfiltered image is needed as a pass-1 watershed mask.
    """
    spect = np.ascontiguousarray(np.asarray(spect, dtype=np.float64))
    F, T = spect.shape
    d_time = float(stimes[1] - stimes[0])
    d_freq = float(sfreqs[1] - sfreqs[0])

    if prom_min is None:
        prom_min = min_prominence(num_tapers_for_prom, 0.95)

    # 1) Downsample (pyDYNAM-O: simple stride slicing)
    if downsample is not None and (downsample[0] > 1 or downsample[1] > 1):
        t_f, f_f = int(downsample[0]), int(downsample[1])
        seg_LR = spect[::f_f, ::t_f]
    else:
        f_f = t_f = 1
        seg_LR = spect

    # 2) Watershed on negated spectrogram (+ watershed_line)
    labels = watershed(-seg_LR, connectivity=2, watershed_line=True)

    # 3 + 4) Build RAG and merge
    labels_merged = merge_segment(
        labels, seg_LR,
        merge_thresh=merge_thresh, max_merges=max_merges,
    )

    # 5) Fill the remaining 0-line (former borders) with surrounding region ids.
    # MATLAB's extractTFPeaks.m:272 paints interior+border into Ldata
    # (`Ldata(ii_pixels)=ii` where ii_pixels = rgn{ii} = union(interior,
    # border) via Ldata2graph.m:233). Two paths:
    #   - _dynamo_rs.matlab_paint_labels: 8-conn 1-pixel dilation per label
    #     in ascending order with dense 1..N cell indices — bit-matches the
    #     MATLAB paint and closed the pipeline gap from +1.96% to -0.80%
    #     when this was adopted in the Rust extract pipeline (2026-04-21).
    #   - skimage.segmentation.expand_labels(distance=5): approximation that
    #     fills 0-line pixels with the nearest region label. Kept as the
    #     pure-Python fallback when dynamo_rs isn't built.
    labels_merged = labels_merged.astype(np.int64)
    if _HAS_RUST_WS:
        labels_merged = _dynamo_rs.matlab_paint_labels(
            np.ascontiguousarray(labels_merged)
        )
    else:
        labels_merged = expand_labels(labels_merged, distance=5).astype(np.int64)

    # Resize merged labels back up to the full-segment shape if we downsampled.
    if downsample is not None and (f_f > 1 or t_f > 1):
        # order=0 to preserve label values
        labels_full = resize(
            labels_merged, spect.shape, order=0,
            preserve_range=True, anti_aliasing=False,
        ).astype(np.int64)
    else:
        labels_full = labels_merged

    # 6a) Pre-trim dur/bw filter (MATLAB extractTFPeaks.m:300-307). MATLAB
    # drops ~57% of merged regions here before trim runs, using the inclusive
    # N_pixels * step span reported as Duration/Bandwidth.
    # Dropping small regions before trim both changes the final peak count
    # (because trim can create spurious sub-regions from jagged bboxes)
    # AND skips trim's per-region work on regions we'd reject anyway.
    if dur_min > 0 or bw_min > 0:
        pre_props = measure.regionprops(labels_full)
        drop_labels = []
        for p in pre_props:
            minr, minc, maxr, maxc = p.bbox
            pre_dur = (maxc - minc) * d_time
            pre_bw = (maxr - minr) * d_freq
            if not (pre_dur > dur_min and pre_bw > bw_min):
                drop_labels.append(p.label)
        if drop_labels:
            drop_mask = np.isin(labels_full, drop_labels)
            labels_full = np.where(drop_mask, 0, labels_full).astype(np.int64)

    # 6b) Trim all regions in one Rust call (was a per-region Python loop that
    # dominated wallclock — ~800ms/segment vs ~12ms in Rust). Post-filter
    # by duration/bandwidth/prominence happens after stats computation below.
    from pydynamo.tfpeaks.trim import trim_regions as _trim_all
    trim_labels, _ = _trim_all(
        labels_full.astype(np.int64, copy=False),
        spect, vol_thresh=trim_vol, shift_val=trim_shift_val,
    )
    # Cast to int64 so downstream regionprops_table / indexing stays consistent.
    trim_labels = trim_labels.astype(np.int64, copy=False)

    if trim_labels.max() == 0:
        empty = pd.DataFrame()
        if return_labels:
            return empty, trim_labels
        return empty

    # 7) Stats table
    props = pd.DataFrame(
        measure.regionprops_table(
            trim_labels, spect,
            properties=("label", "centroid_weighted", "bbox", "area",
                        "intensity_min", "intensity_max"),
        )
    )
    if props.empty:
        if return_labels:
            return props, (trim_labels if return_raw_labels
                           else np.zeros_like(trim_labels))
        return props

    # Prominence in dB: MATLAB / pyDYNAM-O compute pow2db(max - min) on
    # the baseline-subtracted (linear) spect; reject peaks below prom_min.
    # This was the pre-trim filter; now enforced as a post-stats gate.
    raw_prom_linear = props["intensity_max"] - props["intensity_min"]
    props["Prominence_dB"] = _pow2db(raw_prom_linear.to_numpy())
    minr = props["bbox-0"]
    minc = props["bbox-1"]
    maxr = props["bbox-2"]
    maxc = props["bbox-3"]
    # Reported stats match MATLAB's computePeakStatsTable: bb-width * dx.
    # skimage bbox is (min, max+1), so (maxc - minc) already = N_pixels.
    props["Duration"] = (maxc - minc) * d_time
    props["Bandwidth"] = (maxr - minr) * d_freq

    # Bounding box in (time_tl, freq_tl, width_s, height_Hz)
    props["BoundingBox"] = [
        (float(c) * d_time + float(stimes[0]),
         float(r) * d_freq + float(sfreqs[0]),
         float(mc - c) * d_time,
         float(mr - r) * d_freq)
        for r, c, mr, mc in zip(minr, minc, maxr, maxc)
    ]

    props["PeakTime"] = props["centroid_weighted-1"] * d_time + float(stimes[0])
    props["PeakFrequency"] = props["centroid_weighted-0"] * d_freq + float(sfreqs[0])
    props["Height"] = props["intensity_max"] - props["intensity_min"]
    props["SegmentNum"] = int(segment_num)

    # Area: pixel count * dt * df, in sec*Hz (computePeakStatsTable.m:128)
    props["Area"] = props["area"].to_numpy() * d_time * d_freq

    # Volume: sum(spect at label pixels) * dt * df
    def _vol(lab):
        return float(spect[trim_labels == lab].sum()) * d_time * d_freq
    props["Volume"] = [_vol(int(lab)) for lab in props["label"]]

    # Peakiness in dB: 10*log10(Area * Height / Volume)
    # (computePeakStatsTable.m:206, and dynamo_rs extract_pipeline.rs:487).
    # Volume == 0 or a non-positive ratio leaves the ratio undefined, so emit
    # NaN there rather than letting log10 raise or return -inf.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = props["Area"].to_numpy() * props["Height"].to_numpy() / \
            props["Volume"].to_numpy()
        props["Peakiness"] = np.where(ratio > 0.0, 10.0 * np.log10(ratio), np.nan)

    # Current MATLAB and Rust use the same inclusive N_pixels * step span for
    # filtering and for the reported Duration/Bandwidth values.
    prom_ok = np.isnan(props["Prominence_dB"].to_numpy()) | \
              (props["Prominence_dB"].to_numpy() > prom_min)
    filter_dur = props["Duration"].to_numpy()
    filter_bw = props["Bandwidth"].to_numpy()
    props = props[
        (filter_dur > dur_min) & (props["Duration"] < dur_max) &
        (filter_bw > bw_min) & (props["Bandwidth"] < bw_max) &
        prom_ok
    ].reset_index(drop=True)

    # Drop regionprops raw columns we no longer need
    for c in ["centroid_weighted-0", "centroid_weighted-1",
              "bbox-0", "bbox-1", "bbox-2", "bbox-3", "area",
              "intensity_min", "intensity_max", "Prominence_dB"]:
        if c in props.columns:
            del props[c]
    if return_labels:
        labels_out = (trim_labels if return_raw_labels
                      else _filter_label_image(trim_labels, props))
        return props, labels_out
    return props


_HAS_FUSED = _HAS_RUST_WS and hasattr(_dynamo_rs, "extract_tfpeaks")

# Columns the fused Rust kernel returns → pydynamo stats_table names.
_FUSED_COLS = {
    "peak_time": "PeakTime", "peak_freq": "PeakFrequency",
    "duration": "Duration", "bandwidth": "Bandwidth", "height": "Height",
    "volume": "Volume", "segment_num": "SegmentNum", "area": "Area",
    "peakiness": "Peakiness",
}


def extract_tfpeaks_fused(
    spect: np.ndarray,
    stimes: np.ndarray,
    sfreqs: np.ndarray,
    seg_time: float = 30.0,
    return_labels: bool = False,
    downsample: Tuple[int, int] | None = (2, 2),
    merge_thresh: float = 11.0,
    max_merges: float = float("inf"),
    trim_vol: float = 0.8,
    dur_min: float = 0.5,
    dur_max: float = 5.0,
    bw_min: float = 2.0,
    bw_max: float = 15.0,
    prom_min: float | None = None,
    num_tapers_for_prom: int = 3,
    return_raw_labels: bool = False,
    trim_shift_val: float | None = None,
    **_ignored,
) -> pd.DataFrame | Tuple[pd.DataFrame, np.ndarray]:
    """Whole-spectrogram extraction via the fused `dynamo_rs.extract_tfpeaks`.

    This is the same kernel the MATLAB rust backend calls, so Python and
    MATLAB run identical extraction code instead of Python re-assembling
    watershed / merge / paint / trim itself.

    Filter split follows runSegmentedData.m:169-175: only `dur_min`/`bw_min`
    go into Rust, with the max/height caps deliberately left unbounded there
    and applied here afterwards. That split matters — MATLAB masks the pass-2
    spectrogram with the *unfiltered* pass-1 label image, so applying the caps
    inside Rust would shrink the mask and cost pass-2 peaks.

    Label images are restricted to rows in the returned table by default.
    `return_raw_labels=True` preserves the unfiltered Rust labels for the
    internal pass-1 mask. `HeightData` and `Boundaries` are object-valued
    columns containing one variable-length NumPy array per peak.
    """
    if prom_min is None:
        prom_min = min_prominence(num_tapers_for_prom, 0.95)

    spect = np.ascontiguousarray(spect, dtype=np.float64)
    if downsample is None:
        downsample_t = downsample_f = 1
    else:
        downsample_t = int(downsample[0])
        downsample_f = int(downsample[1])
    if trim_shift_val is None:
        trim_shift_val = float(np.min(spect))

    res = _dynamo_rs.extract_tfpeaks(
        spect,
        np.ascontiguousarray(stimes, dtype=np.float64),
        np.ascontiguousarray(sfreqs, dtype=np.float64),
        None,                       # baseline already divided out upstream
        float(seg_time),
        downsample_f, downsample_t,
        float(merge_thresh), float(max_merges), float(trim_vol),
        float(trim_shift_val),       # one global shift, matching MATLAB
        float(dur_min), float("inf"),   # dur_max capped below, not in Rust
        float(bw_min), float("inf"),    # bw_max  capped below, not in Rust
        float("-inf"), float("inf"),    # freq cuts: MATLAB passes [-inf inf]
        float("-inf"),                  # ht_db_min capped below, not in Rust
        0,                          # expand_labels_distance: MATLAB paint
        True, True,                 # MATLAB's default features='all'
    )

    df = pd.DataFrame({dst: np.asarray(res[src]).ravel()
                       for src, dst in _FUSED_COLS.items()})
    bbox = np.asarray(res["bbox"], dtype=float).reshape(-1, 4)
    df["BoundingBox"] = [tuple(r) for r in bbox]
    df["HeightData"] = [
        np.asarray(values, dtype=float).ravel()
        for values in res["height_data"]
    ]
    df["Boundaries"] = [
        np.asarray(values, dtype=float).reshape(-1, 2)
        for values in res["boundaries"]
    ]
    df.insert(0, "label", np.arange(1, len(df) + 1, dtype=np.int64))

    labels = np.asarray(res["labels"], dtype=np.int64)

    # filterStatsTable.m: strict inequalities, height compared in dB.
    if len(df):
        with np.errstate(divide="ignore", invalid="ignore"):
            height_db = _pow2db(df["Height"].to_numpy())
        keep = ((df["Duration"].to_numpy() < dur_max)
                & (df["Bandwidth"].to_numpy() < bw_max)
                & (np.isnan(height_db) | (height_db > prom_min)))
        df = df[keep].reset_index(drop=True)

    if return_labels:
        labels_out = (labels if return_raw_labels
                      else _filter_label_image(labels, df))
        return df, labels_out
    return df


def extract_tfpeaks(
    spect: np.ndarray,
    stimes: np.ndarray,
    sfreqs: np.ndarray,
    seg_time: float = 30.0,
    n_jobs: int = -1,
    return_labels: bool = False,
    use_fused: bool | None = None,
    return_raw_labels: bool = False,
    **segment_kwargs,
) -> pd.DataFrame | Tuple[pd.DataFrame, np.ndarray]:
    """Split the spectrogram into `seg_time`-second segments and extract
    TF-peaks per segment, concatenating results. Uses joblib for parallelism.

    When `return_labels=True`, also returns a stitched (F, T) label image
    covering the full spectrogram. By default its nonzero ids correspond
    exactly to rows in the returned table. `return_raw_labels=True` keeps
    unfiltered labels for the internal pass-1 watershed mask.

    `use_fused` selects the single-call Rust kernel (the same one the MATLAB
    rust backend uses) instead of assembling the stages in Python. None means
    use it when available, which is both faster and closer to MATLAB; set
    False to force the Python assembly.
    """
    if use_fused is None:
        use_fused = _HAS_FUSED
    if use_fused:
        if not _HAS_FUSED:
            raise RuntimeError(
                "use_fused=True but this dynamo_rs build has no "
                "extract_tfpeaks; rebuild the coordinated native extensions "
                "with the DYNAM-O_toolbox controlled bootstrap "
                "(`./bootstrap.sh --yes` or `bootstrap.ps1 -Yes`)"
            )
        return extract_tfpeaks_fused(
            spect, stimes, sfreqs, seg_time=seg_time,
            return_labels=return_labels,
            return_raw_labels=return_raw_labels,
            **segment_kwargs,
        )

    dt = float(stimes[1] - stimes[0])
    F, T = spect.shape
    trim_shift_val = segment_kwargs.pop("trim_shift_val", None)
    if trim_shift_val is None:
        trim_shift_val = float(np.min(spect))

    # Match MATLAB segmentData.m:85-107 exactly: ceil-divide to get n_segs,
    # then ceil-divide again to get even-sized segments (last one may be
    # truncated). Stride-N with start+N misaligns every segment after the
    # first by 1 sample vs MATLAB, so per-segment watershed + merge produce
    # slightly different regions.
    import math
    max_dx = int(math.floor(seg_time / dt))
    n_segs = int(math.ceil(T / max_dx))
    new_dx = int(math.ceil(T / n_segs))
    segments = []
    for ii in range(n_segs):
        start = ii * new_dx
        end = min(start + new_dx, T)
        if end - start < 2:
            continue
        segments.append((ii + 1, start, end))

    def _run(si, start, end):
        return extract_tfpeaks_segment(
            spect[:, start:end], stimes[start:end], sfreqs,
            segment_num=si,
            return_labels=return_labels,
            return_raw_labels=True,
            trim_shift_val=trim_shift_val,
            **segment_kwargs,
        )

    if n_jobs == 1 or len(segments) <= 1:
        results = [_run(*s) for s in segments]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_run)(*s) for s in segments
        )

    if not results:
        empty = pd.DataFrame()
        return (empty, np.zeros((F, T), dtype=np.int64)) if return_labels else empty

    if return_labels:
        full_labels = np.zeros((F, T), dtype=np.int64)
        tables = []
        offset = 0
        for (si, start, end), (tbl, seg_labels) in zip(segments, results):
            seg_max = int(seg_labels.max())
            if return_raw_labels:
                selected_seg = seg_labels
            else:
                selected_seg = _filter_label_image(seg_labels, tbl)
            shifted = np.where(selected_seg > 0, selected_seg + offset, 0)
            full_labels[:, start:end] = shifted
            if not tbl.empty:
                tbl = tbl.copy()
                tbl["label"] = tbl["label"] + offset
                tables.append(tbl)
            offset += seg_max
        if not tables:
            return pd.DataFrame(), full_labels
        return pd.concat(tables, ignore_index=True), full_labels

    tables = [t for t in results if not t.empty]
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)
