"""TFpeaks stats-table CSV (OUTPUT_FORMAT.md §2.1, format 3).

Serializes the in-memory ``stats_table`` DataFrame built by
`pydynamo.tfpeaks.stats.compute_peak_stats` (plus the per-peak
``PeakStage`` / ``SOpower`` / ``SOphase`` columns appended by
`run_dynamo`) onto the canonical 14-column layout written by
`dynamo_rs::pipeline::write_stats_csv`:

    PeakTime,PeakFrequency,Duration,Bandwidth,Height,Volume,SegmentNum,
    Area,Peakiness,bbox_tl_s,bbox_tl_Hz,PeakStage,SOpower,SOphase

Only the bounding-box top-left corner is emitted (``BoundingBox[0:2]``);
the extents duplicate ``Duration`` / ``Bandwidth`` exactly. The
in-memory-only ``Label`` / ``Boundaries`` / ``HeightData`` columns are
dropped. Float columns round to class-based significant figures
(§6.2 defaults: time 8, freq/phase/other 6) before text-encoding.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from pydynamo.io.stamp import Provenance, parse_preamble

#: Stats-CSV schema version this module writes (§8.2): format 3 = the
#: 14-column layout plus the `#` provenance preamble. Format 2 was the
#: same columns bare; format 1 the 16-column layout with the redundant
#: `bbox_width_s` / `bbox_height_Hz`.
STATS_CSV_FORMAT = 3

#: Canonical format-2/3 header — must stay byte-identical to the Rust
#: writer's.
STATS_CSV_HEADER = (
    "PeakTime,PeakFrequency,Duration,Bandwidth,Height,Volume,SegmentNum,"
    "Area,Peakiness,bbox_tl_s,bbox_tl_Hz,PeakStage,SOpower,SOphase"
)

#: Legacy format-1 header (16 columns, bare).
STATS_CSV_HEADER_V1 = (
    "PeakTime,PeakFrequency,Duration,Bandwidth,Height,Volume,SegmentNum,"
    "Area,Peakiness,bbox_tl_s,bbox_tl_Hz,bbox_width_s,bbox_height_Hz,"
    "PeakStage,SOpower,SOphase"
)

# Default per-class significant figures, matching
# `dynamo_rs::pipeline::StatsCsvPrecision::default()` (fs = 100 Hz).
_TIME_SIGFIGS = 8
_FREQ_SIGFIGS = 6
_PHASE_SIGFIGS = 6
_OTHER_SIGFIGS = 6


def _round_sig(v: float, n: int) -> float:
    """Round to ``n`` significant figures, half away from zero.

    Port of `dynamo_rs::pipeline::round_sig` (which uses f64 ``round``,
    half-away-from-zero — not Python's banker's rounding). ``0`` / NaN /
    infinities pass through.
    """
    if v == 0.0 or not math.isfinite(v):
        return v
    m = math.floor(math.log10(abs(v)))
    p = 10.0 ** ((n - 1) - m)
    scaled = v * p
    r = math.floor(scaled + 0.5) if scaled >= 0.0 else math.ceil(scaled - 0.5)
    return r / p


def _fmt_float(v: float) -> str:
    """Shortest round-trippable text for one float cell.

    Matches the Rust writer's spellings for the special values (`NaN`,
    `inf`, `-inf`) and drops the trailing ``.0`` Python's repr keeps on
    integral floats (Rust Display prints ``1``, not ``1.0``).
    """
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    s = repr(float(v))
    if s.endswith(".0"):
        s = s[:-2]
    return s


def write_stats_csv(
    df: pd.DataFrame,
    path: Path | str,
    stamp: Provenance | None,
    subject_id: str | None = None,
) -> None:
    """Write the in-memory stats table as a canonical stats CSV.

    Parameters
    ----------
    df : DataFrame from `run_dynamo` (``output.stats_table``). Requires
         the numeric peak columns plus ``BoundingBox``; ``PeakStage`` /
         ``SOpower`` / ``SOphase`` default to 0 / NaN / NaN when absent
         (e.g. a partial pipeline run).
    path : output file; parent directories are created.
    stamp : the §8.1 provenance stamp for the preamble. ``format`` is
            overridden to :data:`STATS_CSV_FORMAT`. ``None`` writes the
            bare format-2 file — identical data rows, no ``#`` lines —
            mirroring the Rust writer's ``prov: None`` path.
    subject_id : emitted as ``# subjectID:`` when non-empty (and a
                 preamble is being written).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(df)

    def _col(name: str, default: float) -> np.ndarray:
        if name in df.columns:
            return df[name].to_numpy(dtype=float)
        return np.full(n, default)

    peak_time = _col("PeakTime", math.nan)
    peak_freq = _col("PeakFrequency", math.nan)
    duration = _col("Duration", math.nan)
    bandwidth = _col("Bandwidth", math.nan)
    height = _col("Height", math.nan)
    volume = _col("Volume", math.nan)
    area = _col("Area", math.nan)
    peakiness = _col("Peakiness", math.nan)
    segment_num = _col("SegmentNum", 1.0)
    peak_stage = _col("PeakStage", 0.0)
    so_power = _col("SOpower", math.nan)
    so_phase = _col("SOphase", math.nan)

    if "BoundingBox" in df.columns and n > 0:
        bbox = np.asarray(
            [np.asarray(b, dtype=float).ravel()[:2] for b in df["BoundingBox"]]
        )
        bbox_tl_s = bbox[:, 0]
        bbox_tl_hz = bbox[:, 1]
    else:
        bbox_tl_s = np.full(n, math.nan)
        bbox_tl_hz = np.full(n, math.nan)

    def t(v: float) -> str:
        return _fmt_float(_round_sig(v, _TIME_SIGFIGS))

    def fz(v: float) -> str:
        return _fmt_float(_round_sig(v, _FREQ_SIGFIGS))

    def ph(v: float) -> str:
        return _fmt_float(_round_sig(v, _PHASE_SIGFIGS))

    def other(v: float) -> str:
        return _fmt_float(_round_sig(v, _OTHER_SIGFIGS))

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        if stamp is not None:
            f.write("# DYNAM-O stats table\n")
            f.write(f"# format: {STATS_CSV_FORMAT}\n")
            f.write(f"# writer: {stamp.writer}\n")
            f.write(f"# writer_version: {stamp.writer_version}\n")
            f.write(f"# kernel_version: {stamp.kernel_version}\n")
            if subject_id:
                f.write(f"# subjectID: {subject_id}\n")
        f.write(STATS_CSV_HEADER + "\n")
        for i in range(n):
            f.write(",".join((
                t(peak_time[i]),
                fz(peak_freq[i]),
                t(duration[i]),
                fz(bandwidth[i]),
                other(height[i]),
                other(volume[i]),
                str(int(segment_num[i])),
                other(area[i]),
                other(peakiness[i]),
                t(bbox_tl_s[i]),
                fz(bbox_tl_hz[i]),
                str(int(peak_stage[i]) if math.isfinite(peak_stage[i]) else 0),
                other(so_power[i]),
                ph(so_phase[i]),
            )) + "\n")


def read_stats_csv(path: Path | str) -> tuple[pd.DataFrame, Provenance]:
    """Read a canonical stats CSV, accepting formats 1, 2, and 3 (§8.2).

    * format 3 — `#` preamble + the 14-column header;
    * format 2 — the same 14 columns bare;
    * format 1 — the bare legacy 16-column header (with the redundant
      ``bbox_width_s`` / ``bbox_height_Hz`` extent columns).

    Returns ``(df, provenance)``. ``df`` carries exactly the columns of
    the file's own header; for the 14-column layouts the bbox extents
    equal ``Duration`` / ``Bandwidth``, so nothing is lost. When the
    preamble carries no ``format`` key it is inferred from the column
    count (16 -> 1, 14 -> 2, §8.3 rule 2). Any other header raises
    ``ValueError``.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    stamp, _extras = parse_preamble(lines)
    n_preamble = 0
    for line in lines:
        if not line.startswith("#"):
            break
        n_preamble += 1
    body = lines[n_preamble:]
    if not body:
        raise ValueError(f"stats_csv {path} is empty")
    header = body[0].strip()
    if header == STATS_CSV_HEADER:
        inferred = 2
    elif header == STATS_CSV_HEADER_V1:
        inferred = 1
    else:
        raise ValueError(f"stats_csv {path} has unexpected header: {header}")
    if stamp.format is None:
        stamp.format = inferred

    from io import StringIO

    df = pd.read_csv(StringIO("\n".join(body)))
    return df, stamp
