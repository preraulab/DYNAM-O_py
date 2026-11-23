"""TFpeaks stats-table CSV (OUTPUT_FORMAT.md §2.1, format 3).

Writes a run's ``stats_table`` through the kernel's
`dynamo_rs::pipeline::write_stats_csv`, the DYNAM-O app's writer, onto the
canonical 14-column layout:

    PeakTime,PeakFrequency,Duration,Bandwidth,Height,Volume,SegmentNum,
    Area,Peakiness,bbox_tl_s,bbox_tl_Hz,PeakStage,SOpower,SOphase

Only the bounding-box top-left corner is stored; the extents equal
``Duration`` / ``Bandwidth`` exactly. Float columns round to
class-based significant figures (§6.2).
"""

from __future__ import annotations

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

def write_stats_csv(
    df: pd.DataFrame,
    path: Path | str,
    stamp: Provenance | None = None,
    subject_id: str | None = None,
    *,
    fs: float,
    settings: dict | None = None,
) -> None:
    """Write a stats table as a canonical stats CSV, with the kernel's
    writer (the one the DYNAM-O app uses): same columns, the same
    precision rules (``settings["output"]["csv_*_sigfigs"]`` resolved
    against ``fs``, the rate the run used), and the §8.1 preamble with the
    ``# semantics:`` line.

    Parameters
    ----------
    df : the run's ``stats_table`` (the :data:`STATS_CSV_HEADER` columns).
    path : output file; parent directories are created.
    stamp : the provenance to record; ``None`` uses this process's.
    subject_id : emitted as ``# subjectID:`` when non-empty.
    fs : the run's sample rate (``DynamoOutput.fs``).
    settings : the run's settings section (``DynamoOutput.settings``);
               None uses the default precision.
    """
    import json

    from pydynamo import _kernel
    from pydynamo.io.stamp import current_stamp

    if stamp is None:
        stamp = current_stamp()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {}
    for name in STATS_CSV_HEADER.split(","):
        columns[name] = np.ascontiguousarray(df[name].to_numpy(dtype=float))
    _kernel.rs().write_stats_csv(
        str(path),
        columns,
        float(fs),
        None if settings is None else json.dumps(settings),
        stamp.writer or "pydynamo",
        stamp.writer_version or "unknown",
        subject_id or "",
    )


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
