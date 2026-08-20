"""Append-only ``_runs/*.jsonl`` run index (OUTPUT_FORMAT.md §4.2).

One file per process invocation, named
``<HOST>_<yyyymmdd_HHmmss>_pid<pid>.jsonl`` (UTC timestamp, cached for
the process lifetime so a multi-subject batch appends to a single
file). Lines are JSON objects with the same key sets
`dynamo-export::matlab_layout::append_run_index` emits:

* an optional leading ``"kind": "header"`` line carrying the resolved
  batch metadata (mode / jobs / cohort size);
* one ``"kind": "item"`` line per (subject, channel) processed.

Both line kinds carry the §8.1 stamp keys (``writer``,
``writer_version``, ``kernel_version``) plus the legacy
``code_version`` (same value as ``writer_version``). The run records
are an index, not a gate — scanners still pick up files not listed in
any ``_runs/*.jsonl``.
"""

from __future__ import annotations

import getpass
import json
import os
import socket
import time
from pathlib import Path

from pydynamo.io.stamp import Provenance
from pydynamo.io.tree import runs_dir

_BATCH_ID: tuple[str, str] | None = None


def _hostname_compact() -> str:
    """Hostname token stripped to ASCII alphanumerics and upper-cased so
    it is filesystem-safe on every target OS. Checks ``HOST`` /
    ``HOSTNAME`` / ``COMPUTERNAME`` env vars, then the socket hostname."""
    raw = (
        os.environ.get("HOST")
        or os.environ.get("HOSTNAME")
        or os.environ.get("COMPUTERNAME")
        or socket.gethostname()
        or "HOST"
    )
    out = "".join(ch for ch in raw if ch.isascii() and ch.isalnum())
    return out.upper() or "HOST"


def _iso8601_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _batch_run_id() -> tuple[str, str]:
    """Process-wide ``(run_id, file_ts)``, stable across calls so the
    entire batch writes one .jsonl file."""
    global _BATCH_ID
    if _BATCH_ID is None:
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        host = _hostname_compact()
        pid = os.getpid()
        _BATCH_ID = (f"{host}_{ts}_pid{pid}", ts)
    return _BATCH_ID


def runs_jsonl_path(root: Path | str) -> Path:
    """This process's ``_runs/<HOST>_<ts>_pid<pid>.jsonl`` under ``root``."""
    _, file_ts = _batch_run_id()
    return runs_dir(root) / f"{_hostname_compact()}_{file_ts}_pid{os.getpid()}.jsonl"


def _stamp_fields(stamp: Provenance | None) -> dict:
    if stamp is None:
        from pydynamo.io.stamp import current_stamp
        stamp = current_stamp()
    return {
        # Legacy key, same value as writer_version.
        "code_version": stamp.writer_version,
        "writer": stamp.writer,
        "writer_version": stamp.writer_version,
        "kernel_version": stamp.kernel_version,
    }


def _append_line(root: Path | str, record: dict) -> Path:
    path = runs_jsonl_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record) + "\n")
    return path


def append_run_header(
    root: Path | str,
    *,
    mode: str = "batch",
    jobs_phase1: int = 0,
    jobs_phase2: int = 0,
    n_items: int = 0,
    n_subjects: int = 0,
    stamp: Provenance | None = None,
) -> Path:
    """Append the per-invocation header line; call once, before items.

    Answers "what configuration wrote this file?" for post-hoc analysis
    without rummaging through settings files.
    """
    run_id, _ = _batch_run_id()
    record = {
        "ts": _iso8601_utc(),
        "kind": "header",
        "host": _hostname_compact(),
        "user": _user(),
        "pid": os.getpid(),
        "run_id": run_id,
        **_stamp_fields(stamp),
        "mode": mode,
        "jobs_phase1": jobs_phase1,
        "jobs_phase2": jobs_phase2,
        "n_items": n_items,
        "n_subjects": n_subjects,
    }
    return _append_line(root, record)


def append_run_item(
    root: Path | str,
    *,
    subject: str,
    channel: str,
    input_file: str = "",
    staging_file: str = "",
    peaks: int = 0,
    duration_sec: float = 0.0,
    components=("TFpeaks", "SOPHs", "aux"),
    files=None,
    status: str = "ok",
    failures=(),
    mode: str = "unspecified",
    jobs_phase1: int = 0,
    jobs_phase2: int = 0,
    stamp: Provenance | None = None,
) -> Path:
    """Append one (subject, channel) item line.

    ``channel`` is the output channel label; it is sanitized with
    :func:`~pydynamo.io.tree.fix_filename` for the recorded relative
    paths, exactly as the artifact writers do. ``files`` defaults to the
    canonical stats/SOPH/aux relative paths for this pair.
    """
    from pydynamo.io.tree import fix_filename

    run_id, _ = _batch_run_id()
    channel_safe = fix_filename(channel)
    if files is None:
        files = [
            f"{channel_safe}/TFpeaks/{subject}_stats_table_{channel_safe}.csv",
            f"{channel_safe}/SOPHs/{subject}_SOPHs_power_{channel_safe}.tiff",
            f"{channel_safe}/SOPHs/{subject}_SOPHs_phase_{channel_safe}.tiff",
            f"{channel_safe}/auxiliary_data/{subject}_auxiliary_data_"
            f"{channel_safe}.h5",
        ]
    record = {
        "ts": _iso8601_utc(),
        "kind": "item",
        "host": _hostname_compact(),
        "user": _user(),
        "pid": os.getpid(),
        "run_id": run_id,
        **_stamp_fields(stamp),
        "subject": subject,
        "channel": channel_safe,
        "input_file": input_file,
        "staging_file": staging_file,
        "components": list(components),
        "files": list(files),
        "status": status,
        "failures": list(failures),
        "peaks": int(peaks),
        "duration_sec": float(duration_sec),
        "mode": mode,
        "jobs_phase1": jobs_phase1,
        "jobs_phase2": jobs_phase2,
    }
    return _append_line(root, record)


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return ""
