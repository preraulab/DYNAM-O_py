"""Readers and writers for the canonical DYNAM-O output tree.

Implements the on-disk contract in the DYNAM-O_DesktopApp repo's
``documents/OUTPUT_FORMAT.md`` — the per-(subject, channel) artifact set
(§1-2) and the provenance stamp (§8) — so a pydynamo run can populate a
results root that ``dynamo-cli``, the desktop app, and the MATLAB
toolbox all read, and vice versa.

This package is purely I/O: the numeric core (`pydynamo.pipeline` and
below) stays in-memory-only and never writes files itself.

Modules:

* :mod:`~pydynamo.io.stamp` — the §8.1 provenance stamp.
* :mod:`~pydynamo.io.tree` — canonical path and filename builders (§1).
* :mod:`~pydynamo.io.stats` — TFpeaks stats-table CSV (§2.1, format 3).
* :mod:`~pydynamo.io.paramfit` — parametric-basis fit CSV (§2.4, format 3).
* :mod:`~pydynamo.io.soph_tiff` — SOPH f32 TIFF (§2.2, format 2).
* :mod:`~pydynamo.io.splinefit_tiff` — spline-basis fit TIFF (§2.5, format 2).
* :mod:`~pydynamo.io.aux_h5` — auxiliary-data HDF5 (§2.3, format 2).
* :mod:`~pydynamo.io.runs_jsonl` — ``_runs/*.jsonl`` run index (§4.2).
"""

from pydynamo.io.stamp import (
    Provenance,
    current_stamp,
    kernel_version,
    parse_preamble,
    writer_version,
)
from pydynamo.io.tree import (
    aux_h5_path,
    channel_dir,
    fix_filename,
    paramfit_csv_path,
    runs_dir,
    soph_tiff_path,
    splinefit_tiff_path,
    stats_csv_path,
)
from pydynamo.io.stats import write_stats_csv
from pydynamo.io.paramfit import write_paramfit_csv
from pydynamo.io.soph_tiff import write_soph_tiff
from pydynamo.io.splinefit_tiff import write_splinefit_tiff
from pydynamo.io.aux_h5 import write_auxiliary_data_h5
from pydynamo.io.runs_jsonl import (
    append_run_header,
    append_run_item,
    runs_jsonl_path,
)

__all__ = [
    "Provenance",
    "current_stamp",
    "kernel_version",
    "parse_preamble",
    "writer_version",
    "fix_filename",
    "channel_dir",
    "stats_csv_path",
    "soph_tiff_path",
    "aux_h5_path",
    "paramfit_csv_path",
    "splinefit_tiff_path",
    "runs_dir",
    "write_stats_csv",
    "write_paramfit_csv",
    "write_soph_tiff",
    "write_splinefit_tiff",
    "write_auxiliary_data_h5",
    "append_run_header",
    "append_run_item",
    "runs_jsonl_path",
]
