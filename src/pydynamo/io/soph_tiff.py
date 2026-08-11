"""SOPH histogram TIFF (OUTPUT_FORMAT.md §2.2, ImageDescription format 2).

Single-page f32 grayscale TIFF, uncompressed, row-major
``(n_so, n_freq)``, mirroring `dynamo-export::write_soph_tiff_with_meta`.
The ImageDescription tag carries a JSON object with both the Rust-native
keys (``row_centers`` / ``col_centers``) and the MATLAB-style aliases
(``SOpower_bins`` / ``SOphase_bins`` / ``freq_bins``) so loaders never
need the settings manifest, plus the §8.1 provenance stamp.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from pydynamo.io.stamp import Provenance

#: SOPH-TIFF metadata schema version (§8.2): format 2 = integer
#: ``format`` + ``pixel_format: "f32 row-major"`` + stamp keys. Format 1
#: carried the pixel-layout note in a string-valued ``format`` and no
#: stamp.
SOPH_TIFF_FORMAT = 2

_SO_KEY = {"sopower": "SOpower_bins", "sophase": "SOphase_bins"}


def write_soph_tiff(
    path: Path | str,
    hist: np.ndarray,
    so_bins,
    freq_bins,
    kind: str,
    subject_id: str = "",
    stamp: Provenance | None = None,
) -> None:
    """Write one SOPH as an f32 TIFF with format-2 metadata.

    Parameters
    ----------
    hist : ``(n_so, n_freq)`` histogram (``SOpower_mat`` /
           ``SOphase_mat``); NaN cells are preserved as f32 NaN.
    so_bins : SO-power or SO-phase bin centers (rows).
    freq_bins : frequency bin centers (columns).
    kind : ``'sopower'`` or ``'sophase'``.
    stamp : §8.1 provenance; ``None`` uses this process's current stamp.
    """
    if kind not in _SO_KEY:
        raise ValueError(f"kind must be 'sopower' or 'sophase', got {kind!r}")
    if stamp is None:
        from pydynamo.io.stamp import current_stamp
        stamp = current_stamp()

    hist = np.asarray(hist, dtype=np.float64)
    so_bins = np.asarray(so_bins, dtype=float).ravel()
    freq_bins = np.asarray(freq_bins, dtype=float).ravel()
    h, w = hist.shape
    if so_bins.size != h:
        raise ValueError(f"so_bins len {so_bins.size} != hist rows {h}")
    if freq_bins.size != w:
        raise ValueError(f"freq_bins len {freq_bins.size} != hist cols {w}")

    meta = {
        "label": kind,
        "rows": h,
        "cols": w,
        "row_centers": so_bins.tolist(),
        "col_centers": freq_bins.tolist(),
        _SO_KEY[kind]: so_bins.tolist(),
        "freq_bins": freq_bins.tolist(),
        "subjectID": subject_id,
        "format": SOPH_TIFF_FORMAT,
        "pixel_format": "f32 row-major",
        "writer": stamp.writer,
        "writer_version": stamp.writer_version,
        "kernel_version": stamp.kernel_version,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        np.ascontiguousarray(hist, dtype=np.float32),
        photometric="minisblack",
        description=json.dumps(meta),
        metadata=None,
    )
