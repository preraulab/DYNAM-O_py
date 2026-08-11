"""Spline-basis fit TIFF (OUTPUT_FORMAT.md §2.5, ImageDescription format 2).

Two-page f32 grayscale TIFF mirroring
`dynamo-export::fits_writer::write_splinefit_tiff`:

* page 1 — coefficient matrix ``(m_y, m_x)``, carrying the JSON
  ImageDescription (augmented knots, fit-window bins, subject, stamp);
* page 2 — the reconstructed splinefit on the analysis-window grid
  ``(n_fit_feature, n_fit_freq)`` (a bare page; the metadata lives on
  page 1).

`pydynamo.soph.splinefit.fit_spline_basis` already returns the
analysis-window submatrix, so no re-slicing against the full SOPH grid
is needed here. NaN pixels are preserved (a fully-NaN splinefit means
"no finite SOPH cells"; f32 encodes NaN faithfully).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from pydynamo.io.stamp import Provenance

#: Splinefit-TIFF metadata schema version (§8.2), shared with the SOPH
#: TIFF: format 2 = integer ``format`` + ``pixel_format`` + stamp keys.
SPLINEFIT_TIFF_FORMAT = 2

_SO_FIELD = {"power": "SOpower_bins", "phase": "SOphase_bins"}


def write_splinefit_tiff(
    path: Path | str,
    fit,
    axis: str,
    subject_id: str = "",
    stamp: Provenance | None = None,
) -> None:
    """Write one axis's spline fit as a two-page f32 TIFF.

    Parameters
    ----------
    fit : `SplineFitResult` for this axis; supplies ``coefs``,
          ``splinefit``, the augmented knot vectors, and the fit-window
          bin centers.
    axis : ``'power'`` or ``'phase'``.
    stamp : §8.1 provenance; ``None`` uses this process's current stamp.
    """
    if axis not in _SO_FIELD:
        raise ValueError(f"axis must be 'power' or 'phase', got {axis!r}")
    if stamp is None:
        from pydynamo.io.stamp import current_stamp
        stamp = current_stamp()

    coefs = np.ascontiguousarray(
        np.asarray(fit.coefs, dtype=np.float64), dtype=np.float32
    )
    splinefit = np.ascontiguousarray(
        np.asarray(fit.splinefit, dtype=np.float64), dtype=np.float32
    )
    if coefs.ndim != 2 or splinefit.ndim != 2:
        raise ValueError("coefs and splinefit must be 2-D")
    if splinefit.size == 0:
        raise ValueError("empty fit-window in splinefit")

    meta = {
        "label": "splinefit",
        "knots_x": np.asarray(fit.knots_x_aug, dtype=float).ravel().tolist(),
        "knots_y": np.asarray(fit.knots_y_aug, dtype=float).ravel().tolist(),
        "freq_bins": np.asarray(
            fit.fit_freq_bins, dtype=float).ravel().tolist(),
        _SO_FIELD[axis]: np.asarray(
            fit.fit_SOfeature_bins, dtype=float).ravel().tolist(),
        "subjectID": subject_id,
        "page1": "coefs",
        "page2": "splinefit",
        "format": SPLINEFIT_TIFF_FORMAT,
        "pixel_format": "f32 row-major",
        "writer": stamp.writer,
        "writer_version": stamp.writer_version,
        "kernel_version": stamp.kernel_version,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(path) as tif:
        tif.write(
            coefs,
            photometric="minisblack",
            description=json.dumps(meta),
            metadata=None,
        )
        tif.write(splinefit, photometric="minisblack", metadata=None)
