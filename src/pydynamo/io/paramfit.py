"""Parametric-basis fit CSV (OUTPUT_FORMAT.md §2.4, format 3).

Serializes one axis's `pydynamo.soph.paramfit.ParamFitResult` — whose
``params_table`` (built by `soph/paramfit/output.py`) already carries
the canonical per-mode column values — plus the goodness-of-fit
scalars, background coefficients, and source bins into the same file
`dynamo-export::fits_writer::write_paramfit_csv` produces:

* a ``#``-prefixed preamble with the §8.1 stamp, ``fit_type``,
  ``n_modes``, axis-specific background keys, ``gof.*`` scalars, and
  the bin-center arrays as JSON rows;
* one data row per mode, columns

    power:  Density,FreqMean,FreqStd,SOpowerMean,SOpowerStd,Theta,
            Volume,PrefPhase,Coupling,<Pk*>
    phase:  Density,FreqMean,FreqStd,SOphaseMean,SOphaseStd,Theta,
            Volume,SOpowerMean,<Pk*>

  where ``<Pk*>`` are the trailing per-mode TF-peak summary columns
  (``PkCount`` + nine property means). Numeric cells are fixed-point
  with 17 fractional digits (Rust ``{:.17}``); non-finite cells are
  ``NaN``, and ``PkCount`` is an integer.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from pydynamo.io.stamp import Provenance
from pydynamo.soph.paramfit.output import PK_COLUMNS

#: Paramfit-CSV schema version this module writes (§8.2): format 3 =
#: the §8.1 stamp block + the format-2 numerics (true-σ widths). Format
#: 1 (`# version: 1`) emitted √2·σ widths and must never be pooled with
#: 2/3; format 2 used the legacy `# version:` / `# code_version:` keys.
PARAMFIT_CSV_FORMAT = 3

_PK_HEADER = ",".join(PK_COLUMNS)

#: Canonical data-table headers, matching the Rust writer byte for byte.
PARAMFIT_HEADER_POWER = (
    "Density,FreqMean,FreqStd,SOpowerMean,SOpowerStd,Theta,Volume,"
    "PrefPhase,Coupling," + _PK_HEADER
)
PARAMFIT_HEADER_PHASE = (
    "Density,FreqMean,FreqStd,SOphaseMean,SOphaseStd,Theta,Volume,"
    "SOpowerMean," + _PK_HEADER
)

#: Axis-specific background-coefficient preamble keys. Power background
#: is a tilted plane (two slopes + an offset); phase background is a
#: sinusoid (amplitude + phase + offset).
_BG_KEYS = {
    "power": ("PowSlope", "FreqSlope", "Offset"),
    "phase": ("SinAmp", "SinPhase", "Offset"),
}

_SO_FIELD = {"power": "SOpower_bins", "phase": "SOphase_bins"}


def _fmt17(v: float) -> str:
    """`NaN` for non-finite, else fixed-point with 17 fractional digits
    (the Python spelling of Rust's ``{:.17}``)."""
    return f"{v:.17f}" if math.isfinite(v) else "NaN"


def _json_row(values) -> str:
    """Bin array as a JSON row of ``_fmt17`` numbers; non-finite values
    become ``null`` so the row stays parseable by a strict JSON reader."""
    cells = [
        f"{v:.17f}" if math.isfinite(v) else "null"
        for v in np.asarray(values, dtype=float).ravel()
    ]
    return "[" + ",".join(cells) + "]"


def write_paramfit_csv(
    path: Path | str,
    fit,
    axis: str,
    so_bins,
    freq_bins,
    subject_id: str = "",
    stamp: Provenance | None = None,
    unit_row: float = math.nan,
) -> None:
    """Write one axis's parametric fit as a format-3 paramfit CSV.

    Parameters
    ----------
    fit : `ParamFitResult` for this axis. ``params_table`` supplies the
          per-mode column values (already annotated with PrefPhase /
          Coupling and the Pk* summaries where applicable); ``background``
          and ``gof`` fill the preamble.
    axis : ``'power'`` or ``'phase'``.
    so_bins, freq_bins : the SOPH bin centers the fit ran on, emitted in
          the preamble so ``model_SOPH`` can be reconstructed from the
          file alone.
    subject_id : emitted as ``# subjectID:`` when non-empty.
    stamp : §8.1 provenance; ``None`` uses this process's current stamp.
    unit_row : phase-axis row-normalization scale. pydynamo does not
          carry one, so the default emits ``NaN`` (as the Rust writer
          does for the power axis).
    """
    if axis not in _BG_KEYS:
        raise ValueError(f"axis must be 'power' or 'phase', got {axis!r}")
    if stamp is None:
        from pydynamo.io.stamp import current_stamp
        stamp = current_stamp()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = fit.params_table
    n_modes = len(table)
    background = np.asarray(fit.background, dtype=float).ravel()
    gof = fit.gof

    header = (
        PARAMFIT_HEADER_POWER if axis == "power" else PARAMFIT_HEADER_PHASE
    )
    # The phase table carries no SOpowerMean column (the cross-axis
    # dominant SO-power location); emit NaN placeholders when absent so
    # the canonical column layout is preserved.
    columns = header.split(",")

    def _cell(row, name: str) -> str:
        if name == "PkCount":
            v = row.get("PkCount", 0.0)
            return str(int(v) if math.isfinite(v) else 0)
        v = row.get(name, math.nan)
        return _fmt17(float(v))

    bg0, bg1, bg2 = _BG_KEYS[axis]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# DYNAM-O parametric fit\n")
        f.write(f"# format: {PARAMFIT_CSV_FORMAT}\n")
        f.write(f"# writer: {stamp.writer}\n")
        f.write(f"# writer_version: {stamp.writer_version}\n")
        f.write(f"# kernel_version: {stamp.kernel_version}\n")
        if subject_id:
            f.write(f"# subjectID: {subject_id}\n")
        f.write(f"# fit_type: {axis}\n")
        f.write(f"# n_modes: {n_modes}\n")
        f.write(f"# background.{bg0}: {_fmt17(background[0])}\n")
        f.write(f"# background.{bg1}: {_fmt17(background[1])}\n")
        f.write(f"# background.{bg2}: {_fmt17(background[2])}\n")
        f.write(f"# unit_row: {_fmt17(unit_row)}\n")
        f.write(f"# gof.sse: {_fmt17(float(gof['sse']))}\n")
        f.write(f"# gof.rsquare: {_fmt17(float(gof['rsquare']))}\n")
        f.write(f"# gof.dfe: {_fmt17(float(gof['dfe']))}\n")
        f.write(f"# gof.adjrsquare: {_fmt17(float(gof['adjrsquare']))}\n")
        f.write(f"# gof.rmse: {_fmt17(float(gof['rmse']))}\n")
        f.write(f"# freq_bins: {_json_row(freq_bins)}\n")
        f.write(f"# {_SO_FIELD[axis]}: {_json_row(so_bins)}\n")
        # Empty placeholders — kept for parity with the MATLAB writer;
        # loaders tolerate blanks.
        f.write("# fitobj_coefnames: []\n")
        f.write("# fitobj_coefvalues: []\n")
        f.write(header + "\n")
        for _, row in table.iterrows():
            f.write(",".join(_cell(row, name) for name in columns) + "\n")
