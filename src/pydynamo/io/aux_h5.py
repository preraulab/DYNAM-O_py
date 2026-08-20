"""Auxiliary-data HDF5 (OUTPUT_FORMAT.md §2.3, format 2).

Writes the per-run scalars + the SO-power time series as top-level HDF5
datasets in the same layout as `dynamo-export::aux_h5` (which itself
ports MATLAB `writeAuxFormats.m`), so MATLAB's ``loadAuxData`` and any
h5py reader see identical files:

    /Fs                      f64    (1,1)
    /SOpower_norm            f64    (N,1)   native compute grid
    /SOpower_t_start         f64    (1,1)
    /SOpower_norm_method     string (1,1)
    /SOpower_window_params   f64    (2,1)   [window_s, step_s]
    /SOpower_freqrange       f64    (2,1)   [lo, hi] Hz
    /artifact_spans          f64    (K,2)   [start_s, end_s]
    /stage_times             f64    (1,M)
    /stage_vals              uint8  (1,M)
    /subjectID               string (1,1)

plus the §8.1 provenance stamp as scalar datasets: ``format`` (f64 2.0),
``writer``, ``writer_version``, ``kernel_version``, and the legacy
``code_version`` (same value as ``writer_version``). Empty optional
inputs are omitted rather than written empty; readers must ignore
unknown datasets.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from pydynamo.io.stamp import Provenance

#: Aux-h5 schema version (§8.2): format 2 = the stamp datasets above.
#: Format-1 files carried ``code_version`` only.
AUX_H5_FORMAT = 2


def _write_f64_2d(f: h5py.File, name: str, arr: np.ndarray) -> None:
    f.create_dataset(name, data=np.asarray(arr, dtype=np.float64))


def _write_string_scalar(f: h5py.File, name: str, value: str) -> None:
    """(1,1) variable-length UTF-8 string dataset, matching the Rust
    writer's ``VarLenUnicode`` shape."""
    f.create_dataset(
        name,
        shape=(1, 1),
        dtype=h5py.string_dtype(encoding="utf-8"),
        data=np.array([[value]], dtype=object),
    )


def write_auxiliary_data_h5(
    path: Path | str,
    *,
    fs: float,
    so_power_norm,
    so_power_t_start: float,
    so_power_norm_method: str,
    so_power_window_params: tuple[float, float],
    so_power_freqrange: tuple[float, float],
    artifact_spans=None,
    stage_times=None,
    stage_vals=None,
    subject_id: str = "",
    stamp: Provenance | None = None,
) -> None:
    """Write ``auxiliary_data`` as an HDF5 file at ``path``. Overwrites.

    Parameters
    ----------
    fs : sample rate used by the pipeline (post-resample).
    so_power_norm : SO-power on its native compute grid (window step).
    so_power_t_start : time (s) of native sample 0.
    so_power_norm_method : normalization method string.
    so_power_window_params : ``(window_s, step_s)``.
    so_power_freqrange : SO band ``(lo, hi)`` Hz.
    artifact_spans : ``(K, 2)`` array-like of ``[start_s, end_s]`` runs.
    stage_times, stage_vals : staging timeline; ``stage_vals`` are the
        0..6 stage codes.
    stamp : §8.1 provenance; ``None`` uses this process's current stamp.
    """
    if stamp is None:
        from pydynamo.io.stamp import current_stamp
        stamp = current_stamp()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    so_power_norm = np.asarray(
        [] if so_power_norm is None else so_power_norm, dtype=np.float64
    ).ravel()
    artifact_spans = np.asarray(
        [] if artifact_spans is None else artifact_spans, dtype=np.float64
    ).reshape(-1, 2)
    stage_times = np.asarray(
        [] if stage_times is None else stage_times, dtype=np.float64
    ).ravel()
    stage_vals = np.asarray(
        [] if stage_vals is None else stage_vals
    ).ravel()

    with h5py.File(path, "w") as f:
        _write_f64_2d(f, "Fs", np.array([[float(fs)]]))
        if so_power_norm.size:
            _write_f64_2d(f, "SOpower_norm", so_power_norm.reshape(-1, 1))
        _write_f64_2d(
            f, "SOpower_t_start", np.array([[float(so_power_t_start)]])
        )

        # Provenance (aux format 2, §8.2). The legacy code_version
        # dataset carries the writer_version value for readers that
        # predate the stamp.
        _write_f64_2d(f, "format", np.array([[float(AUX_H5_FORMAT)]]))
        _write_string_scalar(f, "writer", stamp.writer or "unknown")
        _write_string_scalar(
            f, "writer_version", stamp.writer_version or "unknown"
        )
        _write_string_scalar(
            f, "kernel_version", stamp.kernel_version or "unknown"
        )
        _write_string_scalar(
            f, "code_version", stamp.writer_version or "unknown"
        )

        _write_string_scalar(f, "SOpower_norm_method", so_power_norm_method)
        _write_f64_2d(
            f, "SOpower_window_params",
            np.array(so_power_window_params, dtype=np.float64).reshape(2, 1),
        )
        _write_f64_2d(
            f, "SOpower_freqrange",
            np.array(so_power_freqrange, dtype=np.float64).reshape(2, 1),
        )
        if artifact_spans.size:
            _write_f64_2d(f, "artifact_spans", artifact_spans)
        if stage_times.size:
            _write_f64_2d(f, "stage_times", stage_times.reshape(1, -1))
        if stage_vals.size:
            f.create_dataset(
                "stage_vals",
                data=stage_vals.astype(np.uint8).reshape(1, -1),
            )
        if subject_id:
            _write_string_scalar(f, "subjectID", subject_id)


def _read_string_scalar(f: h5py.File, name: str) -> str | None:
    if name not in f:
        return None
    value = np.asarray(f[name][()]).ravel()[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def read_auxiliary_data_h5(path: Path | str) -> tuple[dict, Provenance]:
    """Read an aux file written by any DYNAM-O implementation.

    Returns ``(aux, provenance)``. ``aux`` mirrors the writer's keyword
    arguments — ``fs``, ``so_power_norm``, ``so_power_t_start``,
    ``so_power_norm_method``, ``so_power_window_params``,
    ``so_power_freqrange``, ``artifact_spans``, ``stage_times``,
    ``stage_vals``, ``subject_id`` — with ``None`` / empty values for
    optional datasets that are absent (legacy files predate several).
    Unknown datasets are ignored (§2.3); a missing stamp yields an
    all-``None`` provenance except ``format`` (format-1 files carried
    ``code_version`` only, surfaced as ``writer_version``).
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        def _scalar(name):
            return (float(np.asarray(f[name][()]).ravel()[0])
                    if name in f else None)

        def _vector(name):
            return (np.asarray(f[name][()], dtype=np.float64).ravel()
                    if name in f else np.array([], dtype=np.float64))

        fmt = _scalar("format")
        stamp = Provenance(
            format=int(fmt) if fmt is not None else None,
            writer=_read_string_scalar(f, "writer"),
            writer_version=_read_string_scalar(f, "writer_version"),
            kernel_version=_read_string_scalar(f, "kernel_version"),
        )
        if stamp.writer_version is None:
            # Legacy dataset (format 1): code_version was the writing
            # tool's build.
            stamp.writer_version = _read_string_scalar(f, "code_version")

        window = _vector("SOpower_window_params")
        freqrange = _vector("SOpower_freqrange")
        aux = {
            "fs": _scalar("Fs"),
            "so_power_norm": _vector("SOpower_norm"),
            "so_power_t_start": _scalar("SOpower_t_start"),
            "so_power_norm_method": _read_string_scalar(
                f, "SOpower_norm_method"),
            "so_power_window_params": (
                (float(window[0]), float(window[1]))
                if window.size >= 2 else None),
            "so_power_freqrange": (
                (float(freqrange[0]), float(freqrange[1]))
                if freqrange.size >= 2 else None),
            "artifact_spans": (
                np.asarray(f["artifact_spans"][()], dtype=np.float64)
                .reshape(-1, 2)
                if "artifact_spans" in f
                else np.empty((0, 2), dtype=np.float64)),
            "stage_times": _vector("stage_times"),
            "stage_vals": (
                np.asarray(f["stage_vals"][()], dtype=np.uint8).ravel()
                if "stage_vals" in f else np.array([], dtype=np.uint8)),
            "subject_id": _read_string_scalar(f, "subjectID") or "",
        }
    return aux, stamp
