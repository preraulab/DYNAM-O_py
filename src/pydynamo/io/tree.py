"""Canonical DYNAM-O output-tree paths (OUTPUT_FORMAT.md §1-2).

A "results root" holds one folder per output channel, each with fixed
artifact subdirectories and `<subject>_..._<channel>` filenames:

    <root>/
      <channel>/
        TFpeaks/<subject>_stats_table_<channel>.csv
        SOPHs/<subject>_SOPHs_{power,phase}_<channel>.tiff
        auxiliary_data/<subject>_auxiliary_data_<channel>.h5
        param_basis/<subject>_SO{power,phase}_paramfit_<channel>.csv
        spline_basis/<subject>_SO{power,phase}_splinefit_<channel>.tiff
      _runs/

Subject and channel tokens pass through :func:`fix_filename` (the port
of `dynamo-export::matlab_layout::fix_filename`), which strips only
OS-invalid characters — brackets, hyphens, parens, and spaces in channel
labels are preserved.

These builders are pure (no directories are created); the writers in
the sibling modules create parents on demand.
"""

from __future__ import annotations

from pathlib import Path

#: Union of Windows-invalid (`< > : " / \ | ? *`) and POSIX-invalid (`/`)
#: filename characters. Control chars are handled separately.
_DENY = set('<>:"/\\|?*')

#: Reserved DOS device names (case-insensitive) that Windows refuses as
#: file or folder names.
_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}

_AXIS_TOKEN = {"power": "SOpower", "phase": "SOphase"}


def fix_filename(s: str) -> str:
    """Sanitize one path component, stripping only OS-invalid characters.

    Port of `matlab_layout::fix_filename`: the deny-list characters and
    control chars become `_`, leading dots and trailing dots/spaces are
    trimmed (Windows silently strips them), consecutive `_` runs
    collapse, reserved DOS names get an `_` suffix, and an empty result
    falls back to `"unnamed"`. Everything else — brackets, hyphens,
    parens, spaces — passes through unchanged.
    """
    out = "".join(
        "_" if (ch in _DENY or ord(ch) < 0x20) else ch for ch in s
    )
    out = out.lstrip(".")
    out = out.rstrip(". ")
    out = out.strip()
    collapsed: list[str] = []
    prev_underscore = False
    for ch in out:
        if ch == "_":
            if not prev_underscore:
                collapsed.append("_")
            prev_underscore = True
        else:
            collapsed.append(ch)
            prev_underscore = False
    out = "".join(collapsed).strip()
    if out.upper() in _RESERVED:
        return out + "_"
    if not out:
        return "unnamed"
    return out


def _axis_token(axis: str) -> str:
    try:
        return _AXIS_TOKEN[axis]
    except KeyError:
        raise ValueError(
            f"axis must be 'power' or 'phase', got {axis!r}"
        ) from None


def channel_dir(root: Path | str, channel_label: str) -> Path:
    """`<root>/<channel>` — the per-channel artifact folder."""
    return Path(root) / fix_filename(channel_label)


def stats_csv_path(root: Path | str, subject_id: str, channel_label: str) -> Path:
    """`<channel>/TFpeaks/<subject>_stats_table_<channel>.csv` (§2.1)."""
    subject = fix_filename(subject_id)
    channel = fix_filename(channel_label)
    return (channel_dir(root, channel_label) / "TFpeaks"
            / f"{subject}_stats_table_{channel}.csv")


def soph_tiff_path(
    root: Path | str, subject_id: str, channel_label: str, axis: str,
) -> Path:
    """`<channel>/SOPHs/<subject>_SOPHs_{power,phase}_<channel>.tiff` (§2.2)."""
    _axis_token(axis)  # validate
    subject = fix_filename(subject_id)
    channel = fix_filename(channel_label)
    return (channel_dir(root, channel_label) / "SOPHs"
            / f"{subject}_SOPHs_{axis}_{channel}.tiff")


def aux_h5_path(root: Path | str, subject_id: str, channel_label: str) -> Path:
    """`<channel>/auxiliary_data/<subject>_auxiliary_data_<channel>.h5` (§2.3)."""
    subject = fix_filename(subject_id)
    channel = fix_filename(channel_label)
    return (channel_dir(root, channel_label) / "auxiliary_data"
            / f"{subject}_auxiliary_data_{channel}.h5")


def paramfit_csv_path(
    root: Path | str, subject_id: str, channel_label: str, axis: str,
) -> Path:
    """`<channel>/param_basis/<subject>_SO{power,phase}_paramfit_<channel>.csv` (§2.4)."""
    token = _axis_token(axis)
    subject = fix_filename(subject_id)
    channel = fix_filename(channel_label)
    return (channel_dir(root, channel_label) / "param_basis"
            / f"{subject}_{token}_paramfit_{channel}.csv")


def splinefit_tiff_path(
    root: Path | str, subject_id: str, channel_label: str, axis: str,
) -> Path:
    """`<channel>/spline_basis/<subject>_SO{power,phase}_splinefit_<channel>.tiff` (§2.5)."""
    token = _axis_token(axis)
    subject = fix_filename(subject_id)
    channel = fix_filename(channel_label)
    return (channel_dir(root, channel_label) / "spline_basis"
            / f"{subject}_{token}_splinefit_{channel}.tiff")


def runs_dir(root: Path | str) -> Path:
    """`<root>/_runs` — one .jsonl per batch invocation (§4.2)."""
    return Path(root) / "_runs"
