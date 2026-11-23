"""Loading a recording: one EDF channel plus its staging.

The kernel loads it (``dynamo_rs.load_recording``) exactly as the DYNAM-O
app does: channel expressions and references, staging clock times aligned
to the EDF recording start, an all-N2 hypnogram when there is no staging
file. So a pydynamo run starts from the same samples and stages as an app
run of the same files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pydynamo import _kernel

#: Stage-label groups in the order the staging reader takes them.
STAGE_LABEL_ORDER = ("artifact", "wake", "rem", "n1", "n2", "n3", "unknown")


@dataclass
class Recording:
    """One channel of a recording, ready for :func:`pydynamo.run_dynamo`."""
    data: np.ndarray
    fs: float
    stage_times: np.ndarray
    stage_vals: np.ndarray


def _stage_label_lists(stage_labels) -> list[list[str]]:
    """The reader's seven alias lists from the settings-file form
    (``{"wake": "wake, W, 5", ...}``) or from seven lists."""
    if stage_labels is None:
        return []
    if isinstance(stage_labels, dict):
        lists = []
        for group in STAGE_LABEL_ORDER:
            aliases = []
            for alias in str(stage_labels.get(group, "")).split(","):
                if alias.strip():
                    aliases.append(alias.strip())
            lists.append(aliases)
        return lists
    lists = []
    for group in stage_labels:
        lists.append([str(alias) for alias in group])
    return lists


def load_recording(
    edf_path,
    channel: str,
    staging_path=None,
    *,
    references=(),
    time_col: int = 1,
    stage_col: int = 2,
    header_lines: int = 0,
    delimiter: str = ",",
    epoch_dur: float = 30.0,
    stage_labels=None,
) -> Recording:
    """Load one channel of ``edf_path`` and its staging.

    ``channel`` is an EDF label or expression as in a settings file
    (``C3``, ``C3-A2``, ``mean(C3, C4)``, ``$[C3-A2 - B]$`` for a literal
    label), ``references`` the settings file's ``NAME = expr`` list.
    ``time_col`` / ``stage_col`` / ``header_lines`` / ``delimiter``
    describe the staging file as the settings file's ``staging`` block
    does (``delimiter`` accepts ``"tab"``, ``"comma"``, ``"space"``).
    ``stage_labels`` is the settings file's ``stage_labels`` dict (or
    seven alias lists in :data:`STAGE_LABEL_ORDER`); None uses the
    reader's built-in labels. Without ``staging_path`` every epoch is N2.
    """
    out = _kernel.rs().load_recording(
        str(edf_path),
        channel,
        None if staging_path is None else str(staging_path),
        [str(r) for r in references],
        int(time_col),
        int(stage_col),
        int(header_lines),
        str(delimiter),
        float(epoch_dur),
        _stage_label_lists(stage_labels),
    )
    return Recording(
        data=np.asarray(out["data"]),
        fs=float(out["fs"]),
        stage_times=np.asarray(out["stage_times"]),
        stage_vals=np.asarray(out["stage_vals"]),
    )


def load_from_settings_file(document: dict, edf_path, staging_path=None, channel=None) -> Recording:
    """Load ``edf_path`` with the channel, references, staging layout and
    stage labels of a whole settings file (``document``, parsed JSON).
    ``channel`` defaults to the file's first channel."""
    staging = document.get("staging") or {}
    if channel is None:
        channel = document["channels"][0]
    return load_recording(
        edf_path,
        channel,
        staging_path,
        references=document.get("references") or [],
        time_col=staging.get("time_col", 1),
        stage_col=staging.get("stage_col", 2),
        header_lines=staging.get("header_lines", 0),
        delimiter=staging.get("delim", ","),
        stage_labels=document.get("stage_labels"),
    )


__all__ = ["Recording", "STAGE_LABEL_ORDER", "load_recording", "load_from_settings_file"]
