"""DYNAM-O settings: the same settings files the DYNAM-O app and CLI use.

A settings file (``batch_settings.json`` / a CLI manifest) holds the run's
options under its ``settings`` key, stamped with the ``semantics_version``
it was written under. pydynamo passes that section to the kernel as it is,
so a pydynamo run and an app run of the same file use the same options.
Keys a section omits take the kernel's defaults; the DesktopApp's
``documents/SETTINGS_REFERENCE.md`` lists every key.
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

from pydynamo import _kernel

#: The computation-semantics version of the installed kernel. Outputs
#: stamped with the same number are comparable; a settings file written
#: under an older one is migrated on load (see :func:`load_settings`).
SEMANTICS_VERSION: int = int(_kernel.rs().SEMANTICS_VERSION)


def default_settings() -> dict:
    """The default ``settings`` section, as a dict."""
    return json.loads(_kernel.rs().default_settings())


def merge_settings(base: dict, overrides: dict) -> dict:
    """``base`` with ``overrides`` merged in, key by key.

    Two dicts merge recursively, so keys only ``base`` has are kept;
    anything else (a number, a list) is replaced by the override. This is
    how the DYNAM-O CLI layers ``--override-file`` onto a settings file.
    Neither argument is modified.
    """
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = merge_settings(base_value, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def migrate_settings(section: dict, from_version: int) -> tuple[dict, list[str]]:
    """Bring a ``settings`` section written under semantics ``from_version``
    up to :data:`SEMANTICS_VERSION`. Returns ``(section, log)``, one log
    line per step: the migrations ``dynamo-cli migrate-settings`` applies.
    """
    text, log = _kernel.rs().migrate_settings(json.dumps(section), int(from_version))
    return json.loads(text), list(log)


def load_settings(source) -> dict:
    """The ``settings`` section to run with, from a settings file.

    ``source`` is a path to a settings file, or an already-parsed dict:
    either a whole file (with ``settings`` and ``semantics_version``) or
    a bare section. A file written under an older semantics version is
    migrated, with a warning listing the steps. A bare section is taken
    as current.
    """
    if isinstance(source, (str, Path)):
        with open(source, encoding="utf-8") as f:
            document = json.load(f)
    else:
        document = source

    if "settings" not in document:
        return copy.deepcopy(document)

    section = document["settings"] or {}
    written_under = int(document.get("semantics_version") or 0)
    if written_under < SEMANTICS_VERSION:
        section, log = migrate_settings(section, written_under)
        warnings.warn(
            f"settings written under semantics v{written_under}, migrated to "
            f"v{SEMANTICS_VERSION}:\n  " + "\n  ".join(log),
            UserWarning,
            stacklevel=2,
        )
    return section
