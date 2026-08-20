"""Provenance stamps for DYNAM-O artifacts (OUTPUT_FORMAT.md §8.1).

Every artifact records the same four keys in every carrier:

* ``format`` — integer schema version of the artifact type (per-artifact
  counter, §8.2).
* ``writer`` — the writing tool id; this package writes ``pydynamo``.
* ``writer_version`` — the writing tool's build, grammar
  ``<semver>+<sha12>[.dirty]`` (a plain version string is the best
  available when no git sha is known) or the fail-soft ``unknown``.
* ``kernel_version`` — the ``dynamo_rs`` build that computed the
  numbers, or the literal ``python-native`` when a pure-Python fallback
  computed them (see :mod:`pydynamo._kernel`).

Carriers: CSVs use leading ``# key: value`` preamble lines, TIFFs use
keys in the ImageDescription JSON, HDF5 uses top-level scalar datasets,
JSON files use top-level keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Iterable

from pydynamo import _kernel

#: Tool id stamped as ``writer`` on every artifact this package writes.
WRITER = "pydynamo"

#: Preamble keys that belong to the stamp. ``version`` is the legacy
#: paramfit spelling of ``format``; ``code_version`` the legacy spelling
#: of ``writer_version`` (§8.2 legacy formats).
_STAMP_KEYS = ("format", "version", "writer", "writer_version",
               "code_version", "kernel_version")


@dataclass
class Provenance:
    """The §8.1 stamp recovered from, or destined for, one artifact.

    All fields optional: readers never fail on a missing stamp (§8.3),
    and ``format`` is per-artifact so writers fill it themselves.
    """

    format: int | None = None
    writer: str | None = None
    writer_version: str | None = None
    kernel_version: str | None = None


def writer_version() -> str:
    """This package's build identity, best available.

    The installed distribution version (e.g. ``0.2.0``); ``unknown``
    when pydynamo is not an installed distribution.
    """
    try:
        return _pkg_version("pydynamo")
    except PackageNotFoundError:
        return "unknown"


def kernel_version() -> str:
    """Identity of the code that computed the numbers.

    * ``python-native`` when any pure-Python fallback has run in this
      process (:data:`pydynamo._kernel.FALLBACK_ACTIVE`);
    * otherwise the ``dynamo_rs`` build string
      (``<semver>+<sha12>[.dirty]``);
    * ``unknown`` when ``dynamo_rs`` is absent.
    """
    if _kernel.FALLBACK_ACTIVE:
        return "python-native"
    try:
        import dynamo_rs
    except ImportError:
        return "unknown"
    try:
        return str(dynamo_rs.build_info()["build_info"])
    except Exception:
        return "unknown"


def current_stamp(format: int | None = None) -> Provenance:
    """The stamp for artifacts written by this process right now.

    ``format`` is per-artifact; writers pass their own schema version
    and callers that only want the writer/kernel identity leave it None.
    """
    return Provenance(
        format=format,
        writer=WRITER,
        writer_version=writer_version(),
        kernel_version=kernel_version(),
    )


def parse_preamble(lines: Iterable[str]) -> tuple[Provenance, dict[str, str]]:
    """Parse leading ``# key: value`` preamble lines into a stamp.

    Consumes lines while they start with ``#`` and stops at the first
    line that does not (per §8.3 a ``#`` is only trusted at line start,
    and the preamble is strictly leading). Returns ``(stamp, extras)``
    where ``extras`` maps every non-stamp preamble key to its raw string
    value (``subjectID``, ``fit_type``, bin arrays, ...) for the caller
    to post-process; unknown keys are kept, title lines without a colon
    are skipped.

    Legacy spellings map in without clobbering the new keys:
    ``# version:`` fills ``format`` and ``# code_version:`` fills
    ``writer_version``, each only when the canonical key is absent.
    """
    stamp = Provenance()
    extras: dict[str, str] = {}
    for line in lines:
        if not line.startswith("#"):
            break
        body = line[1:].strip()
        key, sep, value = body.partition(":")
        if not sep:
            continue  # title line, e.g. "# DYNAM-O stats table"
        key = key.strip()
        value = value.strip()
        if key == "format" or key == "version":
            if stamp.format is None:
                try:
                    stamp.format = int(value)
                except ValueError:
                    pass
        elif key == "writer":
            stamp.writer = value
        elif key == "writer_version":
            stamp.writer_version = value
        elif key == "code_version":
            if stamp.writer_version is None:
                stamp.writer_version = value
        elif key == "kernel_version":
            stamp.kernel_version = value
        else:
            extras[key] = value
    return stamp, extras
