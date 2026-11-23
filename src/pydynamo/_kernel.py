"""Access to the ``dynamo_rs`` Rust kernel.

Every number pydynamo produces is computed by the kernel, the same code
the DYNAM-O desktop app and CLI run. There is no pure-Python fallback: a
second implementation is exactly what would let the toolboxes drift
apart. Without the kernel, pydynamo raises :class:`KernelUnavailable`.
"""

from __future__ import annotations

# The bindings this version of pydynamo calls. An older kernel build that
# lacks any of them is refused with a rebuild hint.
_REQUIRED = (
    "run_dynamo", "load_recording", "write_stats_csv", "fit_paramfits",
    "fit_splinefits", "rebuild_phase_model", "default_settings",
    "migrate_settings", "build_info",
)


class KernelUnavailable(ImportError):
    """The ``dynamo_rs`` kernel is missing or too old for this pydynamo."""


_module = None


def rs():
    """The ``dynamo_rs`` module, checked once for the bindings pydynamo needs."""
    global _module
    if _module is not None:
        return _module
    try:
        import dynamo_rs
    except ImportError as exc:  # pragma: no cover - needs a broken install
        raise KernelUnavailable(
            "pydynamo needs the dynamo_rs kernel; build it with the "
            "DYNAM-O_toolbox bootstrap (see README)."
        ) from exc
    missing = [name for name in _REQUIRED if not hasattr(dynamo_rs, name)]
    if missing:  # pragma: no cover - needs a stale build
        raise KernelUnavailable(
            "the installed dynamo_rs lacks " + ", ".join(missing)
            + "; rebuild it from DYNAM-O_rs (see README)."
        )
    _module = dynamo_rs
    return _module
