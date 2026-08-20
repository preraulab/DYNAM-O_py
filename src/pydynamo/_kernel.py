"""Bookkeeping for the dynamo_rs kernel fallbacks.

Several modules delegate their hot path to the ``dynamo_rs`` Rust kernel
and keep a pure-Python fallback for environments where the extension is
not installed (see baseline.py, artifacts.py, tfpeaks/mask.py,
tfpeaks/merge.py, tfpeaks/trim.py, soph/histogram.py,
soph/paramfit/core.py). Numerically the fallbacks match the kernel, but
provenance stamps (OUTPUT_FORMAT.md §8.1) must record which code path
actually computed the numbers. Each fallback site calls
:func:`record_fallback` when it runs in place of the kernel; the first
call warns once and latches :data:`FALLBACK_ACTIVE`, which
``pydynamo.io.stamp.current_stamp`` reads to report
``kernel_version='python-native'``.
"""

from __future__ import annotations

import warnings

# True once any pure-Python fallback has computed in place of the Rust
# kernel in this process. Latching (never reset) is deliberate: a single
# fallback anywhere means downstream numbers can no longer be attributed
# to the kernel build.
FALLBACK_ACTIVE = False


def record_fallback(feature: str) -> None:
    """Note that ``feature`` ran its pure-Python fallback.

    Warns once per process (RuntimeWarning) and latches
    :data:`FALLBACK_ACTIVE` so provenance stamps report
    ``kernel_version='python-native'``.
    """
    global FALLBACK_ACTIVE
    if not FALLBACK_ACTIVE:
        FALLBACK_ACTIVE = True
        warnings.warn(
            f"dynamo_rs kernel unavailable for {feature}; using the "
            "pure-Python fallback. Artifacts written from this process "
            "will be stamped kernel_version='python-native'.",
            RuntimeWarning,
            stacklevel=3,
        )


def fallback_active() -> bool:
    """Return True if any pure-Python fallback has run in this process."""
    return FALLBACK_ACTIVE
