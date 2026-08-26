"""Group-comparison statistics, ported from the MATLAB toolbox.

MATLAB is canonical for computation in DYNAM-O, so these are ports of
``toolbox/helper_functions/statistical_tests`` rather than independent
implementations, and they are checked against MATLAB-generated fixtures
(``tests/test_stats_parity.py``).

Two families, controlling different things:

* :func:`fdr_1d` / :func:`fdr_2d` — per-bin rank (or t) tests with
  Benjamini–Hochberg / Benjamini–Yekutieli control of the false
  discovery rate. More sensitive; the answer is a per-bin adjusted p.
* :func:`gpermtest` / :func:`gpermtest2` — max-statistic permutation
  tests with family-wise control. Stricter; the answer is a single
  global bound you can draw over the whole map, assuming only that the
  group labels are exchangeable.

These delegate to ``dynamo_rs.stats`` — the same Rust the desktop app and
CLI link — so the three front ends cannot drift from one another, and
MATLAB parity has to be established once rather than three times.

There is deliberately **no pure-Python fallback**, unlike the rest of
pydynamo. A fallback would reintroduce the second implementation this
design exists to remove, and one that disagreed only in the tail of a
distribution would be worse than an import error. Without the kernel
these raise :class:`StatsKernelUnavailable`.
"""

from .fdr import (
    DEFAULT_Q,
    FdrResult,
    StatsKernelUnavailable,
    fdr_1d,
    fdr_2d,
    fdr_bh,
)
from .permtest import (
    DEFAULT_ALPHA,
    DEFAULT_ITERATIONS,
    PermTestResult,
    gpermtest,
    gpermtest2,
)

__all__ = [
    "DEFAULT_Q",
    "FdrResult",
    "StatsKernelUnavailable",
    "fdr_bh",
    "fdr_1d",
    "fdr_2d",
    "DEFAULT_ALPHA",
    "DEFAULT_ITERATIONS",
    "PermTestResult",
    "gpermtest",
    "gpermtest2",
]
