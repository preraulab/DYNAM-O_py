"""Global permutation testing — ``gpermtest.m`` / ``gpermtest2.m``.

A wrapper over ``dynamo_rs``, for the same reason as :mod:`.fdr`: one
implementation, three front ends, no drift.

Where :mod:`pydynamo.stats.fdr` controls the *false discovery rate* (the
expected share of flagged bins that are false), this controls the
*family-wise error rate* (the probability of any false positive
anywhere) by the max-statistic method: a single global threshold chosen
so that across permutations of the group labels at most ``alpha`` of
them have *any* bin exceeding it.

That distinction is how you pick between them. FDR is more sensitive and
answers per bin. The global bound is stricter, assumes only that the
group labels are exchangeable, and gives you one line to draw over the
whole map — anything past it is significant.

The statistic is fixed to the NaN-omitting mean across observations.
MATLAB accepts an arbitrary ``statfcn`` handle; that has no clean
analogue across an FFI boundary, and DYNAM-O only ever uses the default.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .fdr import StatsKernelUnavailable, _kernel

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_ITERATIONS",
    "PermTestResult",
    "StatsKernelUnavailable",
    "gpermtest",
    "gpermtest2",
]

#: MATLAB default global acceptance level.
DEFAULT_ALPHA = 0.05
#: MATLAB default permutation count.
DEFAULT_ITERATIONS = 10000


@dataclass
class PermTestResult:
    """Outcome of a global permutation test."""

    sigbins: np.ndarray
    """Boolean mask: ``|true_stat| >= acceptance_bounds``."""
    acceptance_bounds: np.ndarray
    """Per-bin global bound at the requested alpha."""
    true_stat: np.ndarray
    """Observed ``mean(g1) - mean(g2)`` per bin, NaN-omitting."""
    n_excluded: int
    """Permutations excluded by the final bound — the achieved alpha
    numerator. Compare ``n_excluded / iterations`` against ``alpha``."""


def _run(g1, g2, alpha, iterations, seed, grid_shape):
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    n_bins = int(np.prod(grid_shape))
    sig, bounds, true_stat, n_excluded, warning = _kernel().gpermtest(
        np.ascontiguousarray(g1.reshape(n_bins, -1)).ravel(),
        np.ascontiguousarray(g2.reshape(n_bins, -1)).ravel(),
        n_bins,
        float(alpha),
        int(iterations),
        int(seed),
    )
    if warning:
        warnings.warn(warning, RuntimeWarning, stacklevel=3)
    return PermTestResult(
        np.asarray(sig, dtype=bool).reshape(grid_shape),
        np.asarray(bounds).reshape(grid_shape),
        np.asarray(true_stat).reshape(grid_shape),
        int(n_excluded),
    )


def gpermtest(
    group1,
    group2,
    alpha_level: float = DEFAULT_ALPHA,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = 0,
) -> PermTestResult:
    """Global (max-statistic) permutation test — ``gpermtest.m``.

    Parameters
    ----------
    group1, group2 : array_like, shape (n_bins, n_trials)
        Observations that are entirely NaN across bins are dropped first,
        as MATLAB does.
    alpha_level : float
        Family-wise error rate for the global bound.
    iterations : int
        Permutations used to build the null.
    seed : int
        Seed for the permutation RNG, so a run is reproducible. Results
        are *not* expected to match MATLAB draw-for-draw — different
        generators — only to agree in distribution.

    Notes
    -----
    Emits a :class:`RuntimeWarning` when the requested alpha cannot be
    resolved on the permutation grid (too few iterations for the alpha,
    or an alpha so large the bound would exclude everything), and when
    the achieved rate misses the request by more than 0.01. Both mean the
    same thing: raise ``iterations`` or relax ``alpha``.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    if g1.ndim != 2 or g2.ndim != 2:
        raise ValueError("group1/group2 must be 2-D (n_bins, n_trials)")
    if g1.shape[0] != g2.shape[0]:
        raise ValueError(f"bin count differs: {g1.shape[0]} vs {g2.shape[0]}")
    return _run(g1, g2, alpha_level, iterations, seed, (g1.shape[0],))


def gpermtest2(
    group1,
    group2,
    alpha_level: float = DEFAULT_ALPHA,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = 0,
) -> PermTestResult:
    """2-D version of :func:`gpermtest` — ``gpermtest2.m``.

    Parameters
    ----------
    group1, group2 : array_like, shape (n_rows, n_cols, n_trials)
        For a SOPH that is ``(n_so_bins, n_freq_bins, n_subjects)``.

    Notes
    -----
    Like ``gpermtest2.m``, the grid is flattened before the test, so the
    max statistic is taken over the whole map — the bound is global
    across both axes, not per row.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    if g1.ndim != 3 or g2.ndim != 3:
        raise ValueError("group1/group2 must be 3-D (n_rows, n_cols, n_trials)")
    if g1.shape[:2] != g2.shape[:2]:
        raise ValueError(f"grid dims differ: {g1.shape[:2]} vs {g2.shape[:2]}")
    return _run(g1, g2, alpha_level, iterations, seed, g1.shape[:2])
