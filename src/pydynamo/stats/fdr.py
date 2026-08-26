"""FDR-controlled group comparison — ``FDR_1D.m`` / ``FDR_2D.m``.

This is a **wrapper, not a reimplementation**. The arithmetic lives in
``dynamo_rs.stats`` (Rust), which the desktop app and CLI link directly,
so all three front ends run the same code and cannot drift from each
other. MATLAB remains canonical: the Rust port is checked against
MATLAB-generated fixtures, and this layer is checked against the same
ones (``tests/test_stats_parity.py``).

Unlike the rest of pydynamo there is **no pure-Python fallback**. A
fallback here would be exactly the second implementation this design
exists to avoid — and a statistics fallback that silently disagreed in
the tail would be far worse than an import error. Without the kernel
these raise :class:`StatsKernelUnavailable`.

**Missing data is dropped, never imputed.** A SOPH bin is a peak *rate*
whose NaN means "under ``min_time_in_bin``, rate undefined", while a real
``0`` means "occupied the bin, no peaks there". Filling NaN with 0
collapses the first onto the second, and since the values are
non-negative the injected zeros sit at the bottom of every rank ordering
and swamp the real observations. A bin with nothing left to compare is
untestable, returns NaN, and stays out of the FDR family — so it never
counts toward *m* and can never be flagged.

References
----------
Benjamini & Hochberg (1995), J. R. Stat. Soc. B 57(1)
Benjamini & Yekutieli (2001), Ann. Stat. 29(4)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_Q",
    "FdrResult",
    "StatsKernelUnavailable",
    "fdr_bh",
    "fdr_1d",
    "fdr_2d",
]

#: Default false-discovery rate, matching ``FDR_1D.m`` / ``FDR_2D.m`` and
#: ``stats::fdr::DEFAULT_Q`` on the Rust side.
DEFAULT_Q = 0.05


class StatsKernelUnavailable(RuntimeError):
    """Raised when ``dynamo_rs`` is missing or too old for these tests."""


def _kernel():
    try:
        import dynamo_rs
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise StatsKernelUnavailable(
            "pydynamo.stats requires the dynamo_rs kernel; install it with "
            "`maturin develop --release --features python` from DYNAM-O_rs/rust"
        ) from exc
    if not hasattr(dynamo_rs, "fdr_grid"):  # pragma: no cover
        raise StatsKernelUnavailable(
            "the installed dynamo_rs predates the stats bindings; rebuild it "
            "from a kernel that has `fdr_grid`"
        )
    return dynamo_rs


@dataclass
class FdrResult:
    """Outcome of an FDR-controlled comparison."""

    sigbins: np.ndarray
    """Boolean mask, ``p_adj < q``. Same shape as the bin grid."""
    p_adj: np.ndarray
    """Adjusted p-values; NaN where the bin was untestable. Not clipped
    to 1 — BY can exceed it, and MATLAB does not clip either."""
    p_values: np.ndarray
    """Raw per-bin p-values; NaN where untestable."""
    crit_p: float
    """Largest raw p that passes, or 0.0 when none do."""
    m: int
    """Family size — testable bins only."""


def fdr_bh(pvals, q: float = DEFAULT_Q, method: str = "dep"):
    """Benjamini–Hochberg / Benjamini–Yekutieli adjustment.

    Parameters
    ----------
    pvals : array_like
        Raw p-values, any shape. Non-finite entries are treated as
        untestable: excluded before the family size is counted, and
        returned as NaN.
    q : float
        False-discovery rate.
    method : {'dep', 'dependent', 'pdep', 'independent'}
        ``'dep'`` (default) is Benjamini–Yekutieli, valid under arbitrary
        dependence — the right choice for neighbouring SOPH pixels and
        what ``FDR_2D`` defaults to. ``'pdep'`` is Benjamini–Hochberg,
        valid under independence or positive dependence.

    Returns
    -------
    (p_adj, crit_p, m)
    """
    p = np.ascontiguousarray(np.asarray(pvals, dtype=float))
    p_adj, crit_p, m = _kernel().fdr_adjust(p.ravel(), float(q), str(method))
    return np.asarray(p_adj).reshape(p.shape), float(crit_p), int(m)


def _run(g1, g2, q, method, paired, nonparam, grid_shape):
    if not nonparam:
        raise NotImplementedError(
            "parametric (nonparam=False) FDR is not implemented in the kernel; "
            "use the MATLAB toolbox for t-test comparisons"
        )
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    n_bins = int(np.prod(grid_shape))
    # The kernel takes row-major (n_bins, n_trials); C order gives that.
    p_values, p_adj, sig, crit_p, m = _kernel().fdr_grid(
        np.ascontiguousarray(g1.reshape(n_bins, -1)).ravel(),
        np.ascontiguousarray(g2.reshape(n_bins, -1)).ravel(),
        n_bins,
        float(q),
        str(method),
        bool(paired),
    )
    return FdrResult(
        np.asarray(sig, dtype=bool).reshape(grid_shape),
        np.asarray(p_adj).reshape(grid_shape),
        np.asarray(p_values).reshape(grid_shape),
        float(crit_p),
        int(m),
    )


def fdr_1d(
    group1,
    group2,
    q: float = DEFAULT_Q,
    method: str = "dep",
    paired: bool = False,
    nonparam: bool = True,
) -> FdrResult:
    """Per-bin two-group comparison with FDR control — ``FDR_1D.m``.

    Parameters
    ----------
    group1, group2 : array_like, shape (n_bins, n_trials)
        One row per bin, one column per subject/trial. Both need the same
        number of bins; ``paired=True`` also needs matching trial counts.
    q : float
        False-discovery rate (default 0.05, matching MATLAB).
    method : str
        Dependence structure — see :func:`fdr_bh`.
    paired : bool
        Wilcoxon signed-rank rather than rank-sum.
    nonparam : bool
        Must be True; the kernel implements only the rank tests, which is
        all ``FDR_1D``'s default path uses.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    if g1.ndim != 2 or g2.ndim != 2:
        raise ValueError("group1/group2 must be 2-D (n_bins, n_trials)")
    if g1.shape[0] != g2.shape[0]:
        raise ValueError(f"bin count differs: {g1.shape[0]} vs {g2.shape[0]}")
    return _run(g1, g2, q, method, paired, nonparam, (g1.shape[0],))


def fdr_2d(
    group1,
    group2,
    q: float = DEFAULT_Q,
    method: str = "dep",
    paired: bool = False,
    nonparam: bool = True,
) -> FdrResult:
    """2-D (e.g. SOPH) version of :func:`fdr_1d` — ``FDR_2D.m``.

    Parameters
    ----------
    group1, group2 : array_like, shape (n_rows, n_cols, n_trials)
        For a SOPH that is ``(n_so_bins, n_freq_bins, n_subjects)``.

    Notes
    -----
    Pixels are tested independently and the whole grid forms **one** FDR
    family, exactly as ``FDR_2D.m`` does by reshaping to
    ``(n_rows * n_cols, n_trials)`` and delegating to ``FDR_1D``.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    if g1.ndim != 3 or g2.ndim != 3:
        raise ValueError("group1/group2 must be 3-D (n_rows, n_cols, n_trials)")
    if g1.shape[:2] != g2.shape[:2]:
        raise ValueError(f"grid dims differ: {g1.shape[:2]} vs {g2.shape[:2]}")
    return _run(g1, g2, q, method, paired, nonparam, g1.shape[:2])
