"""MATLAB parity for :mod:`pydynamo.stats`.

MATLAB is canonical for computation, so "the Python side agrees" has to
be checkable rather than asserted. These run against fixtures produced
by the real ``FDR_1D`` / ``FDR_2D`` / ``gpermtest2`` (regenerate with
``scripts/gen_stats_fixtures.m``).

Because :mod:`pydynamo.stats` is a thin wrapper over ``dynamo_rs``, this
suite validates the *kernel* through the Python API — the same code the
desktop app and CLI link. A pass here is therefore parity for all three
front ends at once, which is the whole point of not having a separate
Python implementation.

Cases deliberately span where a port drifts: both sides of ``ranksum``'s
exact/approximate switch (n1 + n2 = 20) and ``signrank``'s (n = 15),
heavy/light/no ties, samples containing NaN, fully-missing rows, and
both FDR dependence methods.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pydynamo.stats import StatsKernelUnavailable, fdr_1d, fdr_2d, gpermtest2

FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skipif(
    not FIXTURES.joinpath("stats_fdr1d.csv").exists(),
    reason="stats fixtures absent; run scripts/gen_stats_fixtures.m",
)


def _vec(s: str) -> np.ndarray:
    if not s.strip():
        return np.empty(0, dtype=float)
    return np.array([float(t) for t in s.split()], dtype=float)


def _rows(name: str):
    with (FIXTURES / name).open() as fh:
        header = fh.readline().rstrip("\n").split(",")
        for line in fh:
            if line.strip():
                yield dict(zip(header, line.rstrip("\n").split(",")))


def _agrees(got: float, want: float, rtol: float = 1e-9) -> bool:
    """NaN == NaN; otherwise relative agreement with an absolute floor."""
    if math.isnan(got) or math.isnan(want):
        return math.isnan(got) and math.isnan(want)
    return abs(got - want) <= max(rtol * abs(want), 1e-12)


def test_kernel_is_available():
    """The wrapper must fail loudly, not silently fall back."""
    try:
        fdr_1d(np.zeros((2, 4)), np.ones((2, 4)))
    except StatsKernelUnavailable as exc:  # pragma: no cover
        pytest.fail(f"dynamo_rs stats bindings missing: {exc}")


def test_fdr_1d_matches_matlab():
    n = 0
    worst_p = worst_adj = 0.0
    for row in _rows("stats_fdr1d.csv"):
        R, N1, N2 = int(row["nbins"]), int(row["ntrials1"]), int(row["ntrials2"])
        # MATLAB writes column-major; invert with Fortran order.
        g1 = _vec(row["g1"]).reshape((R, N1), order="F")
        g2 = _vec(row["g2"]).reshape((R, N2), order="F")
        res = fdr_1d(g1, g2, q=float(row["q"]), method=row["method"])

        for i, (g, w) in enumerate(zip(res.p_values, _vec(row["p_values"]))):
            assert _agrees(g, w), f"case {row['case']} bin {i}: raw p {g!r} vs {w!r}"
            if not math.isnan(w) and w > 0:
                worst_p = max(worst_p, abs(g - w) / w)
        for i, (g, w) in enumerate(zip(res.p_adj, _vec(row["p_adj"]))):
            assert _agrees(g, w), f"case {row['case']} bin {i}: adj p {g!r} vs {w!r}"
            if not math.isnan(w) and w > 0:
                worst_adj = max(worst_adj, abs(g - w) / w)
        assert np.array_equal(res.sigbins, _vec(row["sigbins"]).astype(bool)), (
            f"case {row['case']}: significance mask differs"
        )
        n += 1
    assert n >= 40, f"expected the full fixture, got {n}"
    print(
        f"\nFDR_1D parity: {n} families | worst rel err raw {worst_p:.3e}, "
        f"adj {worst_adj:.3e}"
    )


def test_fdr_2d_matches_matlab_including_the_paired_path():
    n = n_paired = 0
    for row in _rows("stats_fdr2d.csv"):
        R, C = int(row["R"]), int(row["C"])
        N1, N2 = int(row["N1"]), int(row["N2"])
        paired = bool(int(row["paired"]))
        g1 = _vec(row["g1"]).reshape((R, C, N1), order="F")
        g2 = _vec(row["g2"]).reshape((R, C, N2), order="F")
        res = fdr_2d(
            g1, g2, q=float(row["q"]), method=row["method"], paired=paired
        )
        want_p = _vec(row["p_values"]).reshape((R, C), order="F")
        want_adj = _vec(row["p_adj"]).reshape((R, C), order="F")
        want_sig = _vec(row["sigbins"]).reshape((R, C), order="F").astype(bool)

        for idx in np.ndindex(R, C):
            assert _agrees(res.p_values[idx], want_p[idx]), (
                f"case {row['case']} px {idx}: raw p "
                f"{res.p_values[idx]!r} vs {want_p[idx]!r}"
            )
            assert _agrees(res.p_adj[idx], want_adj[idx]), (
                f"case {row['case']} px {idx}: adj p "
                f"{res.p_adj[idx]!r} vs {want_adj[idx]!r}"
            )
        assert np.array_equal(res.sigbins, want_sig), (
            f"case {row['case']}: significance mask differs"
        )
        n += 1
        n_paired += paired
    assert n >= 20, f"expected the full fixture, got {n}"
    assert n_paired >= 8, f"too few paired cases: {n_paired}"
    print(f"\nFDR_2D parity: {n} grids ({n_paired} paired) checked")


def test_gpermtest2_agrees_with_matlab_on_statistic_and_decision():
    """Permutation tests share no RNG with MATLAB, so check what is
    deterministic (the observed statistic) exactly, and what is
    stochastic (the bound, and hence the decision) for agreement rather
    than identity."""
    n = 0
    for row in _rows("stats_gpermtest2.csv"):
        R, C = int(row["R"]), int(row["C"])
        N1, N2 = int(row["N1"]), int(row["N2"])
        g1 = _vec(row["g1"]).reshape((R, C, N1), order="F")
        g2 = _vec(row["g2"]).reshape((R, C, N2), order="F")
        res = gpermtest2(
            g1,
            g2,
            alpha_level=float(row["alpha"]),
            iterations=int(row["iterations"]),
            seed=12345,
        )
        want_stat = _vec(row["true_stat"]).reshape((R, C), order="F")
        want_bounds = _vec(row["bounds"]).reshape((R, C), order="F")
        want_sig = _vec(row["sigbins"]).reshape((R, C), order="F").astype(bool)

        # The observed statistic is deterministic — it must match exactly.
        for idx in np.ndindex(R, C):
            assert _agrees(res.true_stat[idx], want_stat[idx], 1e-12), (
                f"case {row['case']} px {idx}: true_stat "
                f"{res.true_stat[idx]!r} vs {want_stat[idx]!r}"
            )
        # The bound comes from a different permutation draw; require it
        # to land close, not to be identical.
        rel = np.abs(res.acceptance_bounds - want_bounds) / np.maximum(
            np.abs(want_bounds), 1e-12
        )
        assert np.median(rel) < 0.10, (
            f"case {row['case']}: bounds differ by {np.median(rel):.1%} (median)"
        )
        # And the decision must agree.
        assert np.array_equal(res.sigbins, want_sig), (
            f"case {row['case']}: significance mask differs "
            f"({int(res.sigbins.sum())} vs {int(want_sig.sum())} flagged)"
        )
        n += 1
    assert n >= 4, f"expected the full fixture, got {n}"
    print(f"\ngpermtest2 parity: {n} grids checked")


def test_fixture_covers_both_sides_of_the_exact_switches():
    """Pin the coverage rather than trusting the random draw to hit it."""
    small = large = tied = 0
    for row in _rows("stats_ranksum.csv"):
        x, y = _vec(row["x"]), _vec(row["y"])
        n = int(np.isfinite(x).sum() + np.isfinite(y).sum())
        small += n < 20
        large += n >= 20
        both = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
        tied += len(np.unique(both)) < both.size
    assert small >= 25, f"too few exact-branch ranksum cases: {small}"
    assert large >= 25, f"too few approximate-branch ranksum cases: {large}"
    assert tied >= 25, f"too few tied ranksum cases: {tied}"

    s_small = s_large = 0
    for row in _rows("stats_signrank.csv"):
        d = _vec(row["x"]) - _vec(row["y"])
        d = d[np.isfinite(d)]
        n = int((np.abs(d) > 0).sum())
        s_small += n <= 15
        s_large += n > 15
    assert s_small >= 25, f"too few exact-branch signrank cases: {s_small}"
    assert s_large >= 25, f"too few approximate-branch signrank cases: {s_large}"
    print(
        f"\ncoverage — ranksum: {small} exact / {large} approx / {tied} tied; "
        f"signrank: {s_small} exact / {s_large} approx"
    )
