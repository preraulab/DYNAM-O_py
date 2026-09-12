"""MATLAB-compatible parametric-fit result packing and annotations."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import i0e

from pydynamo.soph.paramfit import (
    ParamBasisOpts,
    ParamFitResult,
    fit_param_basis_axis,
)
from pydynamo.soph.paramfit.output import (
    PK_COLUMNS,
    _mode_peak_mask,
    annotate_modes_with_peak_stats,
    annotate_power_preferred_phase,
    create_params_table,
)


POWER_COLUMNS = (
    "Density",
    "FreqMean",
    "FreqStd",
    "SOpowerMean",
    "SOpowerStd",
    "Theta",
    "Volume",
    "PrefPhase",
    "Coupling",
    "PkCount",
    "PkFreq",
    "PkDuration",
    "PkBandwidth",
    "PkHeight",
    "PkVolume",
    "PkArea",
    "PkPeakiness",
    "PkSOpower",
    "PkSOphase",
)

PHASE_COLUMNS = (
    "Density",
    "FreqMean",
    "FreqStd",
    "SOphaseMean",
    "SOphaseStd",
    "Theta",
    "Volume",
    "PkCount",
    "PkFreq",
    "PkDuration",
    "PkBandwidth",
    "PkHeight",
    "PkVolume",
    "PkArea",
    "PkPeakiness",
    "PkSOpower",
    "PkSOphase",
)


@pytest.mark.parametrize(
    ("kind", "columns"),
    [("power", POWER_COLUMNS), ("phase", PHASE_COLUMNS)],
)
def test_zero_mode_table_has_stable_float64_schema(kind, columns):
    table = create_params_table(np.empty((0, 6)), kind)

    assert table.shape == (0, len(columns))
    assert tuple(table.columns) == columns
    assert all(dtype == np.dtype(np.float64) for dtype in table.dtypes)


@pytest.mark.parametrize(
    ("kind", "columns"),
    [("power", POWER_COLUMNS), ("phase", PHASE_COLUMNS)],
)
def test_zero_mode_fit_result_exposes_stable_schema(kind, columns):
    x_bins = (
        np.array([-1.0, 1.0])
        if kind == "power"
        else np.array([-np.pi, np.pi])
    )
    freq_bins = np.array([2.0, 3.0, 4.0])
    soph = np.full((freq_bins.size, x_bins.size), np.nan)
    opts = (
        ParamBasisOpts.power()
        if kind == "power"
        else ParamBasisOpts.phase()
    )

    result = fit_param_basis_axis(soph, x_bins, freq_bins, opts)

    assert result.params.shape == (0, 6)
    assert tuple(result.params_table.columns) == columns
    assert result.params_table.empty


def test_param_fit_result_keeps_legacy_constructor_compatible():
    result = ParamFitResult(
        params=np.empty((0, 6)),
        background=np.zeros(3),
        model_soph=np.empty((0, 0)),
        gof={},
        wshed_img=None,
        fit_iteration=0,
        iter_numbers=[],
        iter_rsquared=[],
        n_wshed_modes=0,
    )

    assert isinstance(result.params_table, pd.DataFrame)
    assert result.params_table.empty


def test_analytic_volume_matches_matlab_formulas():
    power_params = np.array([[2.0, 10.0, 3.0, 5.0, 4.0, 0.2]])
    power = create_params_table(power_params, "power")
    assert power.loc[0, "Volume"] == pytest.approx(48.0 * np.pi)

    phase_params = np.array([[0.05, 11.0, 2.0, 0.0, 1.2, 0.0]])
    phase = create_params_table(phase_params, "phase")
    kappa = 1.0 / 1.2**2
    expected = (
        0.05 * 2.0 * np.pi * i0e(kappa) * np.sqrt(2.0 * np.pi) * 2.0
    )
    assert phase.loc[0, "Volume"] == pytest.approx(expected)

    sharp = create_params_table(
        np.array([[0.05, 11.0, 2.0, 0.0, 0.05, 0.0]]), "phase"
    )
    assert np.isfinite(sharp.loc[0, "Volume"])


def test_power_preferred_phase_uses_nearest_frequency_model_argmax():
    table = create_params_table(
        np.array([
            [2.0, 9.6, 1.0, 5.0, 4.0, 0.0],
            [1.0, 11.8, 1.0, 7.0, 3.0, 0.0],
        ]),
        "power",
    )
    freq_bins = np.array([8.0, 10.0, 12.0])
    phase_bins = np.array([-np.pi, -1.0, 1.0, np.pi])
    model = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.6, 0.2, 0.1],
        [0.7, 0.1, 0.1, 0.1],
    ])

    out = annotate_power_preferred_phase(
        table, model.T, freq_bins, phase_bins
    )

    assert np.allclose(out["PrefPhase"], [-1.0, -np.pi])
    assert np.allclose(out["Coupling"], [0.6, 0.7])


def test_missing_phase_model_leaves_preferred_phase_nan():
    table = create_params_table(
        np.array([[2.0, 10.0, 1.0, 5.0, 4.0, 0.0]]), "power"
    )

    out = annotate_power_preferred_phase(table, None, [10.0], None)

    assert out[["PrefPhase", "Coupling"]].isna().all().all()


def test_power_peak_summaries_use_finite_values_and_circular_phase_mean():
    table = create_params_table(
        np.array([[1.0, 10.0, 2.0, 5.0, 4.0, 0.0]]), "power"
    )
    stats = pd.DataFrame({
        "PeakFrequency": [10.0, 11.0, 16.0],
        "Duration": [1.0, 3.0, 100.0],
        "Bandwidth": [2.0, 4.0, 100.0],
        "Height": [10.0, np.nan, 100.0],
        "Volume": [5.0, 7.0, 100.0],
        "Area": [4.0, 8.0, 100.0],
        "Peakiness": [1.0, 3.0, 100.0],
        "SOpower": [5.0, 5.0, 5.0],
        "SOphase": [np.pi - 0.1, -np.pi + 0.1, 0.0],
    })

    out = annotate_modes_with_peak_stats(table, "power", stats, 0.95)

    assert out.loc[0, "PkCount"] == 2.0
    assert out.loc[0, "PkFreq"] == pytest.approx(10.5)
    assert out.loc[0, "PkDuration"] == pytest.approx(2.0)
    assert out.loc[0, "PkBandwidth"] == pytest.approx(3.0)
    assert out.loc[0, "PkHeight"] == pytest.approx(10.0)
    assert out.loc[0, "PkVolume"] == pytest.approx(6.0)
    assert out.loc[0, "PkArea"] == pytest.approx(6.0)
    assert out.loc[0, "PkPeakiness"] == pytest.approx(2.0)
    assert out.loc[0, "PkSOpower"] == pytest.approx(5.0)
    phase_error = np.angle(
        np.exp(1j * (out.loc[0, "PkSOphase"] - np.pi))
    )
    assert abs(phase_error) < 1e-12


def test_phase_peak_assignment_matches_kernel_height_boundaries():
    table = create_params_table(
        np.array([[1.0, 10.0, 2.0, 0.0, np.sqrt(2.0), 0.0]]), "phase"
    )
    stats = pd.DataFrame({
        "PeakFrequency": [12.0, 10.0, 10.0, 12.02],
        "SOphase": [0.0, np.pi / 2.0, np.pi, 0.0],
    })
    prob = -np.expm1(-0.5)

    out = annotate_modes_with_peak_stats(table, "phase", stats, prob)

    assert out.loc[0, "PkCount"] == 2.0
    assert out.loc[0, "PkFreq"] == pytest.approx(11.0)


def test_peak_assignment_includes_exact_confidence_boundary():
    table = create_params_table(
        np.array([[1.0, 10.0, 2.0, 5.0, 4.0, 0.0]]), "power"
    )
    boundary_frequency = 12.0
    stats = pd.DataFrame({
        "PeakFrequency": [
            boundary_frequency,
            np.nextafter(boundary_frequency, np.inf),
        ],
        "SOpower": [5.0, 5.0],
    })
    prob = -np.expm1(-0.5)

    out = annotate_modes_with_peak_stats(table, "power", stats, prob)

    assert out.loc[0, "PkCount"] == 1.0
    assert out.loc[0, "PkFreq"] == boundary_frequency


def test_power_containment_encloses_requested_probability():
    fmean, fstd = 13.0, 1.3
    pmean, pstd = 5.0, 7.0
    mode_params = np.array([1.0, fmean, fstd, pmean, pstd, 0.0])

    edges = np.linspace(-8.0, 8.0, 802)
    z = (edges[:-1] + edges[1:]) / 2.0
    z_power, z_freq = np.meshgrid(z, z)
    stats = pd.DataFrame({
        "PeakFrequency": fmean + z_freq.ravel() * fstd,
        "SOpower": pmean + z_power.ravel() * pstd,
    })
    density = np.exp(-0.5 * (z_freq.ravel() ** 2 + z_power.ravel() ** 2))
    total = density.sum()

    for prob in (0.5, 0.8, 0.95, 0.99):
        members = _mode_peak_mask(mode_params, "power", stats, prob)
        captured = density[members].sum() / total
        assert captured == pytest.approx(prob, abs=2e-3)


def test_missing_peak_stats_use_zero_count_and_nan_summaries():
    table = create_params_table(
        np.array([[1.0, 10.0, 2.0, 5.0, 4.0, 0.0]]), "power"
    )

    out = annotate_modes_with_peak_stats(table, "power", None, 0.95)

    assert out.loc[0, "PkCount"] == 0.0
    assert out[list(PK_COLUMNS[1:])].isna().all().all()


@pytest.mark.parametrize(
    "value", [0.0, 1.0, -0.1, 1.1, np.nan, True, "0.5"]
)
def test_peak_assign_prob_must_be_strictly_between_zero_and_one(value):
    with pytest.raises(ValueError, match="peak_assign_prob"):
        ParamBasisOpts.power(peak_assign_prob=value)


def test_peak_assign_prob_accepts_numpy_float():
    opts = ParamBasisOpts.phase(peak_assign_prob=np.float64(0.75))
    assert opts.peak_assign_prob == 0.75


# ---- peak_assign rule engine (E1-E9; the same analytic fixtures are ----
# ---- pinned by the Rust mode_peaks tests and the MATLAB tests)      ----
#
# Power fixture, theta = 0:
#   Mode A: amp 10 at (13 Hz, 5 dB),  sigma_f = 1, sigma_p = 10
#   Mode B: amp  5 at (13 Hz, 25 dB), sigma_f = 1, sigma_p = 10

from pydynamo.soph.paramfit.opts import parse_peak_assign
from pydynamo.soph.paramfit.output import _assign_mode_peaks

MODE_A = [10.0, 13.0, 1.0, 5.0, 10.0, 0.0]
MODE_B = [5.0, 13.0, 1.0, 25.0, 10.0, 0.0]


def _power_stats(points):
    return pd.DataFrame(
        {"PeakFrequency": [p[0] for p in points],
         "SOpower": [p[1] for p in points]}
    )


def test_e1_background_rule_equals_pstar_contour():
    # Constant bg 0.5, amp 10 -> p* = 0.95: boundary at 13 + sqrt(2*ln 20).
    r = np.sqrt(2.0 * np.log(20.0))
    stats = _power_stats([(13.0 + r - 1e-9, 5.0), (13.0 + r + 1e-3, 5.0)])
    m = _assign_mode_peaks([MODE_A], "power", stats, "background 3 s", [0, 0, 0.5])
    assert m[:, 0].tolist() == [True, False]


def test_e2_background_rule_adapts_to_amp():
    # amp 5 vs bg 0.5 -> p* = 0.9: power boundary at 25 + sqrt(2*ln 10)*10.
    b = 25.0 + np.sqrt(2.0 * np.log(10.0)) * 10.0
    stats = _power_stats([(13.0, b - 0.01), (13.0, b + 0.01)])
    m = _assign_mode_peaks([MODE_B], "power", stats, "background 3 s", [0, 0, 0.5])
    assert m[:, 0].tolist() == [True, False]


def test_e3_background_evaluated_at_the_peak():
    stats = _power_stats([(13.0, 25.0)])  # d_A ~ 1.3534 there
    sloped = _assign_mode_peaks([MODE_A], "power", stats, "background 3 s", [0.1, 0, 0])
    assert not sloped[0, 0]
    flat = _assign_mode_peaks([MODE_A], "power", stats, "background 3 s", [0, 0, 0.5])
    assert flat[0, 0]


def test_e4_argmax_is_exclusive():
    stats = _power_stats([(13.0, 15.0)])
    both = _assign_mode_peaks([MODE_A, MODE_B], "power", stats, 0.95)
    assert both[0].tolist() == [True, True]
    ex = _assign_mode_peaks([MODE_A, MODE_B], "power", stats, "argmax 3 s", [0, 0, 0.5])
    assert ex[0].tolist() == [True, False]


def test_e5_background_wins_leaves_unassigned():
    stats = _power_stats([(13.0, 60.0)])
    for rule in ("argmax 3 s", "background 3 s"):
        m = _assign_mode_peaks([MODE_A, MODE_B], "power", stats, rule, [0, 0, 0.5])
        assert m[0].tolist() == [False, False]


def test_e6_argmax_tie_is_first_wins():
    stats = _power_stats([(13.0, 5.0)])
    ex = _assign_mode_peaks([MODE_A, MODE_A], "power", stats, "argmax 3 s", [0, 0, 0.5])
    assert ex[0].tolist() == [True, False]


def test_e7_phase_axis_falls_back_to_prob():
    mode = [0.5, 13.0, 1.0, 0.0, 0.5, 0.0]
    stats = pd.DataFrame(
        {"PeakFrequency": [13.0, 13.0, 13.0], "SOphase": [0.0, 0.2, np.pi]}
    )
    fallback = _assign_mode_peaks([mode], "phase", stats, 0.95)
    for rule in ("background", "argmax", "argmax 3 s"):
        m = _assign_mode_peaks([mode], "phase", stats, rule, [7, 7, 7])
        assert m.tolist() == fallback.tolist()


def test_e8_prob_form_matches_legacy_mask():
    stats = _power_stats([(13.0, 5.0), (13.0, 15.0), (2.0, -4.0), (13.4, 28.0)])
    m = _assign_mode_peaks([MODE_A, MODE_B], "power", stats, 0.95)
    for col, mode in ((0, MODE_A), (1, MODE_B)):
        legacy = _mode_peak_mask(np.array(mode), "power", stats, 0.95)
        assert m[:, col].tolist() == legacy.tolist()


def test_e9_sigma_form_is_an_exact_sigma_radius():
    # "1 sigma": thr = 0.5 -> freq boundary at 13 + 1*sigma_f exactly, and
    # Sigma(n) equals Prob(1 - exp(-n^2/2)) on the power axis.
    stats = _power_stats([(14.0 - 1e-9, 5.0), (14.0 + 1e-3, 5.0)])
    m = _assign_mode_peaks([MODE_A], "power", stats, "1 sigma")
    assert m[:, 0].tolist() == [True, False]
    via_p = _assign_mode_peaks([MODE_A], "power", stats, 1.0 - np.exp(-0.5))
    assert m.tolist() == via_p.tolist()


def test_e11_density_rules_respect_the_footprint_cap():
    # Near-zero background: a peak at 3.5 sigma (Q = 6.125 > 4.5) must be
    # excluded under both density rules; one at 2.9 sigma (Q = 4.205) stays.
    stats = _power_stats([(13.0, 5.0 + 2.9 * 10.0), (13.0, 5.0 + 3.5 * 10.0)])
    for rule in ("background 3 s", "argmax 3 s"):
        m = _assign_mode_peaks([MODE_A], "power", stats, rule, [0, 0, 1e-6])
        assert m[:, 0].tolist() == [True, False], rule
    # Bare density rules mean a 1.5-sigma footprint: 1.4 sigma in, 1.6 out.
    stats2 = _power_stats([(13.0, 5.0 + 1.4 * 10.0), (13.0, 5.0 + 1.6 * 10.0)])
    for rule in ("argmax", "background"):
        m = _assign_mode_peaks([MODE_A], "power", stats2, rule, [0, 0, 1e-6])
        assert m[:, 0].tolist() == [True, False], f"{rule} default footprint"


def test_e12_capped_out_mode_does_not_block_argmax():
    # A peak 4 sigma from strong mode A but at weak mode C's center: C wins
    # the argmax - A's extrapolated tail neither claims nor blocks it.
    mode_c = [0.01, 13.0, 1.0, 45.0, 10.0, 0.0]
    stats = _power_stats([(13.0, 45.0)])
    m = _assign_mode_peaks([MODE_A, mode_c], "power", stats, "argmax 3 s", [0, 0, 1e-6])
    assert m[0].tolist() == [False, True]


def test_e9_parser_forms_and_rejects():
    a = parse_peak_assign(0.3)
    assert (a["kind"], a["value"]) == ("p", 0.3)
    assert parse_peak_assign("0.3 P")["thr"] == pytest.approx(-np.log(0.7))
    assert parse_peak_assign(" 1.3 SIGMA ")["kind"] == "sigma"
    assert parse_peak_assign("1.3 s")["thr"] == pytest.approx(0.5 * 1.3**2)
    assert parse_peak_assign("Argmax")["label"] == "argmax 1.5 sigma"
    a = parse_peak_assign("argmax 2 s")
    assert (a["kind"], a["footprint"], a["q_cap"]) == ("argmax", 2.0, 2.0)
    assert parse_peak_assign("BACKGROUND 1.75 SIGMA")["footprint"] == 1.75
    assert parse_peak_assign("background")["label"] == "background 1.5 sigma"
    for bad in ("1.5 p", 1.5, "0 sigma", "-1 s", "nonsense", "3 q", "argmax 0 s", "argmax 2 p", "argmax 2"):
        with pytest.raises(ValueError):
            parse_peak_assign(bad)


def test_peak_assign_default_and_alias_mapping():
    assert ParamBasisOpts.power().peak_assign == "argmax"
    with pytest.warns(DeprecationWarning):
        opts = ParamBasisOpts.power(peak_assign_prob=0.8)
    assert opts.peak_assign == "0.8 p"
    # An explicit peak_assign wins over the deprecated alias.
    with pytest.warns(DeprecationWarning):
        opts = ParamBasisOpts.power(peak_assign="1.5 sigma", peak_assign_prob=0.8)
    assert opts.peak_assign == "1.5 sigma"
