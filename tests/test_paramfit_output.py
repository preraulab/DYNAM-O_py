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
    assert power.loc[0, "Volume"] == pytest.approx(24.0 * np.pi)

    phase_params = np.array([[0.05, 11.0, 2.0, 0.0, 1.2, 0.0]])
    phase = create_params_table(phase_params, "phase")
    kappa = 1.0 / 1.2**2
    expected = 0.05 * 2.0 * np.pi * i0e(kappa) * np.sqrt(np.pi) * 2.0
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
        "PeakFrequency": [10.0, 11.0, 14.0],
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


def test_phase_peak_assignment_uses_probability_and_circular_geometry():
    table = create_params_table(
        np.array([[1.0, 10.0, 2.0, 0.0, 1.0, 0.0]]), "phase"
    )
    stats = pd.DataFrame({
        "PeakFrequency": [10.0, 10.0],
        "SOphase": [0.0, np.pi],
    })

    wide = annotate_modes_with_peak_stats(table, "phase", stats, 0.95)
    narrow = annotate_modes_with_peak_stats(table, "phase", stats, 0.5)

    assert wide.loc[0, "PkCount"] == 2.0
    assert narrow.loc[0, "PkCount"] == 1.0


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
    prob = 1.0 - np.exp(-1.0)

    out = annotate_modes_with_peak_stats(table, "power", stats, prob)

    assert out.loc[0, "PkCount"] == 1.0
    assert out.loc[0, "PkFreq"] == boundary_frequency


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
