"""MATLAB-parity contracts for the Python spline-basis wrapper."""

from pathlib import Path

import numpy as np
import pytest

from pydynamo.soph import splinefit
from pydynamo.soph.splinefit import SplineBasisOpts, fit_spline_basis


FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "DYNAM-O_rs"
    / "rust"
    / "tests"
    / "fixtures"
    / "spline"
)


def test_spline_defaults_match_matlab():
    power = SplineBasisOpts.power()
    assert power.kind == "power"
    assert power.feature_limits == (-2.0, 20.0)
    assert power.freq_limits == (2.0, 18.0)
    assert (power.num_knots_x, power.num_knots_y) == (5, 18)

    phase = SplineBasisOpts.phase()
    assert phase.kind == "phase"
    assert phase.feature_limits == pytest.approx((-np.pi, np.pi))
    assert phase.freq_limits == (2.0, 18.0)
    assert (phase.num_knots_x, phase.num_knots_y) == (5, 9)


@pytest.mark.parametrize("kind", ["power", "phase"])
def test_spline_wrapper_matches_matlab_fixtures(kind):
    if not FIXTURE_DIR.exists():
        pytest.skip(f"missing sibling Rust fixtures: {FIXTURE_DIR}")

    soph = np.load(FIXTURE_DIR / f"sim_{kind}_SOPH_input.npy")
    feature_bins = np.load(FIXTURE_DIR / f"sim_{kind}_feat_bins.npy")
    freq_bins = np.load(FIXTURE_DIR / f"sim_{kind}_freq_bins.npy")

    result = fit_spline_basis(soph, feature_bins, freq_bins, kind=kind)

    assert np.allclose(
        result.coefs,
        np.load(FIXTURE_DIR / f"sim_{kind}_coefs.npy"),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.allclose(
        result.splinefit,
        np.load(FIXTURE_DIR / f"sim_{kind}_fit.npy"),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.allclose(
        result.knots_x,
        np.load(FIXTURE_DIR / f"sim_{kind}_internal_knots_x.npy"),
        rtol=0.0,
        atol=1e-14,
    )
    assert np.allclose(
        result.knots_y,
        np.load(FIXTURE_DIR / f"sim_{kind}_internal_knots_y.npy"),
        rtol=0.0,
        atol=1e-14,
    )
    assert np.allclose(
        result.knots_x_aug,
        np.load(FIXTURE_DIR / f"sim_{kind}_knots_x_aug.npy"),
        rtol=0.0,
        atol=1e-14,
    )
    assert np.allclose(
        result.knots_y_aug,
        np.load(FIXTURE_DIR / f"sim_{kind}_knots_y_aug.npy"),
        rtol=0.0,
        atol=1e-14,
    )
    assert np.array_equal(result.fit_SOfeature_bins, feature_bins)
    assert np.array_equal(result.fit_freq_bins, freq_bins)
    assert result.spline_obj.form == "B-"
    assert result.spline_obj.dim == 1
    assert result.spline_obj.order == (4, 4)
    assert result.spline_obj.number == (
        result.coefs.shape[1], result.coefs.shape[0],
    )
    assert np.array_equal(result.spline_obj.coefs, result.coefs)
    assert np.array_equal(result.spline_obj.knots_x, result.knots_x_aug)
    assert np.array_equal(result.spline_obj.knots_y, result.knots_y_aug)
    assert np.allclose(
        result.spline_obj(feature_bins, freq_bins),
        result.splinefit,
        rtol=0.0,
        atol=1e-12,
    )

    expected_shape = (20, 7) if kind == "power" else (11, 7)
    assert result.coefs.shape == expected_shape
    assert result.splinefit.shape == (feature_bins.size, freq_bins.size)


def test_spline_wrapper_accepts_frequency_by_feature_orientation():
    if not FIXTURE_DIR.exists():
        pytest.skip(f"missing sibling Rust fixtures: {FIXTURE_DIR}")

    feature_bins = np.load(FIXTURE_DIR / "sim_power_feat_bins.npy")
    freq_bins = np.load(FIXTURE_DIR / "sim_power_freq_bins.npy")
    matlab_layout = np.load(FIXTURE_DIR / "sim_power_SOPH_input.npy")
    frequency_by_feature = np.load(FIXTURE_DIR / "sim_power_SOPH.npy")

    a = fit_spline_basis(matlab_layout, feature_bins, freq_bins, kind="power")
    b = fit_spline_basis(
        frequency_by_feature, feature_bins, freq_bins, kind="power",
    )

    assert np.array_equal(a.coefs, b.coefs)
    assert np.array_equal(a.splinefit, b.splinefit)


def test_spline_wrapper_filters_domain_and_builds_matlab_knots(monkeypatch):
    captured = {}

    class FakeRust:
        @staticmethod
        def fit_tensor_product_spline(
            soph,
            x_eval,
            y_eval,
            knots_x,
            knots_y,
            order,
            boundary_multiplicity,
        ):
            captured.update(
                soph=soph.copy(),
                x_eval=x_eval.copy(),
                y_eval=y_eval.copy(),
                knots_x=knots_x.copy(),
                knots_y=knots_y.copy(),
                order=order,
                boundary_multiplicity=boundary_multiplicity,
            )
            return {
                "coefs": np.zeros((knots_y.size, knots_x.size)),
                "splinefit": np.zeros((x_eval.size, y_eval.size)),
                "knots_x_aug": knots_x.copy(),
                "knots_y_aug": knots_y.copy(),
            }

    monkeypatch.setattr(splinefit, "_rs", FakeRust())

    feature_bins = np.array([-3.0, -2.0, 0.0, 20.0, 21.0])
    freq_bins = np.array([1.0, 2.0, 10.0, 18.0, 19.0])
    soph = np.arange(1.0, 26.0).reshape(5, 5)
    soph[2, :] = np.nan
    soph[1, 2] = np.nan

    result = fit_spline_basis(
        soph.T,
        feature_bins,
        freq_bins,
        opts=SplineBasisOpts.power(num_knots_x=2, num_knots_y=2),
    )

    assert np.array_equal(result.fit_SOfeature_bins, [-2.0, 20.0])
    assert np.array_equal(result.fit_freq_bins, [2.0, 18.0])
    assert np.array_equal(captured["x_eval"], [-2.0, 20.0])
    assert np.array_equal(captured["y_eval"], [2.0, 18.0])
    assert np.array_equal(captured["soph"], [[7.0, 17.0], [9.0, 19.0]])
    assert np.allclose(captured["knots_x"], [-2.1, -2.0, 20.0, 20.1])
    assert np.allclose(captured["knots_y"], [1.9, 2.0, 18.0, 18.1])
    assert captured["order"] == 4
    assert captured["boundary_multiplicity"] == 3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"freq_limits": (2.0, 2.0)}, "freq_limits"),
        ({"num_knots_x": 0}, "num_knots_x"),
    ],
)
def test_spline_options_reject_invalid_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SplineBasisOpts.power(**kwargs)


def test_spline_wrapper_rejects_invalid_kind():
    with pytest.raises(ValueError, match="kind"):
        fit_spline_basis(
            np.ones((2, 3)),
            np.arange(2.0),
            np.arange(3.0),
            kind="other",
        )
