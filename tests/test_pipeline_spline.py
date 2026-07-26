"""Focused default-wiring tests for spline fits in ``run_dynamo``."""

import numpy as np
import pandas as pd
import pytest

import pydynamo.pipeline as pipeline
from pydynamo.defaults import DetectionOpts


@pytest.fixture
def stubbed_pipeline(monkeypatch):
    calls = []
    power_result = object()

    monkeypatch.setattr(
        pipeline,
        "detect_artifacts",
        lambda data, _fs: np.zeros(data.size, dtype=bool),
    )
    monkeypatch.setattr(
        pipeline,
        "mtm_spectrogram",
        lambda *_args, **_kwargs: (
            np.ones((3, 3)),
            np.array([0.0, 1.0, 2.0]),
            np.array([2.0, 3.0, 4.0]),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_baseline",
        lambda spect, *_args, **_kwargs: np.ones((spect.shape[0], 1)),
    )
    monkeypatch.setattr(
        pipeline, "subtract_baseline", lambda spect, _baseline: spect,
    )
    monkeypatch.setattr(
        pipeline,
        "extract_tfpeaks",
        lambda spect, *_args, **_kwargs: (
            pd.DataFrame(),
            np.zeros(spect.shape, dtype=int),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_so_power",
        lambda *_args, **_kwargs: (
            np.zeros(3),
            np.array([0.0, 1.0, 2.0]),
            np.ones(3),
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_so_phase",
        lambda *_args, **_kwargs: (
            np.zeros(3),
            np.array([0.0, 1.0, 2.0]),
            np.ones(3),
            None,
        ),
    )

    freq_bins = np.array([2.0, 3.0, 4.0])

    def fake_histogram(feature_bins):
        return {
            "c_mat": np.ones((feature_bins.size, freq_bins.size)),
            "c_cbins": feature_bins,
            "freq_cbins": freq_bins,
            "time_in_bin": np.ones(feature_bins.size),
            "peak_at_freq": np.zeros(freq_bins.size),
            "peak_selection_inds": np.zeros(0, dtype=bool),
        }

    monkeypatch.setattr(
        pipeline,
        "so_power_histogram",
        lambda *_args, **_kwargs: fake_histogram(
            np.array([-2.0, 0.0, 20.0]),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "so_phase_histogram",
        lambda *_args, **_kwargs: fake_histogram(
            np.array([-np.pi, 0.0, np.pi]),
        ),
    )

    def fake_fit(soph, feature_bins, received_freq_bins, *, opts, kind):
        calls.append({
            "kind": kind,
            "soph": soph,
            "feature_bins": feature_bins,
            "freq_bins": received_freq_bins,
            "opts": opts,
        })
        if kind == "phase":
            raise RuntimeError("phase fixture failure")
        return power_result

    monkeypatch.setattr(pipeline, "_fit_spline_basis", fake_fit)
    return calls, power_result


def _run_stubbed_pipeline(**kwargs):
    return pipeline.run_dynamo(
        np.zeros(3),
        1.0,
        stage_times=np.array([0.0, 2.0]),
        stage_vals=np.array([2.0, 2.0]),
        detection_opts=DetectionOpts(
            double_watershed=False,
            refinement=False,
        ),
        fit_param_basis=False,
        plot=False,
        verbose=False,
        **kwargs,
    )


def test_run_dynamo_fits_splines_by_default_and_isolates_axis_failures(
    stubbed_pipeline,
):
    calls, power_result = stubbed_pipeline

    with pytest.warns(RuntimeWarning, match="SO-phase spline fit failed"):
        output = _run_stubbed_pipeline()

    assert [call["kind"] for call in calls] == ["power", "phase"]
    assert calls[0]["opts"] is None
    assert calls[1]["opts"] is None
    assert output.SOPHs.SOpower_splinefit is power_result
    assert output.SOPHs.SOphase_splinefit is None
    assert "fit_spline_basis" in output.timings


def test_run_dynamo_can_disable_spline_fits(stubbed_pipeline):
    calls, _ = stubbed_pipeline

    output = _run_stubbed_pipeline(fit_spline_basis=False)

    assert calls == []
    assert output.SOPHs.SOpower_splinefit is None
    assert output.SOPHs.SOphase_splinefit is None
    assert "fit_spline_basis" not in output.timings
