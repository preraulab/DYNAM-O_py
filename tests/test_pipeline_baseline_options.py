"""Focused tests for BaselineOpts handling in run_dynamo."""

import numpy as np
import pandas as pd
import pytest

import pydynamo.pipeline as pipeline
from pydynamo.defaults import BaselineOpts, DetectionOpts


@pytest.mark.parametrize(
    ("baseline_exclude", "expected"),
    [
        ([0, 0, 1, 0, 1, 0], [False, True, False, True]),
        ([0, 1, 0, 1], [False, True, False, True]),
        ([], [False, False, False, False]),
    ],
)
def test_crop_baseline_exclude_aligns_full_or_cropped_mask(
    baseline_exclude, expected,
):
    result = pipeline._crop_baseline_exclude(
        baseline_exclude, data_size=6, time_slice=slice(1, 5),
    )

    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    ("baseline_exclude", "match"),
    [
        ([0, 2, 0, 0], "binary"),
        ([False, True], "full data length"),
    ],
)
def test_crop_baseline_exclude_rejects_invalid_input(
    baseline_exclude, match,
):
    with pytest.raises(ValueError, match=match):
        pipeline._crop_baseline_exclude(
            baseline_exclude, data_size=6, time_slice=slice(1, 5),
        )


@pytest.mark.parametrize(
    ("baseline_trim", "expected"),
    [
        ((), (-np.inf, np.inf)),
        ((15.0, 180.0), (15.0, 180.0)),
        (0.5, (30.0, 150.0)),
        (2.0, (0.0, 240.0)),
    ],
)
def test_resolve_baseline_range_matches_matlab_trim_semantics(
    baseline_trim, expected,
):
    """Scalar trims are minute buffers; pairs are absolute seconds."""
    result = pipeline._resolve_baseline_range(
        baseline_trim,
        stage_times=np.array([0.0, 60.0, 120.0, 180.0, 240.0]),
        stage_vals=np.array([5.0, 2.0, 2.0, 5.0, 5.0]),
    )

    assert result == pytest.approx(expected)


def test_resolve_baseline_range_rejects_invalid_shape():
    with pytest.raises(ValueError, match="zero, one, or two"):
        pipeline._resolve_baseline_range(
            (1.0, 2.0, 3.0),
            stage_times=np.array([0.0, 30.0]),
            stage_vals=np.array([2.0, 2.0]),
        )


def test_scalar_baseline_trim_without_nonwake_uses_staging_span():
    result = pipeline._resolve_baseline_range(
        1.0,
        stage_times=np.array([0.0, 30.0]),
        stage_vals=np.array([5.0, 5.0]),
    )

    assert result == (0.0, 30.0)


def test_run_dynamo_defaults_to_first_and_last_valid_scored_stage(monkeypatch):
    captured = {}

    class StopAfterArtifacts(Exception):
        pass

    def capture_artifact_input(data, fs):
        captured["data"] = data.copy()
        captured["fs"] = fs
        raise StopAfterArtifacts

    monkeypatch.setattr(pipeline, "detect_artifacts", capture_artifact_input)

    with pytest.raises(StopAfterArtifacts):
        pipeline.run_dynamo(
            np.arange(8, dtype=float),
            1.0,
            stage_times=np.array([0.0, 2.0, 4.0, 6.0]),
            stage_vals=np.array([0.0, 2.0, 5.0, 6.0]),
            plot=False,
            verbose=False,
        )

    assert captured["fs"] == 1.0
    assert np.array_equal(captured["data"], [2.0, 3.0, 4.0])


def test_run_dynamo_default_range_requires_a_valid_scored_stage():
    with pytest.raises(ValueError, match="No valid stages"):
        pipeline.run_dynamo(
            np.arange(4, dtype=float),
            1.0,
            stage_times=np.array([0.0, 2.0]),
            stage_vals=np.array([0.0, 6.0]),
            plot=False,
            verbose=False,
        )


def test_run_dynamo_passes_custom_baseline_options_to_both_passes(
    monkeypatch,
):
    calls = []

    class StopAfterSecondBaseline(Exception):
        pass

    def fake_mtm_spectrogram(*_args, **_kwargs):
        return (
            np.ones((2, 3)),
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 2.0]),
        )

    def fake_compute_baseline(
        spect, _stimes, _t_data, baseline_exclude, *,
        baseline_range, baseline_ptile,
    ):
        calls.append({
            "baseline_exclude": baseline_exclude.copy(),
            "baseline_range": baseline_range,
            "baseline_ptile": baseline_ptile,
        })
        return np.ones((spect.shape[0], 1))

    def fake_extract_tfpeaks(*_args, return_labels=False, **_kwargs):
        assert return_labels
        return pd.DataFrame({"peak": [1]}), np.zeros((2, 3), dtype=int)

    def stop_after_second_baseline(*_args, **_kwargs):
        raise StopAfterSecondBaseline

    monkeypatch.setattr(
        pipeline, "detect_artifacts",
        lambda *_args, **_kwargs: np.array([False, False, True, False]),
    )
    monkeypatch.setattr(pipeline, "mtm_spectrogram", fake_mtm_spectrogram)
    monkeypatch.setattr(pipeline, "compute_baseline", fake_compute_baseline)
    monkeypatch.setattr(
        pipeline, "subtract_baseline", lambda spect, _baseline: spect,
    )
    monkeypatch.setattr(pipeline, "extract_tfpeaks", fake_extract_tfpeaks)
    monkeypatch.setattr(
        pipeline, "mask_spectrogram", stop_after_second_baseline,
    )

    with pytest.raises(StopAfterSecondBaseline):
        pipeline.run_dynamo(
            np.zeros(6),
            1.0,
            stage_times=np.array([0.0, 2.0, 4.0]),
            stage_vals=np.array([1.0, 5.0, 2.0]),
            time_range=(1.0, 4.0),
            detection_opts=DetectionOpts(
                double_watershed=True,
                reuse_baseline=False,
                refinement=False,
            ),
            baseline_opts=BaselineOpts(
                baseline_stages=(1, 2, 3, 4),
                baseline_exclude=(0, 0, 1, 0, 0, 0),
                baseline_ptile=7.5,
                baseline_trim=(1.5, 3.5),
            ),
            plot=False,
            verbose=False,
        )

    assert len(calls) == 2
    for call in calls:
        assert np.array_equal(
            call["baseline_exclude"], [False, True, True, False],
        )
        assert call["baseline_range"] == (1.5, 3.5)
        assert call["baseline_ptile"] == 7.5
