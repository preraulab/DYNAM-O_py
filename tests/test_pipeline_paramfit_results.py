"""Focused pipeline wiring for MATLAB-compatible parametric-fit outputs."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

import pydynamo.pipeline as pipeline
from pydynamo.defaults import DetectionOpts


def test_run_dynamo_passes_sophs_and_peaks_to_the_kernel_fit(monkeypatch):
    stats = pd.DataFrame({
        "PeakTime": [0.5, 1.5, 2.5],
        "PeakFrequency": [6.0, 8.0, 10.0],
        "Duration": [1.0, 1.0, 1.0],
        "Bandwidth": [2.0, 2.0, 2.0],
        "Height": [3.0, 3.0, 3.0],
        "Volume": [4.0, 4.0, 4.0],
        "Area": [2.0, 2.0, 2.0],
        "Peakiness": [1.0, 1.0, 1.0],
    })
    phase_model = np.array([[0.2, 0.8], [0.6, 0.4]])
    calls = []

    monkeypatch.setattr(
        pipeline, "detect_artifacts",
        lambda data, _fs: np.zeros(data.size, dtype=bool),
    )
    monkeypatch.setattr(
        pipeline, "mtm_spectrogram",
        lambda *_args, **_kwargs: (
            np.ones((2, 3)),
            np.array([0.0, 1.0, 2.0]),
            np.array([2.0, 4.0]),
        ),
    )
    monkeypatch.setattr(
        pipeline, "compute_baseline",
        lambda spect, *_args, **_kwargs: np.ones((spect.shape[0], 1)),
    )
    monkeypatch.setattr(
        pipeline, "subtract_baseline", lambda spect, _baseline: spect,
    )
    monkeypatch.setattr(
        pipeline, "extract_tfpeaks",
        lambda *_args, **_kwargs: (
            stats.copy(), np.zeros((2, 3), dtype=int)
        ),
    )
    monkeypatch.setattr(
        pipeline, "compute_so_power",
        lambda *_args, **_kwargs: (
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([2.0, 2.0, 2.0, 2.0]),
            "none",
            None,
        ),
    )
    monkeypatch.setattr(
        pipeline, "compute_so_phase",
        lambda *_args, **_kwargs: (
            np.array([0.0, 0.1, 0.2, 0.3]),
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([2.0, 2.0, 2.0, 2.0]),
            None,
        ),
    )

    def histogram(selection, feature_bins):
        return {
            "c_mat": np.ones((2, 2)),
            "c_cbins": np.asarray(feature_bins, dtype=float),
            "freq_cbins": np.array([6.0, 10.0]),
            "time_in_bin": np.ones((2, 1)),
            "peak_at_freq": np.ones(2),
            "peak_selection_inds": np.asarray(selection, dtype=bool),
        }

    monkeypatch.setattr(
        pipeline,
        "so_power_histogram",
        lambda *_args, **_kwargs: histogram(
            [True, False, True], [-1.0, 1.0]
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "so_phase_histogram",
        lambda *_args, **_kwargs: histogram(
            [True, False, True], [-np.pi, np.pi]
        ),
    )

    phase_result = SimpleNamespace(model_soph=phase_model)
    power_result = SimpleNamespace(model_soph=np.ones((2, 2)))

    def fake_fit(*args, **kwargs):
        calls.append((args, kwargs))
        return power_result, phase_result, {}

    monkeypatch.setattr(pipeline, "fit_paramfits", fake_fit)

    out = pipeline.run_dynamo(
        np.zeros(4),
        1.0,
        stage_times=np.array([0.0, 3.0]),
        stage_vals=np.array([2.0, 2.0]),
        detection_opts=DetectionOpts(
            double_watershed=False,
            refinement=False,
        ),
        plot=False,
        verbose=False,
    )

    # One kernel call fits both axes. It gets every TF-peak plus the mask
    # of the ones the SOPHs counted, so the kernel picks the Pk* members.
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert np.array_equal(args[3], [-np.pi, np.pi])
    assert kwargs["stats_table"]["PeakFrequency"].tolist() == [6.0, 8.0, 10.0]
    assert kwargs["peak_in_hist"].tolist() == [True, False, True]
    assert kwargs["baseline"] is not None
    assert out.SOPHs.SOphase_paramfit is phase_result
    assert out.SOPHs.SOpower_paramfit is power_result
