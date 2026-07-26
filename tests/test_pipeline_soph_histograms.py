"""Focused regressions for paired SO-power and SO-phase histograms."""

import numpy as np
import pandas as pd

import pydynamo.pipeline as pipeline
from pydynamo.defaults import DetectionOpts, SOPHOpts


def test_run_dynamo_synchronizes_power_invalidity_before_histograms(
    monkeypatch,
):
    captured = {}
    real_power_histogram = pipeline.so_power_histogram
    real_phase_histogram = pipeline.so_phase_histogram

    def capture_power_histogram(*args, **kwargs):
        result = real_power_histogram(*args, **kwargs)
        captured["power"] = result
        return result

    def capture_phase_histogram(*args, **kwargs):
        captured["phase_input"] = np.asarray(args[3]).copy()
        result = real_phase_histogram(*args, **kwargs)
        captured["phase"] = result
        return result

    def fake_spectrogram(*_args, **_kwargs):
        return (
            np.ones((2, 5)),
            np.arange(5, dtype=float),
            np.array([9.0, 10.0]),
        )

    def fake_extract(*_args, **kwargs):
        stats = pd.DataFrame({
            "PeakTime": [0.25, 1.5, 3.5],
            "PeakFrequency": [10.0, 10.0, 10.0],
        })
        if kwargs.get("return_labels"):
            return stats, np.ones((2, 5), dtype=int)
        return stats

    power = np.array([0.0, np.nan, np.nan, 0.0, 0.0])
    power_times = np.arange(5, dtype=float)
    phase_times = np.arange(0.0, 4.01, 0.25)
    phase = np.linspace(-1.0, 1.0, phase_times.size)

    monkeypatch.setattr(
        pipeline, "detect_artifacts",
        lambda data, _fs: np.zeros(data.size, dtype=bool),
    )
    monkeypatch.setattr(pipeline, "mtm_spectrogram", fake_spectrogram)
    monkeypatch.setattr(
        pipeline, "compute_baseline",
        lambda spect, *_args, **_kwargs: np.ones((spect.shape[0], 1)),
    )
    monkeypatch.setattr(
        pipeline, "subtract_baseline",
        lambda spect, _baseline: spect,
    )
    monkeypatch.setattr(pipeline, "extract_tfpeaks", fake_extract)
    monkeypatch.setattr(
        pipeline, "compute_so_power",
        lambda *_args, **_kwargs: (
            power, power_times, np.full(power_times.size, 2.0), "direct", 0.0,
        ),
    )
    monkeypatch.setattr(
        pipeline, "compute_so_phase",
        lambda *_args, **_kwargs: (
            phase, phase_times, np.full(phase_times.size, 2.0), None,
        ),
    )
    monkeypatch.setattr(
        pipeline, "so_power_histogram", capture_power_histogram,
    )
    monkeypatch.setattr(
        pipeline, "so_phase_histogram", capture_phase_histogram,
    )

    pipeline.run_dynamo(
        np.ones(5), 1.0,
        stage_times=np.array([0.0, 4.0]),
        stage_vals=np.array([2.0, 2.0]),
        time_range=(0.0, 4.0),
        detection_opts=DetectionOpts(
            double_watershed=False,
            refinement=False,
        ),
        soph_opts=SOPHOpts(
            SOpower_min_time_in_bin=0.0,
            compute_rate=False,
        ),
        fit_param_basis=False,
        plot=False,
        verbose=False,
    )

    assert np.isnan(captured["phase_input"]).any()
    assert np.array_equal(
        captured["power"]["peak_selection_inds"],
        captured["phase"]["peak_selection_inds"],
    )
    assert np.array_equal(
        captured["power"]["peak_at_freq"],
        captured["phase"]["peak_at_freq"],
    )
