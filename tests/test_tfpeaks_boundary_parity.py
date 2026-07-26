"""Boundary regressions shared by MATLAB, Rust, and Python TF peaks."""

import numpy as np
import pandas as pd
import pytest

import pydynamo.tfpeaks.extract as extract_module
import pydynamo.tfpeaks.refine as refine_module
import pydynamo.tfpeaks.trim as trim_module


def test_pure_extraction_filters_on_inclusive_pixel_span(monkeypatch):
    # Two pixels span 1.0 s and 2.0 Hz. The legacy N-1 calculation made
    # those spans equal the configured minima and incorrectly rejected them.
    labels = np.zeros((4, 4), dtype=np.int64)
    labels[1:3, 1:3] = 1

    monkeypatch.setattr(extract_module, "_HAS_RUST_WS", False)
    monkeypatch.setattr(
        extract_module, "watershed",
        lambda *_args, **_kwargs: labels.copy(),
    )
    monkeypatch.setattr(
        extract_module, "merge_segment",
        lambda ws_labels, *_args, **_kwargs: ws_labels,
    )
    monkeypatch.setattr(
        extract_module, "expand_labels",
        lambda ws_labels, **_kwargs: ws_labels,
    )
    monkeypatch.setattr(
        trim_module, "trim_regions",
        lambda ws_labels, *_args, **_kwargs: (ws_labels.copy(), {}),
    )

    peaks = extract_module.extract_tfpeaks_segment(
        np.arange(1.0, 17.0).reshape(4, 4),
        stimes=np.arange(4, dtype=float) * 0.5,
        sfreqs=np.arange(4, dtype=float),
        downsample=None,
        dur_min=0.5,
        bw_min=1.0,
        dur_max=float("inf"),
        bw_max=float("inf"),
        prom_min=float("-inf"),
    )

    assert len(peaks) == 1
    assert peaks.loc[0, "Duration"] == pytest.approx(1.0)
    assert peaks.loc[0, "Bandwidth"] == pytest.approx(2.0)


def test_refinement_retains_time_edge_peaks_at_original_frequency(
    monkeypatch,
):
    stats = pd.DataFrame({
        "PeakTime": [1.0, 5.0, 9.0],
        "PeakFrequency": [5.0, 6.0, 8.0],
        "BoundingBox": [
            (0.5, 4.0, 1.0, 4.0),
            (4.5, 4.0, 1.0, 4.0),
            (8.5, 4.0, 1.0, 4.0),
        ],
    })
    seen_event_times = []

    def fake_hann_event_spectra(
        _data, _fs, event_times, _t, **_kwargs,
    ):
        seen_event_times.extend(event_times)
        return (
            np.array([[0.0], [1.0], [2.0], [5.0], [1.0]]),
            np.array([4.0, 5.0, 6.0, 7.0, 8.0]),
        )

    monkeypatch.setattr(
        refine_module, "_hann_event_spectra", fake_hann_event_spectra,
    )

    refined = refine_module.refine_peak_frequency(
        stats,
        np.zeros(11),
        1.0,
        t=np.arange(11, dtype=float),
        window_size=4.0,
        refine_method="spect_max",
    )

    assert seen_event_times == [5.0]
    assert refined["PeakTime"].tolist() == [1.0, 5.0, 9.0]
    assert refined["PeakFrequency"].tolist() == [5.0, 7.0, 8.0]
