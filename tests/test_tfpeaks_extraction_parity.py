"""Focused contracts for Python/Rust TF-peak extraction parity."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pydynamo.tfpeaks import extract as extract_module


def _fake_fused_result(shape):
    labels = np.zeros(shape, dtype=np.int64)
    labels[:, : shape[1] // 2] = 1
    labels[:, shape[1] // 2 :] = 2
    return {
        "peak_time": np.array([1.0, 2.0]),
        "peak_freq": np.array([3.0, 4.0]),
        "duration": np.array([1.0, 6.0]),
        "bandwidth": np.array([1.0, 1.0]),
        "height": np.array([10.0, 10.0]),
        "volume": np.array([1.0, 1.0]),
        "segment_num": np.array([1.0, 1.0]),
        "area": np.array([1.0, 1.0]),
        "peakiness": np.array([1.0, 1.0]),
        "bbox": np.zeros((2, 4)),
        "labels": labels,
    }


@pytest.mark.parametrize(
    ("downsample", "expected_rust_strides"),
    [
        ((3, 5), (5, 3)),
        (None, (1, 1)),
    ],
)
def test_fused_passes_global_shift_and_rust_axis_order(
    monkeypatch, downsample, expected_rust_strides,
):
    calls = []

    def fake_extract_tfpeaks(*args):
        calls.append(args)
        return _fake_fused_result(args[0].shape)

    monkeypatch.setattr(
        extract_module, "_dynamo_rs",
        SimpleNamespace(extract_tfpeaks=fake_extract_tfpeaks),
    )
    spect = np.arange(48, dtype=float).reshape(6, 8)
    spect[4, 6] = -7.0
    stimes = np.arange(8, dtype=float)
    sfreqs = np.arange(6, dtype=float)

    extract_module.extract_tfpeaks_fused(
        spect, stimes, sfreqs,
        downsample=downsample,
        prom_min=float("-inf"),
    )

    args = calls[0]
    assert args[5:7] == expected_rust_strides
    assert args[10] == -7.0


def test_fused_return_labels_match_rows_unless_raw_requested(monkeypatch):
    def fake_extract_tfpeaks(*args):
        return _fake_fused_result(args[0].shape)

    monkeypatch.setattr(
        extract_module, "_dynamo_rs",
        SimpleNamespace(extract_tfpeaks=fake_extract_tfpeaks),
    )
    spect = np.ones((6, 8))
    stimes = np.arange(8, dtype=float)
    sfreqs = np.arange(6, dtype=float)
    kwargs = dict(dur_max=5.0, prom_min=float("-inf"))

    filtered_table, filtered_labels = extract_module.extract_tfpeaks_fused(
        spect, stimes, sfreqs, return_labels=True, **kwargs,
    )
    raw_table, raw_labels = extract_module.extract_tfpeaks_fused(
        spect, stimes, sfreqs,
        return_labels=True, return_raw_labels=True, **kwargs,
    )

    assert filtered_table["label"].tolist() == [1]
    assert set(np.unique(filtered_labels)) == {0, 1}
    pd.testing.assert_frame_equal(raw_table, filtered_table)
    assert set(np.unique(raw_labels)) == {1, 2}


@pytest.mark.parametrize(
    ("downsample", "expected_shape"),
    [
        ((4, 3), (4, 5)),
        (None, (12, 20)),
    ],
)
def test_pure_segment_uses_documented_time_frequency_order(
    monkeypatch, downsample, expected_shape,
):
    seen = {}

    class WatershedReached(RuntimeError):
        pass

    def capture_watershed(img, **_kwargs):
        seen["shape"] = img.shape
        raise WatershedReached

    monkeypatch.setattr(extract_module, "watershed", capture_watershed)
    spect = np.ones((12, 20))
    stimes = np.arange(20, dtype=float)
    sfreqs = np.arange(12, dtype=float)

    with pytest.raises(WatershedReached):
        extract_module.extract_tfpeaks_segment(
            spect, stimes, sfreqs, downsample=downsample,
        )

    assert seen["shape"] == expected_shape


def test_pure_wrapper_shares_global_shift_and_separates_label_modes(monkeypatch):
    shifts = []

    def fake_segment(spect, _stimes, _sfreqs, *, segment_num,
                     trim_shift_val, **_kwargs):
        shifts.append(trim_shift_val)
        labels = np.ones(spect.shape, dtype=np.int64)
        labels[:, spect.shape[1] // 2 :] = 2
        table = pd.DataFrame({"label": [1], "SegmentNum": [segment_num]})
        return table, labels

    monkeypatch.setattr(
        extract_module, "extract_tfpeaks_segment", fake_segment,
    )
    spect = np.arange(32, dtype=float).reshape(4, 8)
    spect[3, 7] = -9.0
    stimes = np.arange(8, dtype=float)
    sfreqs = np.arange(4, dtype=float)

    table, labels = extract_module.extract_tfpeaks(
        spect, stimes, sfreqs,
        seg_time=4.0, n_jobs=1, return_labels=True, use_fused=False,
    )
    raw_table, raw_labels = extract_module.extract_tfpeaks(
        spect, stimes, sfreqs,
        seg_time=4.0, n_jobs=1, return_labels=True,
        return_raw_labels=True, use_fused=False,
    )

    assert shifts == [-9.0, -9.0, -9.0, -9.0]
    assert set(np.unique(labels)) - {0} == set(table["label"])
    pd.testing.assert_frame_equal(raw_table, table)
    assert set(np.unique(raw_labels)) == {1, 2, 3, 4}


@pytest.mark.skipif(
    not extract_module._HAS_FUSED,
    reason="dynamo_rs.extract_tfpeaks is not installed",
)
@pytest.mark.parametrize(
    ("downsample", "rust_strides"),
    [
        ((3, 2), (2, 3)),
        (None, (1, 1)),
    ],
)
def test_real_fused_kernel_matches_explicit_matlab_parity_arguments(
    downsample, rust_strides,
):
    rng = np.random.default_rng(17)
    spect = rng.normal(size=(16, 60))
    spect[4:8, 10:18] += 5.0
    spect[9:13, 40:49] += 7.0
    stimes = np.arange(spect.shape[1], dtype=float) * 0.1
    sfreqs = np.arange(spect.shape[0], dtype=float) * 0.5
    trim_shift = float(np.min(spect))

    table, labels = extract_module.extract_tfpeaks_fused(
        spect, stimes, sfreqs,
        seg_time=3.0,
        return_labels=True,
        return_raw_labels=True,
        downsample=downsample,
        merge_thresh=0.0,
        trim_vol=0.8,
        dur_min=0.0,
        dur_max=float("inf"),
        bw_min=0.0,
        bw_max=float("inf"),
        prom_min=float("-inf"),
    )
    direct = extract_module._dynamo_rs.extract_tfpeaks(
        np.ascontiguousarray(spect),
        np.ascontiguousarray(stimes),
        np.ascontiguousarray(sfreqs),
        None,
        3.0,
        rust_strides[0],
        rust_strides[1],
        0.0,
        float("inf"),
        0.8,
        trim_shift,
        0.0,
        float("inf"),
        0.0,
        float("inf"),
        float("-inf"),
        float("inf"),
        float("-inf"),
        0,
        False,
        False,
    )

    np.testing.assert_array_equal(labels, np.asarray(direct["labels"]))
    np.testing.assert_allclose(
        table["PeakTime"].to_numpy(), np.asarray(direct["peak_time"]),
    )
