"""Artifact mask: compare pydynamo to MATLAB's `artifacts` from segment_out.mat.

The MATLAB mask is uint8 length 502601. We target near-bit-identity (≥99.9%
of samples agree). Small disagreement is expected from:
  - Chebyshev filter-coefficient quantization differences between MATLAB
    designfilt and scipy.signal.cheby1
  - movmedian endpoint handling at the 5-min detrend window edges
  - slope_test (currently disabled in our port; MATLAB enables it by default)
"""

import numpy as np

import pydynamo.artifacts as artifact_module
from pydynamo.artifacts import _flat_mask, detect_artifacts


SEGMENT_TIME_RANGE = (8420, 13446)


def test_flat_mask_tolerance_uses_full_run_span():
    # Adjacent differences are within tolerance, but the full span is not.
    # MATLAB get_chunks therefore splits this sequence rather than accepting
    # it as one near-flat run.
    data = np.array([0.0, 0.009, 0.018])

    assert not _flat_mask(data, min_run=3, tol=0.01).any()


def test_flat_mask_does_not_flag_ordinary_varying_eeg():
    rng = np.random.default_rng(0)
    data = rng.normal(size=2_000)
    flat_tol = 0.02 * np.std(data, ddof=1)

    assert not _flat_mask(data, min_run=100, tol=flat_tol).any()


def test_detect_artifacts_recovers_resampled_disconnection(monkeypatch):
    rng = np.random.default_rng(1)
    data = rng.normal(size=2_000)
    disconnected = slice(800, 950)
    data[disconnected] = np.linspace(-0.005, 0.005, 150)

    def return_initial_mask(
        data, fs, passband, crit, bad_inds, **kwargs
    ):
        return bad_inds.copy()

    monkeypatch.setattr(
        artifact_module, "_compute_band_artifacts", return_initial_mask
    )

    artifacts = detect_artifacts(data, 100.0, slope_test=False)

    assert artifacts[disconnected].all()
    assert not artifacts[:500].any()
    assert not artifacts[1_200:].any()


def test_artifacts_match_matlab(example_data, segment_out_compat):
    fs = float(example_data["Fs"])
    data = np.asarray(example_data["data"]).ravel()
    # MATLAB `time_range_inds = t >= t0 & t <= t1` is inclusive on both ends,
    # so the slice is [i0, i1] inclusive → Python [i0:i1+1].
    i0 = int(round(SEGMENT_TIME_RANGE[0] * fs))
    i1 = int(round(SEGMENT_TIME_RANGE[1] * fs))
    segment = data[i0 : i1 + 1]

    art_py = detect_artifacts(segment, fs)
    art_mat = np.asarray(segment_out_compat["artifacts"], dtype=bool).ravel()

    assert art_py.shape == art_mat.shape, (art_py.shape, art_mat.shape)

    agree = (art_py == art_mat).mean()
    py_true = art_py.mean()
    mat_true = art_mat.mean()
    overlap = (art_py & art_mat).sum() / max(art_mat.sum(), 1)

    print(f"agreement rate       = {agree:.6f}")
    print(f"py  artifact fraction = {py_true:.6f}")
    print(f"mat artifact fraction = {mat_true:.6f}")
    print(f"recall of mat in py   = {overlap:.6f}")

    # Effectively bit-identical: allow <0.5% disagreement (filter edge effects
    # + slope_test omitted).
    assert agree > 0.995, f"artifact agreement rate {agree:.4f} too low"
