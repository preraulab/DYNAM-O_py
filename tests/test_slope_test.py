"""Focused smoke coverage for slope-test artifact detection."""

import numpy as np

from pydynamo.artifacts import _slope_test
from pydynamo.io_compat import load_example_data


def test_slope_test_standalone_returns_correct_shape_and_type():
    """Smoke test: _slope_test runs without NaN-propagation errors on real data."""
    ed = load_example_data()
    fs = float(ed["Fs"])
    data_tr = ed["data"].ravel()[: int(600 * fs)].astype(np.float64)  # 10 min

    mask = _slope_test(data_tr, fs, slope_crit=-0.5)
    assert mask.dtype == np.bool_
    assert mask.shape == data_tr.shape
    assert 0.0 <= mask.mean() <= 0.5, \
        f"slope_test flagging {mask.mean()*100:.1f}% is outside sensible range"
