"""Regressions for MATLAB DYNAM-O PRs 71, 73, 74, 75, and 80."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pydynamo.defaults import SOPHOpts
from pydynamo.plot import summary_plot
from pydynamo.soph.sopower import compute_so_power


def test_pipeline_default_sopower_norm_method_is_stage_aware():
    assert SOPHOpts().SOpower_norm_method == "p2shift1234"


def test_shift_method_controls_stages_and_preserves_real_stage_labels(
    monkeypatch,
):
    import pydynamo.soph.sopower as sopower_module

    db_values = np.array([0.0, 0.0, 10.0, 20.0])

    def fake_mts(*_args, **_kwargs):
        power = 10.0 ** (db_values / 10.0)
        return (
            np.vstack([power, np.zeros_like(power)]),
            np.arange(db_values.size, dtype=float),
            np.array([0.0, 1.0]),
        )

    monkeypatch.setattr(sopower_module, "_mts", fake_mts)
    kwargs = {
        "stage_times": np.array([0.0, 2.0, 4.0]),
        "stage_vals": np.array([5.0, 2.0, 2.0]),
        "eeg_times": np.arange(4, dtype=float),
        "time_range": (0.0, 3.0),
        "retain_Fs": False,
    }

    stage2, _, stages, _, stage2_ptile = compute_so_power(
        np.ones(4), 1.0, norm_method="p50shift2", **kwargs,
    )
    stage2_and_wake, _, stages_with_wake, _, stage2_and_wake_ptile = (
        compute_so_power(
            np.ones(4), 1.0, norm_method="p50shift25", **kwargs,
        )
    )
    bare_shift, _, _, _, bare_shift_ptile = compute_so_power(
        np.ones(4), 1.0, norm_method="shift", **kwargs,
    )
    explicit_default, _, _, _, explicit_default_ptile = compute_so_power(
        np.ones(4), 1.0, norm_method="p2shift1234", **kwargs,
    )

    assert np.array_equal(stages, [5.0, 5.0, 2.0, 2.0])
    assert np.array_equal(stages_with_wake, stages)
    assert stage2_ptile == pytest.approx(15.0)
    assert stage2_and_wake_ptile == pytest.approx(5.0)
    assert np.allclose(stage2_and_wake - stage2, 10.0)
    assert bare_shift_ptile == pytest.approx(explicit_default_ptile)
    assert np.allclose(bare_shift, explicit_default)


@pytest.mark.parametrize(
    ("volumes", "expected_sizes"),
    [
        ([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], [0.625, 2.5, 5.0, 10.0]),
        ([2.0, 3.0, 5.0, 7.0, 50.0, 100.0], [0.4, 1.0, 1.4, 10.0]),
    ],
    ids=("baseline_volumes", "skewed_volumes"),
)
def test_summary_plot_shows_all_nonartifact_peaks_and_shades_exclusions(
    volumes, expected_sizes,
):
    stats = pd.DataFrame({
        "PeakTime": [0.25, 1.0, 2.0, 4.0, 6.5, 8.0],
        "PeakFrequency": [6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        "Volume": volumes,
        "SOphase": [-np.pi / 2, np.nan, 0.0, np.pi / 2, np.pi / 4, np.nan],
    })
    stats_before = stats.copy(deep=True)
    hist_peakidx = np.array([False, False, True, False, True, False])
    stage_times = np.array([-1.5, 3.0, 6.0, 10.5])
    stage_vals = np.array([2.0, 5.0, 2.0, 2.0])
    sopower_times = np.arange(10, dtype=float)
    sopower = np.array([0.0, np.nan, 2.0, 3.0, np.nan,
                        5.0, 6.0, 7.0, np.nan, 9.0])
    sophs = SimpleNamespace(
        SOpower_norm=sopower,
        SOpower_times=sopower_times,
        SOpower_mat=np.ones((2, 2)),
        SOphase_mat=np.ones((2, 2)),
        SOpower_bins=np.array([-5.0, 25.0]),
        SOphase_bins=np.array([-np.pi, np.pi]),
        freq_bins=np.array([2.0, 25.0]),
    )

    fig = summary_plot(
        np.ones((2, 2)), np.array([-1.5, 10.5]), np.array([2.0, 25.0]),
        np.zeros(25, dtype=bool), np.linspace(-1.5, 10.5, 25),
        stage_times, stage_vals, stats, sophs,
        time_range=(-1.5, 10.5), freq_limits=(2.0, 25.0),
        mtm_freq_range=(2.0, 25.0), peak_size_prctiles=(50.0, 100.0),
        hist_peakidx=hist_peakidx, SOPH_stages=(2,),
    )
    try:
        scatter_ax = next(
            ax for ax in fig.axes
            if ax.get_title() == "Extracted Time-Frequency Peaks"
        )
        scatter = scatter_ax.collections[0]
        order = np.argsort(scatter.get_offsets()[:, 0])
        offsets = scatter.get_offsets()[order]
        sizes = scatter.get_sizes()[order]

        assert np.allclose(
            offsets[:, 0], np.array([0.25, 2.0, 4.0, 6.5]) / 3600.0,
        )
        assert np.allclose(offsets[:, 1], [6.0, 10.0, 12.0, 14.0])
        assert np.allclose(sizes, expected_sizes)
        assert np.max(sizes) == pytest.approx(10.0)
        assert np.allclose(
            np.asarray(scatter.get_array())[order],
            [-np.pi / 2, 0.0, np.pi / 2, np.pi / 4],
        )

        spans = sorted(
            (patch.get_x(), patch.get_x() + patch.get_width())
            for patch in scatter_ax.patches
        )
        expected_spans = np.array([
            [-1.5, -1.0],
            [0.0, 2.0],
            [3.0, 6.0],
            [7.0, 9.0],
            [10.0, 10.5],
        ]) / 3600.0
        assert np.allclose(spans, expected_spans)
        assert all(
            np.allclose(patch.get_facecolor()[:3], [0.85, 0.85, 0.85])
            and patch.get_alpha() == pytest.approx(0.6)
            and patch.get_zorder() > scatter.get_zorder()
            for patch in scatter_ax.patches
        )
        pd.testing.assert_frame_equal(stats, stats_before)
    finally:
        plt.close(fig)
