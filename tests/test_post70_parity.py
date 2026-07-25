"""Regressions for MATLAB DYNAM-O_dev PRs 71, 73, 74, and 75."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pydynamo.plot import summary_plot
from pydynamo.soph.paramfit import core
from pydynamo.soph.paramfit.basis import eval_modes, vm_gauss
from pydynamo.soph.paramfit.opts import ParamBasisOpts, resolve_bounds
from pydynamo.soph.sopower import compute_so_power


def test_vm_gauss_frequency_width_is_standard_deviation():
    phase = np.array([[0.0, 0.0]])
    freq = np.array([[10.0, 12.5]])

    values = vm_gauss(
        phase, freq, 2.0, 10.0, 2.5, 0.0, 1.0, 0.0,
    )

    assert values[0, 1] / values[0, 0] == pytest.approx(
        np.exp(-1.0), abs=1e-12,
    )


def test_phase_model_normalizes_background_and_modes_together():
    phase_bins = np.linspace(-np.pi, np.pi, 9)
    freq_bins = np.array([8.0, 10.0, 12.0])
    params = np.array([[0.08, 10.0, 1.5, 0.4, 1.0, 0.0]])
    background = np.array([0.01, 0.2, 0.1])

    model = eval_modes(
        params, phase_bins, freq_bins, kind="phase",
        background=background, unit_row=True,
    )

    assert np.allclose(model.sum(axis=1), 1.0, atol=1e-12)


def test_center_constraints_have_axis_specific_names_and_bounds():
    amp = np.array([0.05])
    feature = np.linspace(-np.pi, np.pi, 41)
    freq = np.linspace(2.0, 18.0, 65)

    power = ParamBasisOpts.power()
    phase = ParamBasisOpts.phase()
    assert power.constrain_freq_center is True
    assert power.constrain_power_center is True
    assert not hasattr(power, "constrain_phase_center")
    assert phase.constrain_freq_center is True
    assert phase.constrain_phase_center is True
    assert not hasattr(phase, "constrain_power_center")
    assert ParamBasisOpts(kind="phase").constrain_phase_center is True
    assert phase.LB_default[2] == 1.0
    assert phase.UB_default[2] == pytest.approx(np.sqrt(15.0))

    lb, ub = resolve_bounds(
        ParamBasisOpts.phase(constrain_freq_center=False),
        amp, feature, freq,
    )
    assert np.isneginf(lb[1]) and np.isposinf(ub[1])
    assert lb[3] == pytest.approx(-2 * np.pi)
    assert ub[3] == pytest.approx(2 * np.pi)

    lb, ub = resolve_bounds(
        ParamBasisOpts.phase(constrain_phase_center=False),
        amp, feature, freq,
    )
    assert lb[1] == pytest.approx(freq.min())
    assert ub[1] == pytest.approx(freq.max())
    assert np.isneginf(lb[3]) and np.isposinf(ub[3])

    with pytest.raises(TypeError):
        ParamBasisOpts.phase(constrain_power_center=False)
    with pytest.raises(ValueError):
        ParamBasisOpts.power(constrain_power_center=2)
    with pytest.raises(ValueError):
        ParamBasisOpts.phase(constrain_phase_center=np.array([True, False]))


def test_phase_min_amp_and_density_use_circular_empirical_amplitude(monkeypatch):
    phase_bins = np.linspace(-np.pi, np.pi, 5)
    freq_bins = np.array([9.0, 10.0, 11.0])
    raw_params = np.array([[0.05, 10.0, 1.0, 2 * np.pi, 0.5, 0.0]])
    background = np.array([0.02, 0.4, 0.01])
    soph = np.full((freq_bins.size, phase_bins.size), 0.2)

    def fake_fit_once(*_args, **_kwargs):
        return {
            "params": raw_params.copy(),
            "background": background.copy(),
            "sse": 0.0,
            "rsquare": 1.0,
            "adjrsquare": 1.0,
            "rmse": 0.0,
            "dfe": 1.0,
            "dfm": 9.0,
        }

    monkeypatch.setattr(core, "_fit_once", fake_fit_once)
    opts = ParamBasisOpts.phase(
        max_peaks=1, criterion="max", min_amp=0.2,
        feature_limits=(-np.pi, np.pi), freq_limits=(9.0, 11.0),
    )

    result = core.fit_param_basis_axis(
        soph, phase_bins, freq_bins, opts, seed_modes=raw_params,
    )

    no_sinusoid = background.copy()
    no_sinusoid[0] = 0.0
    model_no_sinusoid = eval_modes(
        raw_params, phase_bins, freq_bins, kind="phase",
        background=no_sinusoid, unit_row=True,
    )
    freq_idx = np.argmin(np.abs(freq_bins - raw_params[0, 1]))
    phase_delta = np.arctan2(
        np.sin(phase_bins - raw_params[0, 3]),
        np.cos(phase_bins - raw_params[0, 3]),
    )
    phase_idx = np.argmin(np.abs(phase_delta))
    expected_density = model_no_sinusoid[freq_idx, phase_idx]

    assert result.params.shape == (1, 6)
    assert result.params[0, 0] == pytest.approx(expected_density, abs=1e-12)
    assert result.params[0, 3] == pytest.approx(0.0, abs=1e-12)
    assert -np.pi <= result.params[0, 3] <= np.pi


def test_phase_background_only_uses_matlab_empty_mode_bounds(monkeypatch):
    captured = {}

    def fake_fit_vmgauss(
        _soph, _phase, _freq, _initial, _lower, _upper,
        bg_initial, bg_lower, bg_upper, _max_iters, _unit_row,
    ):
        captured["initial"] = bg_initial
        captured["lower"] = bg_lower
        captured["upper"] = bg_upper
        return {"background": bg_initial}

    monkeypatch.setattr(
        core, "_rs", SimpleNamespace(fit_vmgauss=fake_fit_vmgauss),
    )
    soph = np.arange(1.0, 7.0).reshape(2, 3)

    background = core._fit_background_only(
        soph, np.arange(3.0), np.arange(2.0), "phase",
    )

    assert np.allclose(background, [0.0, 0.0, 3.5])
    assert np.allclose(captured["initial"], [0.0, 0.0, 3.5])
    assert np.array_equal(captured["lower"], [0.0, 0.0, 0.0])
    assert np.array_equal(captured["upper"], [1.0, 1.0, 6.0])


def test_all_stage_shift_preserves_real_stage_labels(monkeypatch):
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
        "norm_method": "p50shift2",
        "retain_Fs": False,
    }

    all_stage, _, stages, _, all_ptile = compute_so_power(
        np.ones(4), 1.0, shift_uses_stages=False, **kwargs,
    )
    stage_only, _, stages_restricted, _, stage_ptile = compute_so_power(
        np.ones(4), 1.0, shift_uses_stages=True, **kwargs,
    )

    assert np.array_equal(stages, [5.0, 5.0, 2.0, 2.0])
    assert np.array_equal(stages_restricted, stages)
    assert all_ptile == pytest.approx(5.0)
    assert stage_ptile == pytest.approx(15.0)
    assert np.allclose(all_stage - stage_only, 10.0)


def test_summary_plot_shows_all_nonartifact_peaks_and_shades_exclusions():
    stats = pd.DataFrame({
        "PeakTime": [0.25, 1.0, 2.0, 4.0, 6.5, 8.0],
        "PeakFrequency": [6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        "Volume": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
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
        assert np.allclose(sizes, [0.05, 0.2, 0.4, 0.8])
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
