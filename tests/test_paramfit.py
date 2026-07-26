"""Parametric-basis fitting: model recovery and loop behaviour.

Mirrors the intent of DYNAM-O_dev/tests/test_paramfit_model_recovery.m — build
a SOPH from known modes, fit it, and check the recovered parameters. No MATLAB
fixture needed, so these run anywhere dynamo_rs is built.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pydynamo.soph.paramfit import (
    ParamBasisOpts, eval_modes, extract_hist_peaks, fit_param_basis,
    fit_param_basis_axis, mode_overlap, residual_max_seed, seeds_from_stats,
    select_iteration,
)
from pydynamo.soph.paramfit.matlab_compat import prctile
from pydynamo.soph.paramfit.output import create_params_table

dynamo_rs = pytest.importorskip("dynamo_rs")

POWER_BINS = np.linspace(-5.0, 25.0, 61)
FREQ_BINS = np.linspace(2.0, 18.0, 81)


def _make_power_soph(modes, background=(0.004, 0.002, 0.05)):
    return eval_modes(np.asarray(modes, float), POWER_BINS, FREQ_BINS,
                      kind="power", background=np.asarray(background, float))


def test_recovers_two_power_modes():
    """A clean 2-mode surface is recovered to tight tolerance."""
    truth = np.array([
        [3.0, 13.5, 1.2, 10.0, 6.0, 0.01],
        [1.5, 6.0, 1.8, 2.0, 5.0, -0.02],
    ])
    soph = _make_power_soph(truth)

    # Seed slightly off truth, as the watershed would.
    seeds = truth * np.array([0.8, 1.04, 1.3, 1.2, 1.2, 0.0])
    opts = ParamBasisOpts.power(max_peaks=3, criterion="max", min_amp=0.0)
    res = fit_param_basis_axis(soph, POWER_BINS, FREQ_BINS, opts,
                               seed_modes=seeds)

    assert res.params.shape[0] == 2, "should keep exactly the two real modes"
    got = res.params[np.argsort(res.params[:, 1])]
    want = truth[np.argsort(truth[:, 1])]
    assert np.allclose(got[:, 1], want[:, 1], atol=0.05), "frequency centers"
    assert np.allclose(got[:, 3], want[:, 3], atol=0.5), "SO-power centers"
    assert res.gof["rsquare"] > 0.999
    assert res.model_soph.shape == soph.shape


def test_model_soph_spans_full_grid_not_just_window():
    """MATLAB evaluates the model over the whole grid even though it fits only
    the analysis window; the shapes must not silently shrink."""
    truth = np.array([[2.0, 12.0, 1.5, 8.0, 5.0, 0.0]])
    soph = _make_power_soph(truth)
    opts = ParamBasisOpts.power(
        max_peaks=1, feature_limits=(0.0, 20.0), freq_limits=(4.0, 16.0),
    )
    res = fit_param_basis_axis(soph, POWER_BINS, FREQ_BINS, opts,
                               seed_modes=truth)
    assert res.model_soph.shape == (FREQ_BINS.size, POWER_BINS.size)


def test_overlap_revert_rejects_duplicate_mode():
    """Two seeds on the same ridge must not both survive: the overlap check
    reverts the second one."""
    truth = np.array([[3.0, 13.5, 1.2, 10.0, 6.0, 0.0]])
    soph = _make_power_soph(truth)
    dup = np.vstack([truth, truth * np.array([1.0, 1.01, 1.0, 1.0, 1.0, 1.0])])
    opts = ParamBasisOpts.power(max_peaks=2, criterion="max",
                                max_overlap=0.25, min_freq_diff=0.0)
    res = fit_param_basis_axis(soph, POWER_BINS, FREQ_BINS, opts,
                               seed_modes=dup)
    assert res.params.shape[0] == 1, "duplicate mode should have been reverted"


def test_min_freq_diff_reverts_close_modes():
    truth = np.array([[3.0, 13.5, 1.2, 10.0, 6.0, 0.0]])
    soph = _make_power_soph(truth)
    close = np.vstack([truth, [2.5, 13.6, 1.2, 10.0, 6.0, 0.0]])
    opts = ParamBasisOpts.power(max_peaks=2, criterion="max",
                                min_freq_diff=0.5, max_overlap=1.0)
    res = fit_param_basis_axis(soph, POWER_BINS, FREQ_BINS, opts,
                               seed_modes=close)
    assert res.params.shape[0] == 1


def test_residual_seeding_finds_second_mode():
    """With only one watershed seed, matching pursuit should discover the
    second mode from the residual."""
    truth = np.array([
        [3.0, 13.5, 1.2, 10.0, 6.0, 0.0],
        [2.0, 6.0, 1.5, 5.0, 5.0, 0.0],
    ])
    soph = _make_power_soph(truth)
    opts = ParamBasisOpts.power(max_peaks=3, criterion="max", min_amp=0.0)
    res = fit_param_basis_axis(soph, POWER_BINS, FREQ_BINS, opts,
                               seed_modes=truth[:1])
    assert res.params.shape[0] >= 2, "residual seeding should add a mode"
    freqs = np.sort(res.params[:, 1])
    assert np.any(np.abs(freqs - 6.0) < 0.5), "second mode near 6 Hz"


def test_residual_max_seed_respects_freq_mask():
    y = np.linspace(2.0, 18.0, 81)
    x = np.linspace(-5.0, 25.0, 61)
    resid = np.zeros((y.size, x.size))
    resid[np.argmin(np.abs(y - 13.0)), 30] = 5.0    # masked out
    resid[np.argmin(np.abs(y - 6.0)), 10] = 1.0     # should win
    accepted = np.array([[1.0, 13.0, 1.0, 0.0, 5.0, 0.0]])

    row, found = residual_max_seed(resid, np.zeros_like(resid), x, y,
                                   accepted, min_freq_diff=2.0)
    assert found and abs(row[1] - 6.0) < 0.3

    # min_freq_diff=0 disables the mask, so the bigger peak wins again.
    row0, found0 = residual_max_seed(resid, np.zeros_like(resid), x, y,
                                     accepted, min_freq_diff=0.0)
    assert found0 and abs(row0[1] - 13.0) < 0.3


def test_residual_max_seed_reports_not_found_when_model_covers_data():
    y, x = np.linspace(2, 18, 21), np.linspace(-5, 25, 21)
    soph = np.ones((21, 21))
    row, found = residual_max_seed(soph, soph * 2, x, y, None, 0.0)
    assert not found and row is None


def test_mode_overlap_is_upper_triangular_iou():
    modes = np.array([
        [1.0, 8.0, 1.0, 5.0, 5.0, 0.0],
        [1.0, 8.0, 1.0, 5.0, 5.0, 0.0],
    ])
    ol = mode_overlap(modes, POWER_BINS, FREQ_BINS)
    assert ol.shape == (2, 2)
    assert ol[0, 1] == pytest.approx(1.0, abs=1e-9), "identical modes -> IoU 1"
    assert ol[1, 0] == 0.0 and ol[0, 0] == 0.0, "strictly upper triangular"


@pytest.mark.parametrize("criterion,expected", [
    ("max", 3),          # highest r2
    ("minpctr2", 2),     # last iteration whose % gain cleared the threshold
    ("mindr2", 2),
])
def test_select_iteration_criteria(criterion, expected):
    nums = [1, 2, 3]
    r2 = [0.90, 0.95, 0.9501]   # big jump to 2, negligible to 3
    got = select_iteration(criterion, nums, r2, min_dr2=0.01, min_pctr2=0.01)
    assert got == expected


def test_watershed_seeding_finds_both_modes():
    """The seeding watershed must keep both modes separate.

    This is the case that exposed the merge-threshold strictness issue: the
    dynamic threshold equals an actual edge weight, so a `>=` comparison
    merges the two basins into one and only one seed survives.
    """
    truth = np.array([
        [3.0, 13.5, 1.2, 10.0, 6.0, 0.0],
        [1.5, 6.0, 1.8, 2.0, 5.0, 0.0],
    ])
    soph = _make_power_soph(truth, background=(0.0, 0.0, 0.05))
    opts = ParamBasisOpts.power()
    merge_thresh, dur_min, bw_min, height_min, trim_vol = opts.watershed_params

    stats, labels = extract_hist_peaks(soph, POWER_BINS, FREQ_BINS,
                                       merge_thresh, dur_min, bw_min,
                                       height_min, trim_vol)
    assert len(stats) == 2, f"expected 2 seed regions, got {len(stats)}"

    seeds = seeds_from_stats(stats, opts.freq_limits, opts.min_freq_diff)
    assert seeds.shape == (2, 6)
    # Seeds are sorted by Height descending, so the tall 13.5 Hz mode is first.
    assert seeds[0, 1] == pytest.approx(13.5, abs=0.2)
    assert seeds[1, 1] == pytest.approx(6.0, abs=0.2)
    assert labels.max() >= 2


def test_seeds_from_stats_dedups_by_min_freq_diff():
    import pandas as pd
    stats = pd.DataFrame({
        "PeakFrequency": [13.5, 13.7, 6.0],
        "SOFeature": [10.0, 10.0, 2.0],
        "Height": [3.0, 2.9, 1.5],
        "Duration": [10.0, 10.0, 8.0],
        "Bandwidth": [2.0, 2.0, 3.0],
    })
    seeds = seeds_from_stats(stats, (2.0, 18.0), min_freq_diff=0.5)
    assert seeds.shape[0] == 2, "13.7 Hz is within 0.5 Hz of the taller 13.5"
    assert sorted(np.round(seeds[:, 1], 1)) == [6.0, 13.5]

    kept = seeds_from_stats(stats, (2.0, 18.0), min_freq_diff=0.0)
    assert kept.shape[0] == 3, "min_freq_diff=0 disables dedup"


def test_seeds_from_stats_respects_freq_limits():
    import pandas as pd
    stats = pd.DataFrame({
        "PeakFrequency": [1.0, 13.5, 25.0],
        "SOFeature": [10.0, 10.0, 10.0],
        "Height": [3.0, 3.0, 3.0],
        "Duration": [10.0, 10.0, 10.0],
        "Bandwidth": [2.0, 2.0, 2.0],
    })
    seeds = seeds_from_stats(stats, (2.0, 18.0), min_freq_diff=0.0)
    assert seeds.shape[0] == 1 and seeds[0, 1] == pytest.approx(13.5)


def test_fit_param_basis_end_to_end():
    """Driver: watershed seeding + fitting, recovering known modes."""
    truth = np.array([
        [3.0, 13.5, 1.2, 10.0, 6.0, 0.0],
        [1.5, 6.0, 1.8, 2.0, 5.0, 0.0],
    ])
    soph = _make_power_soph(truth, background=(0.0, 0.0, 0.05))
    opts = ParamBasisOpts.power(criterion="max", min_amp=0.0)
    res = fit_param_basis(soph, POWER_BINS, FREQ_BINS, opts)

    assert res.params.shape[0] == 2
    got = res.params[np.argsort(res.params[:, 1])]
    want = truth[np.argsort(truth[:, 1])]
    assert np.allclose(got[:, 1], want[:, 1], atol=0.05), "frequency centers"
    assert np.allclose(got[:, 3], want[:, 3], atol=0.5), "SO-power centers"
    assert res.gof["rsquare"] > 0.999
    assert res.wshed_img is not None


def test_accepts_either_soph_orientation():
    """The pipeline stores SOPH as (n_feature, n_freq) but the fitter works in
    (n_freq, n_feature); MATLAB sniffs the shape, so we must too."""
    from pydynamo.soph.paramfit.core import orient_soph

    truth = np.array([[3.0, 13.5, 1.2, 10.0, 6.0, 0.0]])
    soph = _make_power_soph(truth)                 # (n_freq, n_feature)
    assert soph.shape == (FREQ_BINS.size, POWER_BINS.size)

    assert orient_soph(soph, POWER_BINS, FREQ_BINS).shape == soph.shape
    assert orient_soph(soph.T, POWER_BINS, FREQ_BINS).shape == soph.shape

    opts = ParamBasisOpts.power(max_peaks=1, criterion="max")
    a = fit_param_basis_axis(soph, POWER_BINS, FREQ_BINS, opts, seed_modes=truth)
    b = fit_param_basis_axis(np.ascontiguousarray(soph.T), POWER_BINS,
                             FREQ_BINS, opts, seed_modes=truth)
    assert np.allclose(a.params, b.params), "both orientations must agree"

    with pytest.raises(ValueError):
        orient_soph(np.zeros((5, 7)), POWER_BINS, FREQ_BINS)


def test_matlab_prctile_convention():
    """MATLAB places sorted values at 100*(k-0.5)/n and clamps outside."""
    v = np.arange(1.0, 11.0)          # 1..10
    # Midpoints are at 5,15,...,95 percent, so the 5th pct is exactly v[0].
    assert prctile(v, 5) == pytest.approx(1.0)
    assert prctile(v, 95) == pytest.approx(10.0)
    assert prctile(v, 50) == pytest.approx(5.5)
    # Below/above the first/last plotting position it clamps rather than
    # extrapolating, which is where numpy's default disagrees.
    assert prctile(v, 0) == pytest.approx(1.0)
    assert prctile(v, 100) == pytest.approx(10.0)
    assert prctile(np.array([np.nan, 2.0, 4.0]), 50) == pytest.approx(3.0)
    assert np.isnan(prctile(np.array([np.nan]), 50))


def test_phase_axis_runs_and_wraps():
    """The phase axis uses the vmGauss kernel and a circular x axis."""
    phase_bins = np.linspace(-np.pi, np.pi, 51)
    truth = np.array([[2.0, 13.0, 2.0, 0.6, 1.0, 0.05]])
    soph = eval_modes(truth, phase_bins, FREQ_BINS, kind="phase",
                      background=np.array([0.01, 0.2, 0.1]),
                      unit_row=True)
    opts = ParamBasisOpts.phase(max_peaks=2, criterion="max", verbose=False)
    res = fit_param_basis_axis(soph, phase_bins, FREQ_BINS, opts,
                               seed_modes=truth)
    assert res.params.shape[0] >= 1
    assert res.model_soph.shape == soph.shape
    assert np.isfinite(res.params).all()
    assert res.params[0, 2] == pytest.approx(truth[0, 2], rel=0.12)
    assert np.all((-np.pi <= res.params[:, 3]) & (res.params[:, 3] <= np.pi))


def test_phase_seeded_iterations_restart_and_skip_rejected_seed(monkeypatch):
    """Seeded phase fits restart from MATLAB's original watershed guesses."""
    from pydynamo.soph.paramfit import core

    phase_bins = np.linspace(-np.pi, np.pi, 31)
    freq_bins = np.linspace(2.0, 18.0, 33)
    seeds = np.array([
        [0.05, 6.0, 1.2, -1.0, 0.8, 0.0],
        [0.04, 10.0, 1.3, 0.0, 0.9, 0.0],
        [0.03, 14.0, 1.4, 1.0, 1.0, 0.0],
    ])
    soph = eval_modes(
        seeds, phase_bins, freq_bins, kind="phase",
        background=np.array([0.0, 0.0, 0.01]), unit_row=True,
    )
    initial_guesses = []
    adjusted_r2 = iter((0.5, 0.5, 0.7))

    def fake_fit(soph_win, x_win, y_win, B0, LB, UB, opts):
        assert LB.shape == B0.shape == UB.shape
        initial_guesses.append(B0.copy())
        params = B0.copy()
        params[:, 1] += 0.75
        params[:, 3] += 0.25
        r2 = next(adjusted_r2)
        return {
            "params": params,
            "background": np.zeros(3),
            "sse": 1.0 - r2,
            "rsquare": r2,
            "adjrsquare": r2,
            "rmse": 0.1,
            "dfe": soph_win.size - params.size - 3,
            "dfm": params.size + 3,
        }

    monkeypatch.setattr(core, "_fit_once", fake_fit)
    opts = ParamBasisOpts.phase(
        max_peaks=3, criterion="mindr2", min_dr2=0.01,
        min_amp=0.0, max_overlap=1.0,
    )

    fit_param_basis_axis(
        soph, phase_bins, freq_bins, opts, seed_modes=seeds,
    )

    np.testing.assert_array_equal(initial_guesses[0], seeds[[0]])
    np.testing.assert_array_equal(initial_guesses[1], seeds[[0, 1]])
    np.testing.assert_array_equal(initial_guesses[2], seeds[[0, 2]])


def test_phase_background_only_fit_keeps_unit_row_normalization():
    phase_bins = np.linspace(-np.pi, np.pi, 31)
    freq_bins = np.linspace(8.0, 12.0, 17)
    background = np.array([0.01, 0.2, 0.1])
    soph = eval_modes(
        np.empty((0, 6)), phase_bins, freq_bins, kind="phase",
        background=background, unit_row=True,
    )
    seed = np.array([[0.01, 10.0, 1.5, 0.0, 1.0, 0.0]])
    opts = ParamBasisOpts.phase(
        max_peaks=1, criterion="max", min_amp=1.1,
        freq_limits=(8.0, 12.0),
    )

    res = fit_param_basis_axis(
        soph, phase_bins, freq_bins, opts, seed_modes=seed,
    )

    assert res.params.shape == (0, 6)
    assert np.allclose(res.model_soph.sum(axis=1), 1.0, atol=1e-12)


def test_power_background_fallback_returns_matching_gof():
    truth = np.array([[2.0, 10.0, 1.0, 5.0, 4.0, 0.0]])
    soph = _make_power_soph(truth, background=(0.0, 0.0, 0.2))
    opts = ParamBasisOpts.power(
        max_peaks=1, criterion="max", min_amp=np.inf, min_freq_diff=0.0,
    )

    res = fit_param_basis_axis(
        soph, POWER_BINS, FREQ_BINS, opts, seed_modes=truth,
    )

    valid_power = (
        (POWER_BINS >= opts.feature_limits[0])
        & (POWER_BINS <= opts.feature_limits[1])
    )
    valid_freq = (
        (FREQ_BINS >= opts.freq_limits[0])
        & (FREQ_BINS <= opts.freq_limits[1])
    )
    selected_sse = np.sum(
        (soph[np.ix_(valid_freq, valid_power)]
         - res.model_soph[np.ix_(valid_freq, valid_power)]) ** 2
    )

    assert res.params.shape == (0, 6)
    assert res.gof["sse"] == pytest.approx(selected_sse, abs=1e-10)


def test_phase_background_fallback_returns_matching_gof():
    phase_bins = np.linspace(-np.pi, np.pi, 31)
    freq_bins = np.linspace(2.0, 18.0, 33)
    truth = np.array([[0.05, 10.0, 1.5, 0.0, 1.0, 0.0]])
    soph = eval_modes(
        truth, phase_bins, freq_bins, kind="phase",
        background=np.array([0.0, 0.0, 0.001]), unit_row=True,
    )
    opts = ParamBasisOpts.phase(
        max_peaks=1, criterion="max", min_amp=np.inf,
    )

    res = fit_param_basis_axis(
        soph, phase_bins, freq_bins, opts, seed_modes=truth,
    )

    valid_phase = (
        (phase_bins >= opts.feature_limits[0])
        & (phase_bins <= opts.feature_limits[1])
    )
    valid_freq = (
        (freq_bins >= opts.freq_limits[0])
        & (freq_bins <= opts.freq_limits[1])
    )
    selected_sse = np.sum(
        (soph[np.ix_(valid_freq, valid_phase)]
         - res.model_soph[np.ix_(valid_freq, valid_phase)]) ** 2
    )

    assert res.params.shape == (0, 6)
    assert res.gof["sse"] == pytest.approx(selected_sse, abs=1e-10)


@pytest.mark.parametrize("center", [np.pi - 0.1, -np.pi + 0.1])
def test_phase_watershed_is_periodic_across_seam(monkeypatch, center):
    from pydynamo.soph.paramfit import core

    phase_bins = np.linspace(-np.pi, np.pi, 81)
    freq_bins = np.linspace(2.0, 18.0, 65)
    truth = np.array([[0.08, 10.0, 1.4, center, 0.7, 0.0]])
    soph = eval_modes(
        truth, phase_bins, freq_bins, kind="phase",
        background=np.array([0.0, 0.0, 0.01]),
    )
    captured = {}
    captured_result = SimpleNamespace(
        params_table=create_params_table(np.empty((0, 6)), "phase")
    )

    def capture_fit(
        soph_arg, x_arg, y_arg, opts_arg, seed_modes=None, wshed_img=None,
    ):
        captured.update(
            soph=soph_arg, x=x_arg, y=y_arg, opts=opts_arg,
            seed_modes=seed_modes, wshed_img=wshed_img,
        )
        return captured_result

    monkeypatch.setattr(core, "fit_param_basis_axis", capture_fit)
    opts = ParamBasisOpts.phase(
        gauss_filt_std=(1.5, 1.5),
        watershed_params=(np.nan, np.pi / 12, 0.5, 0.0, 0.7),
    )

    result = core.fit_param_basis(soph, phase_bins, freq_bins, opts)

    assert result is captured_result
    assert np.array_equal(captured["soph"], soph)
    assert captured["wshed_img"].shape == (
        freq_bins.size, 3 * phase_bins.size - 2,
    )
    assert captured["seed_modes"].shape == (1, 6)
    phase_error = np.arctan2(
        np.sin(captured["seed_modes"][0, 3] - center),
        np.cos(captured["seed_modes"][0, 3] - center),
    )
    assert abs(phase_error) < 0.05


def test_fused_and_assembled_extraction_agree_on_seeds():
    """The fused kernel is the seeding path of record.

    extracthistpeaks.m runs the same chain the fused kernel implements, and on
    a real SOPH the fused path recovers the seed regions the Python assembly
    misses (3 vs 1 on the validation night, matching MATLAB's "3 watershed
    modes found"). Guard that it finds at least as many.
    """
    from pydynamo.soph.paramfit.histpeaks import extract_hist_peaks

    truth = np.array([
        [3.0, 13.5, 1.2, 10.0, 6.0, 0.0],
        [1.5, 6.0, 1.8, 2.0, 5.0, 0.0],
    ])
    soph = _make_power_soph(truth, background=(0.0, 0.0, 0.05))
    opts = ParamBasisOpts.power()
    ws = opts.watershed_params

    fused, _ = extract_hist_peaks(soph, POWER_BINS, FREQ_BINS, *ws,
                                  use_fused=True)
    assembled, _ = extract_hist_peaks(soph, POWER_BINS, FREQ_BINS, *ws,
                                      use_fused=False)
    assert len(fused) >= len(assembled)
    assert len(fused) >= 2, "both planted modes should seed"
    for col in ("PeakFrequency", "SOFeature", "Height", "Duration",
                "Bandwidth"):
        assert col in fused.columns


def test_gof_matches_the_selected_iteration_not_the_last():
    """After a revert, the returned gof must describe the returned params.

    DYNAM-O_dev PR #79 fixed MATLAB to select `fitobj` and `gof` together.
    Keep the Python port on that same contract.
    """
    truth = np.array([[3.0, 13.5, 1.2, 10.0, 6.0, 0.0]])
    soph = _make_power_soph(truth)
    # max_peaks=3 with a strict criterion forces a revert after iteration 1.
    opts = ParamBasisOpts.power(max_peaks=3, criterion="minpctr2",
                                min_pctr2=0.9, min_amp=0.0)
    res = fit_param_basis_axis(soph, POWER_BINS, FREQ_BINS, opts,
                               seed_modes=truth)
    if res.fit_iteration and res.iter_rsquared:
        want = res.iter_rsquared[res.iter_numbers.index(res.fit_iteration)]
        assert res.gof["adjrsquare"] == pytest.approx(want, rel=1e-12)
