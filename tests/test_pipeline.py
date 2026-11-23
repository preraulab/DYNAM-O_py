"""run_dynamo is the kernel's run: its outputs equal a direct dynamo_rs call.

The numbers themselves are the kernel's business (it is tested against
MATLAB in DYNAM-O_rs). These tests pin the wrapper: settings reach the
kernel unchanged, every output is passed through without alteration, and
the result types are shaped as documented.
"""

import json
import warnings

import dynamo_rs
import numpy as np
import pytest

from pydynamo import run_dynamo
from pydynamo.pipeline import STATS_COLUMNS
from pydynamo.settings import default_settings


@pytest.fixture(scope="module")
def run(synthetic_night):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = run_dynamo(
            synthetic_night["data"], synthetic_night["fs"],
            synthetic_night["stage_times"], synthetic_night["stage_vals"],
            plot=False, verbose=False,
        )
    return out


@pytest.fixture(scope="module")
def direct(synthetic_night):
    """The same run, straight through the binding with default settings."""
    return dynamo_rs.run_dynamo(
        synthetic_night["data"], synthetic_night["fs"],
        synthetic_night["stage_times"], synthetic_night["stage_vals"],
        json.dumps(default_settings()), None, False, None,
    )


def test_finds_peaks(run):
    assert len(run.stats_table) > 50
    assert list(run.stats_table.columns) == list(STATS_COLUMNS)
    assert run.peak_in_hist.shape == (len(run.stats_table),)
    assert run.peak_in_hist.any()


def test_stats_table_is_the_kernels(run, direct):
    for name in STATS_COLUMNS:
        np.testing.assert_array_equal(
            run.stats_table[name].to_numpy(dtype=float),
            np.asarray(direct["peaks"][name], dtype=float),
            err_msg=name,
        )


def test_series_and_sophs_are_the_kernels(run, direct):
    pairs = {
        "artifacts": (run.artifacts, direct["artifacts"]),
        "baseline": (run.baseline, direct["baseline"]),
        "so_power_norm": (run.SOPHs.SOpower_norm, direct["so_power_norm"]),
        "so_phase_unwrapped": (run.SOphase_unwrapped, direct["so_phase_unwrapped"]),
        "sopower_hist": (run.SOPHs.SOpower_mat, direct["sopower_hist"]),
        "sophase_hist": (run.SOPHs.SOphase_mat, direct["sophase_hist"]),
        "freq_bins": (run.SOPHs.freq_bins, direct["freq_bins"]),
        "summary_spectrogram": (run.spect, direct["summary_spectrogram"]),
    }
    for name, (ours, theirs) in pairs.items():
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs), err_msg=name)


def test_fits_are_the_kernels(run, direct):
    power = run.SOPHs.SOpower_paramfit
    assert power is not None
    np.testing.assert_array_equal(
        power.params_table, np.asarray(direct["paramfit"]["power"]["table"]),
    )
    # model_soph is the kernel's model transposed to (n_freq, n_feature).
    np.testing.assert_array_equal(
        power.model_soph, np.asarray(direct["paramfit"]["power"]["model"]).T,
    )
    spline = run.SOPHs.SOpower_splinefit
    assert spline is not None
    np.testing.assert_array_equal(
        spline.coefs, np.asarray(direct["splinefit"]["power"]["coefs"]),
    )


def test_soph_orientation(run):
    sophs = run.SOPHs
    assert sophs.SOpower_mat.shape == (sophs.SOpower_bins.size, sophs.freq_bins.size)
    assert sophs.SOphase_mat.shape == (sophs.SOphase_bins.size, sophs.freq_bins.size)
    assert np.all(np.abs(sophs.SOphase[np.isfinite(sophs.SOphase)]) <= np.pi)


def test_flat_line_is_an_artifact(run):
    assert run.artifacts.any()
    spans = run.artifact_spans
    assert spans.ndim == 2 and spans.shape[1] == 2
    assert np.any((spans[:, 0] <= 605.0) & (spans[:, 1] >= 607.0))


def test_stamps_semantics(run):
    assert run.semantics == dynamo_rs.SEMANTICS_VERSION
    assert run.provenance.semantics == dynamo_rs.SEMANTICS_VERSION


def test_settings_reach_the_kernel(synthetic_night):
    settings = {"hist": {"soph_stages": [1]}}
    out = run_dynamo(
        synthetic_night["data"], synthetic_night["fs"],
        synthetic_night["stage_times"], synthetic_night["stage_vals"],
        settings=settings, fit_param_basis=False, fit_spline_basis=False,
        plot=False, verbose=False,
    )
    # The merged section keeps every default and applies the override.
    assert out.settings["hist"]["soph_stages"] == [1]
    assert out.settings["mtm"] == default_settings()["mtm"]
    # Only N3 peaks are counted in the SOPHs.
    counted_stages = out.stats_table["PeakStage"].to_numpy()[out.peak_in_hist]
    assert set(np.unique(counted_stages)) <= {1}
    assert out.SOPHs.SOpower_paramfit is None
    assert out.SOPHs.SOpower_splinefit is None
    assert out.skipped == {}


def test_time_range_limits_the_run(synthetic_night):
    out = run_dynamo(
        synthetic_night["data"], synthetic_night["fs"],
        synthetic_night["stage_times"], synthetic_night["stage_vals"],
        time_range=(120.0, 480.0), fit_param_basis=False, fit_spline_basis=False,
        plot=False, verbose=False,
    )
    times = out.stats_table["PeakTime"].to_numpy()
    assert times.min() >= 120.0 and times.max() <= 480.0


@pytest.mark.parametrize("settings", [
    {"mtm": {"dsfreqs": 0}},        # rejected by the run
    {"mtm": {"dsfreqs": "fine"}},   # rejected when the section is parsed
])
def test_bad_settings_raise(synthetic_night, settings):
    with pytest.raises(ValueError):
        run_dynamo(
            synthetic_night["data"], synthetic_night["fs"],
            synthetic_night["stage_times"], synthetic_night["stage_vals"],
            settings=settings, plot=False, verbose=False,
        )


def test_keep_spectrogram(synthetic_night):
    out = run_dynamo(
        synthetic_night["data"][: int(300 * synthetic_night["fs"])], synthetic_night["fs"],
        synthetic_night["stage_times"][:10], synthetic_night["stage_vals"][:10],
        fit_param_basis=False, fit_spline_basis=False,
        keep_spectrogram=True, plot=False, verbose=False,
    )
    spect = out.pass2_spectrogram
    assert spect["spect"].shape == (spect["sfreqs"].size, spect["stimes"].size)


def test_summary_figure(run, synthetic_night):
    import matplotlib
    matplotlib.use("Agg")
    from pydynamo import summary_plot

    fig = summary_plot(
        run.spect, run.stimes, run.sfreqs, run.artifacts, run.t,
        run.stage_times, run.stage_vals, run.stats_table, run.SOPHs,
        hist_peakidx=run.peak_in_hist,
    )
    assert len(fig.axes) >= 4


def test_standalone_fits_match_the_run(run):
    """Fitting the run's SOPHs again gives the run's fits."""
    from pydynamo import fit_paramfits, fit_splinefits

    sophs = run.SOPHs
    spline_power, spline_phase, _ = fit_splinefits(
        sophs.SOpower_mat, sophs.SOphase_mat,
        sophs.SOpower_bins, sophs.SOphase_bins, sophs.freq_bins,
        run.settings,
    )
    np.testing.assert_array_equal(spline_power.splinefit, sophs.SOpower_splinefit.splinefit)
    np.testing.assert_array_equal(spline_phase.coefs, sophs.SOphase_splinefit.coefs)

    param_power, _, _ = fit_paramfits(
        sophs.SOpower_mat, sophs.SOphase_mat,
        sophs.SOpower_bins, sophs.SOphase_bins, sophs.freq_bins,
        run.settings,
        stats_table=run.stats_table, peak_in_hist=run.peak_in_hist,
        baseline=(run.baseline, run.baseline_freqs),
    )
    np.testing.assert_array_equal(param_power.params_table, sophs.SOpower_paramfit.params_table)


def test_write_run_round_trips(run, tmp_path):
    import pydynamo.io as pio
    from pydynamo import write_run

    written = write_run(run, tmp_path, "S1", "C3")
    assert set(written) == {
        "stats", "soph_power", "soph_phase", "aux",
        "paramfit_power", "paramfit_phase", "splinefit_power", "splinefit_phase",
    }
    stats, prov = pio.read_stats_csv(written["stats"])
    assert len(stats) == len(run.stats_table)
    assert prov.semantics == run.semantics
    hist, meta, _ = pio.read_soph_tiff(written["soph_power"])
    np.testing.assert_array_equal(hist, run.SOPHs.SOpower_mat.astype(np.float32))
    aux, aux_prov = pio.read_auxiliary_data_h5(written["aux"])
    np.testing.assert_array_equal(aux["baseline"], run.baseline)
    assert aux_prov.semantics == run.semantics


def test_write_run_honors_output_switches(run, tmp_path):
    import copy
    from pydynamo import write_run

    quiet = copy.copy(run)
    quiet.settings = copy.deepcopy(run.settings)
    quiet.settings["output"].update(write_sophs=False, write_splinefits=False)
    written = write_run(quiet, tmp_path, "S1", "C3")
    assert not any(kind.startswith(("soph_", "splinefit_")) for kind in written)
