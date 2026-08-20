"""Round-trip tests for every pydynamo.io writer/reader pair.

Synthetic data only — these test the on-disk contract
(OUTPUT_FORMAT.md §1-2, §8), not the pipeline numerics.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pydynamo.io as pio
from pydynamo.io.stats import STATS_CSV_HEADER, STATS_CSV_HEADER_V1
from pydynamo.soph.paramfit.output import create_params_table


@pytest.fixture
def stamp():
    return pio.Provenance(
        format=None,
        writer="pydynamo",
        writer_version="0.2.0+abcdefabcdef",
        kernel_version="0.2.1+abcdefabcdef",
    )


@pytest.fixture
def stats_df():
    return pd.DataFrame({
        "PeakTime": [10.123456789, 20.0],
        "PeakFrequency": [12.5, 14.0],
        "Height": [0.8, 0.9],
        "Area": [4.0, 4.5],
        "Duration": [1.2, 1.3],
        "Bandwidth": [0.5, 0.7],
        "Volume": [1.0, 1.1],
        "Peakiness": [5.0, 5.5],
        "BoundingBox": [(9.5, 12.0, 1.2, 0.5), (19.5, 13.5, 1.3, 0.7)],
        "Boundaries": [np.zeros((2, 2)), np.zeros((2, 2))],
        "HeightData": [np.array([1.0]), np.array([2.0])],
        "SegmentNum": [3, 3],
        "PeakStage": [2.0, 3.0],
        "SOpower": [0.42, np.nan],
        "SOphase": [-1.7, np.nan],
    })


# ---------------------------------------------------------------- stats CSV

def test_stats_csv_first_data_line_is_canonical_header(
    tmp_path, stats_df, stamp,
):
    p = tmp_path / "s.csv"
    pio.write_stats_csv(stats_df, p, stamp, subject_id="S001")
    lines = p.read_text().splitlines()
    non_comment = [ln for ln in lines if not ln.startswith("#")]
    assert non_comment[0] == STATS_CSV_HEADER
    # Preamble carries exactly the frozen stamp block (§8.1).
    assert lines[0] == "# DYNAM-O stats table"
    assert lines[1] == "# format: 3"
    assert lines[2] == "# writer: pydynamo"
    assert lines[3] == "# writer_version: 0.2.0+abcdefabcdef"
    assert lines[4] == "# kernel_version: 0.2.1+abcdefabcdef"
    assert lines[5] == "# subjectID: S001"


def test_stats_csv_round_trip(tmp_path, stats_df, stamp):
    p = tmp_path / "s.csv"
    pio.write_stats_csv(stats_df, p, stamp, subject_id="S001")
    df, prov = pio.read_stats_csv(p)
    assert prov.format == 3
    assert prov.writer == "pydynamo"
    assert list(df.columns) == STATS_CSV_HEADER.split(",")
    assert df["PeakFrequency"].tolist() == [12.5, 14.0]
    assert df["bbox_tl_s"].tolist() == [9.5, 19.5]
    assert df["PeakStage"].tolist() == [2, 3]
    assert np.isnan(df["SOpower"][1]) and np.isnan(df["SOphase"][1])
    # Class-based rounding: PeakTime at 8 significant figures.
    assert df["PeakTime"][0] == 10.123457


def test_stats_csv_provenance_is_preamble_only(tmp_path, stats_df, stamp):
    # stamp=None writes the bare format-2 file with byte-identical data.
    stamped = tmp_path / "stamped.csv"
    bare = tmp_path / "bare.csv"
    pio.write_stats_csv(stats_df, stamped, stamp, subject_id="S001")
    pio.write_stats_csv(stats_df, bare, None)
    stamped_lines = stamped.read_text().splitlines()
    stripped = [ln for ln in stamped_lines if not ln.startswith("#")]
    assert stripped == bare.read_text().splitlines()
    _, prov = pio.read_stats_csv(bare)
    assert prov.format == 2, "bare 14-column file infers format 2"
    assert prov.writer is None


def test_stats_csv_reads_handwritten_format1_fixture(tmp_path):
    p = tmp_path / "v1.csv"
    p.write_text(
        STATS_CSV_HEADER_V1 + "\n"
        "10.0,12.5,1.2,0.5,0.8,1.0,3,4.0,5.0,9.5,12.0,1.3,0.7,2,0.42,-1.7\n"
    )
    df, prov = pio.read_stats_csv(p)
    assert prov.format == 1
    assert prov.writer is None
    assert df["bbox_width_s"][0] == 1.3
    assert df["bbox_height_Hz"][0] == 0.7
    assert df["SOphase"][0] == -1.7


def test_stats_csv_reads_handwritten_format2_fixture(tmp_path):
    p = tmp_path / "v2.csv"
    p.write_text(
        STATS_CSV_HEADER + "\n"
        "10.0,12.5,1.2,0.5,0.8,1.0,3,4.0,5.0,9.5,12.0,2,NaN,NaN\n"
    )
    df, prov = pio.read_stats_csv(p)
    assert prov.format == 2
    assert df["PeakTime"][0] == 10.0
    assert np.isnan(df["SOpower"][0])


def test_stats_csv_rejects_unknown_header(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("PeakTime,PeakFrequency\n1.0,2.0\n")
    with pytest.raises(ValueError, match="unexpected header"):
        pio.read_stats_csv(p)


def test_io_compat_load_stats_csv_skips_preamble(tmp_path, stats_df, stamp):
    from pydynamo.io_compat import load_stats_csv

    p = tmp_path / "s.csv"
    pio.write_stats_csv(stats_df, p, stamp, subject_id="S001")
    df = load_stats_csv(p)
    assert len(df) == 2
    assert "PeakTime" in df.columns


# ------------------------------------------------------------- paramfit CSV

def _power_fit():
    table = create_params_table(
        np.array([[2.0, 11.0, 1.5, 3.0, 4.0, 0.0]]), "power"
    )
    return SimpleNamespace(
        background=np.array([0.1, 0.2, 0.3]),
        gof={"sse": 1.0, "rsquare": 0.9, "adjrsquare": 0.8,
             "rmse": 0.1, "dfe": 6.0, "dfm": 5.0},
        params_table=table,
    )


def test_paramfit_csv_round_trip_power(tmp_path, stamp):
    p = tmp_path / "pf.csv"
    pio.write_paramfit_csv(
        p, _power_fit(), "power", [0.0, 1.0, 2.0], [10.0, 11.0, 12.0],
        "S1", stamp,
    )
    df, meta, prov = pio.read_paramfit_csv(p)
    assert prov.format == 3
    assert prov.writer == "pydynamo"
    assert meta["fit_type"] == "power"
    assert meta["n_modes"] == 1
    assert meta["gof.rsquare"] == 0.9
    assert meta["background.PowSlope"] == 0.1
    np.testing.assert_allclose(meta["SOpower_bins"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(meta["freq_bins"], [10.0, 11.0, 12.0])
    assert df.columns[0] == "Density"
    assert "PrefPhase" in df.columns and "PkSOphase" in df.columns
    # Mode volume: density * 2π * xstd * fstd.
    np.testing.assert_allclose(
        df["Volume"][0], 2.0 * 2.0 * np.pi * 4.0 * 1.5
    )
    assert df["PkCount"][0] == 0


def test_paramfit_csv_round_trip_phase(tmp_path, stamp):
    table = create_params_table(
        np.array([[0.5, 11.0, 1.5, 0.3, 1.2, 0.0]]), "phase"
    )
    fit = SimpleNamespace(
        background=np.array([0.4, -0.5, 0.6]),
        gof={"sse": 1.0, "rsquare": 0.9, "adjrsquare": 0.8,
             "rmse": 0.1, "dfe": 6.0, "dfm": 5.0},
        params_table=table,
    )
    p = tmp_path / "pf.csv"
    pio.write_paramfit_csv(
        p, fit, "phase", [-3.0, 0.0, 3.0], [10.0, 11.0, 12.0], "S1", stamp,
    )
    df, meta, prov = pio.read_paramfit_csv(p)
    assert meta["fit_type"] == "phase"
    assert meta["background.SinAmp"] == 0.4
    # The canonical phase column set includes the cross-axis SOpowerMean
    # (NaN placeholder — pydynamo's table does not carry it).
    assert list(df.columns[:8]) == [
        "Density", "FreqMean", "FreqStd", "SOphaseMean", "SOphaseStd",
        "Theta", "Volume", "SOpowerMean",
    ]
    assert np.isnan(df["SOpowerMean"][0])
    assert np.isnan(meta["unit_row"])


def test_paramfit_csv_reads_legacy_format1(tmp_path):
    p = tmp_path / "legacy.csv"
    p.write_text(
        "# DYNAM-O parametric fit\n"
        "# version: 1\n"
        "# code_version: 0.1.0 (abc)\n"
        "# fit_type: power\n"
        "# n_modes: 1\n"
        "# freq_bins: [10.0,null]\n"
        "Density,FreqMean\n"
        "1.0,11.0\n"
    )
    df, meta, prov = pio.read_paramfit_csv(p)
    assert prov.format == 1
    assert prov.writer_version == "0.1.0 (abc)"
    assert meta["n_modes"] == 1
    np.testing.assert_array_equal(
        meta["freq_bins"], np.array([10.0, np.nan])
    )
    assert list(df.columns) == ["Density", "FreqMean"]


# ----------------------------------------------------------------- SOPH TIFF

def test_soph_tiff_round_trip(tmp_path, stamp):
    hist = np.arange(12, dtype=float).reshape(3, 4)
    hist[0, 0] = np.nan
    p = tmp_path / "h.tiff"
    pio.write_soph_tiff(
        p, hist, [0.0, 1.0, 2.0], [10.0, 11.0, 12.0, 13.0],
        "sophase", "S1", stamp,
    )
    back, meta, prov = pio.read_soph_tiff(p)
    assert prov.format == 2
    assert prov.kernel_version == "0.2.1+abcdefabcdef"
    assert back.shape == (3, 4)
    assert np.isnan(back[0, 0])
    np.testing.assert_allclose(back.ravel()[1:], hist.ravel()[1:])
    assert meta["label"] == "sophase"
    assert meta["pixel_format"] == "f32 row-major"
    # Both the Rust-native keys and the MATLAB aliases are present.
    np.testing.assert_allclose(meta["row_centers"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(meta["SOphase_bins"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(meta["freq_bins"], [10.0, 11.0, 12.0, 13.0])
    assert meta["subjectID"] == "S1"


def test_soph_tiff_format1_reads_as_legacy(tmp_path):
    # Format 1: string-valued "format", no stamp keys.
    import tifffile

    meta = {"label": "sopower", "rows": 2, "cols": 2,
            "row_centers": [0.0, 1.0], "col_centers": [10.0, 11.0],
            "format": "f32 row-major"}
    p = tmp_path / "legacy.tiff"
    tifffile.imwrite(
        p, np.zeros((2, 2), dtype=np.float32),
        photometric="minisblack", description=json.dumps(meta),
        metadata=None,
    )
    _, meta_back, prov = pio.read_soph_tiff(p)
    assert prov.format == 1
    assert prov.writer is None
    assert meta_back["label"] == "sopower"


def test_soph_tiff_rejects_mismatched_bins(tmp_path, stamp):
    with pytest.raises(ValueError, match="hist rows"):
        pio.write_soph_tiff(
            tmp_path / "x.tiff", np.zeros((3, 4)), [0.0], [1.0] * 4,
            "sopower", "S1", stamp,
        )


# ------------------------------------------------------------ splinefit TIFF

def test_splinefit_tiff_round_trip(tmp_path, stamp):
    fit = SimpleNamespace(
        coefs=np.arange(20, dtype=float).reshape(5, 4),
        splinefit=np.arange(12, dtype=float).reshape(3, 4),
        knots_x_aug=np.array([0.0, 0.0, 1.0, 2.0, 2.0]),
        knots_y_aug=np.array([5.0, 5.0, 6.0, 7.0, 7.0]),
        fit_freq_bins=np.array([10.0, 11.0, 12.0, 13.0]),
        fit_SOfeature_bins=np.array([0.0, 1.0, 2.0]),
    )
    p = tmp_path / "sf.tiff"
    pio.write_splinefit_tiff(p, fit, "power", "S1", stamp)
    coefs, splinefit, meta, prov = pio.read_splinefit_tiff(p)
    assert prov.format == 2
    np.testing.assert_allclose(coefs, fit.coefs)
    np.testing.assert_allclose(splinefit, fit.splinefit)
    assert meta["label"] == "splinefit"
    assert meta["page1"] == "coefs" and meta["page2"] == "splinefit"
    np.testing.assert_allclose(meta["knots_x"], fit.knots_x_aug)
    np.testing.assert_allclose(meta["knots_y"], fit.knots_y_aug)
    np.testing.assert_allclose(meta["SOpower_bins"], fit.fit_SOfeature_bins)
    np.testing.assert_allclose(meta["freq_bins"], fit.fit_freq_bins)


# --------------------------------------------------------------------- aux h5

def test_aux_h5_round_trip(tmp_path, stamp):
    p = tmp_path / "aux.h5"
    pio.write_auxiliary_data_h5(
        p,
        fs=100.0,
        so_power_norm=[1.0, 2.0, 3.0, 4.0, 5.0],
        so_power_t_start=2.5,
        so_power_norm_method="p2shift1234",
        so_power_window_params=(5.0, 0.5),
        so_power_freqrange=(0.3, 1.5),
        artifact_spans=[[10.0, 20.0], [55.5, 60.0]],
        stage_times=[0.0, 30.0, 60.0],
        stage_vals=[5, 2, 1],
        subject_id="TS00404",
        stamp=stamp,
    )
    aux, prov = pio.read_auxiliary_data_h5(p)
    assert prov.format == 2
    assert prov.writer == "pydynamo"
    assert prov.writer_version == "0.2.0+abcdefabcdef"
    assert aux["fs"] == 100.0
    np.testing.assert_allclose(aux["so_power_norm"], [1, 2, 3, 4, 5])
    assert aux["so_power_t_start"] == 2.5
    assert aux["so_power_norm_method"] == "p2shift1234"
    assert aux["so_power_window_params"] == (5.0, 0.5)
    assert aux["so_power_freqrange"] == (0.3, 1.5)
    np.testing.assert_allclose(
        aux["artifact_spans"], [[10.0, 20.0], [55.5, 60.0]]
    )
    np.testing.assert_allclose(aux["stage_times"], [0.0, 30.0, 60.0])
    np.testing.assert_array_equal(aux["stage_vals"], [5, 2, 1])
    assert aux["subject_id"] == "TS00404"


def test_aux_h5_matlab_shapes_and_legacy_read(tmp_path, stamp):
    import h5py

    p = tmp_path / "aux.h5"
    pio.write_auxiliary_data_h5(
        p, fs=256.0, so_power_norm=[1.0, 2.0, 3.0],
        so_power_t_start=0.0, so_power_norm_method="p2shift1234",
        so_power_window_params=(5.0, 0.5), so_power_freqrange=(0.3, 1.5),
        stage_times=[0.0, 30.0], stage_vals=[2, 3], stamp=stamp,
    )
    with h5py.File(p) as f:
        # MATLAB orientation: column vector for SO-power, row vectors
        # for the staging timeline, (1,1) scalars.
        assert f["Fs"].shape == (1, 1)
        assert f["SOpower_norm"].shape == (3, 1)
        assert f["SOpower_window_params"].shape == (2, 1)
        assert f["stage_times"].shape == (1, 2)
        assert f["stage_vals"].shape == (1, 2)
        assert f["stage_vals"].dtype == np.uint8
        # Legacy dataset kept beside the stamp, same value.
        assert f["code_version"][0, 0].decode() == "0.2.0+abcdefabcdef"
        # Empty optionals omitted, not written empty.
        assert "artifact_spans" not in f
        assert "subjectID" not in f

    # A format-1 file (code_version only) still reads; the legacy build
    # id surfaces as writer_version.
    legacy = tmp_path / "legacy.h5"
    with h5py.File(legacy, "w") as f:
        f.create_dataset("Fs", data=np.array([[100.0]]))
        f.create_dataset(
            "code_version",
            shape=(1, 1),
            dtype=h5py.string_dtype(encoding="utf-8"),
            data=np.array([["0.1.0 (abc)"]], dtype=object),
        )
    aux, prov = pio.read_auxiliary_data_h5(legacy)
    assert prov.format is None
    assert prov.writer is None
    assert prov.writer_version == "0.1.0 (abc)"
    assert aux["fs"] == 100.0
    assert aux["so_power_norm"].size == 0


# ---------------------------------------------------------------- runs jsonl

def test_runs_jsonl_header_and_item_key_sets(tmp_path, stamp):
    pio.append_run_header(
        tmp_path, mode="run", n_items=1, n_subjects=1, stamp=stamp,
    )
    pio.append_run_item(
        tmp_path, subject="S001", channel="[C3-A2 - B]",
        input_file="/data/S001.edf", peaks=42, duration_sec=1.5,
        stamp=stamp,
    )
    path = pio.runs_jsonl_path(tmp_path)
    assert path.parent == tmp_path / "_runs"
    assert path.name.endswith(".jsonl") and "_pid" in path.name

    lines = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert [ln["kind"] for ln in lines] == ["header", "item"]

    header, item = lines
    # Key sets mirror matlab_layout::append_run_index.
    assert set(header) == {
        "ts", "kind", "host", "user", "pid", "run_id", "code_version",
        "writer", "writer_version", "kernel_version", "mode",
        "jobs_phase1", "jobs_phase2", "n_items", "n_subjects",
    }
    assert set(item) == {
        "ts", "kind", "host", "user", "pid", "run_id", "code_version",
        "writer", "writer_version", "kernel_version", "subject",
        "channel", "input_file", "staging_file", "components", "files",
        "status", "failures", "peaks", "duration_sec", "mode",
        "jobs_phase1", "jobs_phase2",
    }
    for ln in lines:
        assert ln["writer"] == "pydynamo"
        assert ln["writer_version"] == "0.2.0+abcdefabcdef"
        assert ln["kernel_version"] == "0.2.1+abcdefabcdef"
        assert ln["code_version"] == ln["writer_version"]
    assert item["channel"] == "[C3-A2 - B]"
    assert item["peaks"] == 42
    assert item["files"][0] == (
        "[C3-A2 - B]/TFpeaks/S001_stats_table_[C3-A2 - B].csv"
    )


# ------------------------------------------------------- fallback provenance

def test_writers_stamp_python_native_when_fallback_latched(tmp_path):
    from pydynamo import _kernel
    from pydynamo.io.stamp import current_stamp

    saved = _kernel.FALLBACK_ACTIVE
    try:
        _kernel.FALLBACK_ACTIVE = True
        df = pd.DataFrame()
        p = tmp_path / "s.csv"
        pio.write_stats_csv(df, p, current_stamp(), subject_id="S1")
        _, prov = pio.read_stats_csv(p)
        assert prov.kernel_version == "python-native"
    finally:
        _kernel.FALLBACK_ACTIVE = saved
