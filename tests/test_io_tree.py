"""Canonical path builders and fix_filename sanitization."""

from pathlib import Path

import pytest

from pydynamo.io.tree import (
    aux_h5_path,
    fix_filename,
    paramfit_csv_path,
    runs_dir,
    soph_tiff_path,
    splinefit_tiff_path,
    stats_csv_path,
)


def test_fix_filename_only_strips_os_invalid_chars():
    # Mirrors the Rust matlab_layout tests: user-facing characters
    # (brackets, hyphens, parens, spaces) pass through unchanged.
    assert fix_filename("[C3-A2 - B]") == "[C3-A2 - B]"
    assert fix_filename("C3-A2 - B") == "C3-A2 - B"
    assert fix_filename("mean(C3, C4)") == "mean(C3, C4)"
    # Deny-list chars become '_', consecutive '_' runs collapse.
    assert fix_filename("Cz_avg = (C3+C4)/2") == "Cz_avg = (C3+C4)_2"
    assert fix_filename("C3:A2") == "C3_A2"
    assert fix_filename("a/b") == "a_b"
    assert fix_filename("x|y?z") == "x_y_z"
    assert fix_filename("hello*") == "hello_"
    # Reserved DOS names get an underscore suffix.
    assert fix_filename("CON") == "CON_"
    # Trailing dots and spaces are trimmed.
    assert fix_filename("foo.. ") == "foo"
    assert fix_filename("") == "unnamed"


def test_canonical_paths():
    root = Path("/results")
    assert stats_csv_path(root, "S001", "C3") == (
        root / "C3" / "TFpeaks" / "S001_stats_table_C3.csv"
    )
    assert soph_tiff_path(root, "S001", "C3", "power") == (
        root / "C3" / "SOPHs" / "S001_SOPHs_power_C3.tiff"
    )
    assert soph_tiff_path(root, "S001", "C3", "phase") == (
        root / "C3" / "SOPHs" / "S001_SOPHs_phase_C3.tiff"
    )
    assert aux_h5_path(root, "S001", "C3") == (
        root / "C3" / "auxiliary_data" / "S001_auxiliary_data_C3.h5"
    )
    assert paramfit_csv_path(root, "S001", "C3", "power") == (
        root / "C3" / "param_basis" / "S001_SOpower_paramfit_C3.csv"
    )
    assert splinefit_tiff_path(root, "S001", "C3", "phase") == (
        root / "C3" / "spline_basis" / "S001_SOphase_splinefit_C3.tiff"
    )
    assert runs_dir(root) == root / "_runs"


def test_channel_labels_are_sanitized_in_paths():
    root = Path("/results")
    p = stats_csv_path(root, "S 001", "[C3-A2 - B]")
    assert p == (root / "[C3-A2 - B]" / "TFpeaks"
                 / "S 001_stats_table_[C3-A2 - B].csv")
    q = stats_csv_path(root, "S/001", "C3:A2")
    assert q == root / "C3_A2" / "TFpeaks" / "S_001_stats_table_C3_A2.csv"


def test_invalid_axis_rejected():
    with pytest.raises(ValueError):
        paramfit_csv_path("/r", "S", "C3", "both")
    with pytest.raises(ValueError):
        soph_tiff_path("/r", "S", "C3", "sopower")
