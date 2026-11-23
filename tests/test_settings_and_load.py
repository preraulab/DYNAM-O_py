"""Settings files and recording loading go through the kernel."""

import json
import warnings

import numpy as np
import pytest

from pydynamo.load import _stage_label_lists, load_recording
from pydynamo.settings import (
    SEMANTICS_VERSION, default_settings, load_settings, merge_settings,
)


def test_default_settings_has_every_section():
    settings = default_settings()
    for section in ("mtm", "hist", "sopower", "paramfit", "splinefit", "output"):
        assert section in settings


def test_merge_keeps_unmentioned_keys():
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2]}
    merged = merge_settings(base, {"a": {"y": 3}, "b": [9]})
    assert merged == {"a": {"x": 1, "y": 3}, "b": [9]}
    assert base == {"a": {"x": 1, "y": 2}, "b": [1, 2]}


def test_load_current_file_is_silent(tmp_path):
    path = tmp_path / "settings.json"
    section = default_settings()
    path.write_text(json.dumps({"semantics_version": SEMANTICS_VERSION, "settings": section}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert load_settings(path) == section


def test_load_bare_section(tmp_path):
    section = {"hist": {"soph_stages": [1, 2]}}
    assert load_settings(section) == section


def test_load_old_file_migrates_with_warning():
    document = {"semantics_version": SEMANTICS_VERSION - 1, "settings": default_settings()}
    with pytest.warns(UserWarning, match="migrated"):
        section = load_settings(document)
    assert "mtm" in section


def test_stage_labels_from_settings_dict():
    lists = _stage_label_lists({"wake": "wake, W ,5", "n2": "N2"})
    assert len(lists) == 7
    assert lists[1] == ["wake", "W", "5"]
    assert lists[4] == ["N2"]
    assert lists[0] == []


def _write_edf(path, fs, signal, label="C3"):
    """A minimal one-signal EDF (16-bit, 1 s records)."""
    n_records = signal.size // int(fs)
    samples = int(fs)

    def field(text, width):
        return text.ljust(width)[:width].encode("ascii")

    header = b"".join([
        field("0", 8), field("X X X X", 80), field("Startdate X X X", 80),
        field("01.01.26", 8), field("22.00.00", 8), field(str(512), 8),
        field("", 44), field(str(n_records), 8), field("1", 8), field("1", 4),
    ])
    header += b"".join([
        field(label, 16), field("", 80), field("uV", 8),
        field("-500", 8), field("500", 8), field("-32768", 8), field("32767", 8),
        field("", 80), field(str(samples), 8), field("", 32),
    ])
    digital = np.round((signal[: n_records * samples] + 500.0) / 1000.0 * 65535.0 - 32768.0)
    digital = np.clip(digital, -32768, 32767).astype("<i2")
    path.write_bytes(header + digital.tobytes())


def test_load_recording_reads_edf_and_staging(tmp_path):
    fs = 100.0
    t = np.arange(int(120 * fs)) / fs
    signal = 50.0 * np.sin(2 * np.pi * 1.0 * t)
    edf = tmp_path / "rec.edf"
    _write_edf(edf, fs, signal)
    staging = tmp_path / "rec.csv"
    staging.write_text("0,2\n30,3\n60,1\n90,5\n")

    rec = load_recording(edf, "C3", staging)
    assert rec.fs == fs
    assert rec.data.size == signal.size
    np.testing.assert_allclose(rec.data, signal, atol=1000.0 / 65535.0)
    # The reader maps the file's labels (1=N1, 3=N3) to DYNAM-O codes (1=N3, 3=N1).
    np.testing.assert_array_equal(rec.stage_vals[:4], [2, 1, 3, 5])

    no_staging = load_recording(edf, "C3")
    assert set(np.unique(no_staging.stage_vals)) == {2}
