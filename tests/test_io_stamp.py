"""Provenance stamp: grammar, preamble parsing, and fallback stamping."""

import pytest

from pydynamo import _kernel
from pydynamo.io.stamp import (
    Provenance,
    current_stamp,
    kernel_version,
    parse_preamble,
)


@pytest.fixture
def restore_fallback_flag():
    """Save and restore the process-wide fallback latch around a test."""
    saved = _kernel.FALLBACK_ACTIVE
    yield
    _kernel.FALLBACK_ACTIVE = saved


def test_current_stamp_fields(restore_fallback_flag):
    _kernel.FALLBACK_ACTIVE = False
    stamp = current_stamp(format=3)
    assert stamp.format == 3
    assert stamp.writer == "pydynamo"
    # Installed distribution version (plain semver is the best available
    # writer_version when no git sha is known).
    assert stamp.writer_version == "0.2.0"
    # dynamo_rs is installed in the test env, so the kernel identity
    # follows the <semver>+<sha12>[.dirty] grammar.
    semver, _, rest = stamp.kernel_version.partition("+")
    assert semver and rest
    sha = rest.removesuffix(".dirty")
    assert len(sha) == 12 and all(c in "0123456789abcdef" for c in sha)


def test_current_stamp_format_defaults_to_none():
    assert current_stamp().format is None


def test_python_native_when_fallback_latched(restore_fallback_flag):
    _kernel.FALLBACK_ACTIVE = True
    assert kernel_version() == "python-native"
    assert current_stamp().kernel_version == "python-native"


def test_record_fallback_warns_once(restore_fallback_flag):
    _kernel.FALLBACK_ACTIVE = False
    with pytest.warns(RuntimeWarning, match="pure-Python fallback"):
        _kernel.record_fallback("feature-one")
    # Second call: latch already set, no further warning.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _kernel.record_fallback("feature-two")
    assert _kernel.FALLBACK_ACTIVE


def test_parse_preamble_canonical_keys():
    stamp, extras = parse_preamble([
        "# DYNAM-O stats table",
        "# format: 3",
        "# writer: dynamo-cli",
        "# writer_version: 0.2.0+abcdefabcdef",
        "# kernel_version: 0.2.1+abcdefabcdef",
        "# subjectID: S001",
        "PeakTime,PeakFrequency",
        "# not a preamble line (after the table starts)",
    ])
    assert stamp == Provenance(
        format=3,
        writer="dynamo-cli",
        writer_version="0.2.0+abcdefabcdef",
        kernel_version="0.2.1+abcdefabcdef",
    )
    assert extras == {"subjectID": "S001"}


def test_parse_preamble_legacy_spellings():
    # Paramfit formats 1-2 spelled format as 'version' and
    # writer_version as 'code_version'.
    stamp, _ = parse_preamble([
        "# version: 1",
        "# code_version: 0.1.0 (abc)",
    ])
    assert stamp.format == 1
    assert stamp.writer_version == "0.1.0 (abc)"

    # Canonical keys win over legacy ones regardless of order.
    stamp, _ = parse_preamble([
        "# format: 3",
        "# version: 1",
        "# writer_version: 0.2.0",
        "# code_version: 0.1.0",
    ])
    assert stamp.format == 3
    assert stamp.writer_version == "0.2.0"


def test_parse_preamble_never_fails_on_junk():
    stamp, extras = parse_preamble([
        "# title line without colon",
        "# format: not-an-int",
        "# mystery_key: kept raw",
    ])
    assert stamp.format is None
    assert extras["mystery_key"] == "kept raw"
