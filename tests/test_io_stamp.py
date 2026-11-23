"""Provenance stamp: grammar and preamble parsing."""

import pydynamo
from pydynamo.io.stamp import (
    Provenance,
    current_stamp,
    kernel_version,
    parse_preamble,
    semantics_version,
)
from pydynamo.settings import SEMANTICS_VERSION


def test_current_stamp_fields():
    stamp = current_stamp(format=3)
    assert stamp.format == 3
    assert stamp.writer == "pydynamo"
    assert stamp.writer_version == pydynamo.__version__
    # The kernel identity follows the <semver>+<sha12>[.dirty] grammar.
    semver, _, rest = stamp.kernel_version.partition("+")
    assert semver and rest
    sha = rest.removesuffix(".dirty")
    assert len(sha) == 12 and all(c in "0123456789abcdef" for c in sha)
    assert stamp.semantics == SEMANTICS_VERSION == semantics_version()


def test_current_stamp_format_defaults_to_none():
    assert current_stamp().format is None


def test_kernel_version_is_the_kernel_build():
    import dynamo_rs
    assert kernel_version() == dynamo_rs.build_info()["build_info"]


def test_parse_preamble_canonical_keys():
    stamp, extras = parse_preamble([
        "# DYNAM-O stats table",
        "# format: 3",
        "# writer: dynamo-cli",
        "# writer_version: 0.2.0+abcdefabcdef",
        "# kernel_version: 0.2.1+abcdefabcdef",
        "# semantics: 16",
        "# subjectID: S001",
        "PeakTime,PeakFrequency",
        "# not a preamble line (after the table starts)",
    ])
    assert stamp == Provenance(
        format=3,
        writer="dynamo-cli",
        writer_version="0.2.0+abcdefabcdef",
        kernel_version="0.2.1+abcdefabcdef",
        semantics=16,
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
