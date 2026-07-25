#!/usr/bin/env python3
"""Smoke-check the documented three-repository development install."""

from importlib import import_module


def _require_module(name, required=()):
    try:
        module = import_module(name)
    except ImportError as exc:
        raise SystemExit(
            f"FAILED: could not import {name}: {exc}\n"
            "Follow the sibling checkout and native build steps in README.md."
        ) from exc

    missing = [attr for attr in required if not hasattr(module, attr)]
    if missing:
        raise SystemExit(
            f"FAILED: {name} is missing {', '.join(missing)}. "
            "Rebuild it from the branch documented in README.md."
        )
    return module


def main():
    pydynamo = _require_module("pydynamo", ("run_dynamo",))
    _require_module(
        "pydynamo.soph.paramfit",
        ("ParamBasisOpts", "fit_param_basis"),
    )
    _require_module(
        "dynamo_rs",
        (
            "extract_tfpeaks",
            "fit_rotgauss",
            "fit_vmgauss",
            "read_edf",
            "read_staging",
        ),
    )
    _require_module("multitaper_rs", ("compute_spectrogram",))
    wrapper = _require_module("multitaper_spectrogram_python")
    if not getattr(wrapper, "_HAS_RUST", False):
        raise SystemExit(
            "FAILED: the multitaper wrapper imported without its Rust backend. "
            "Rebuild multitaper_rs using the README.md command."
        )

    print(
        "OK: pydynamo "
        f"{pydynamo.__version__}, standalone paramfit, dynamo_rs, and "
        "multitaper_rs are importable."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
