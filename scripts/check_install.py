#!/usr/bin/env python3
"""Smoke-check a pydynamo install: the package and the kernel bindings it calls."""

from importlib import import_module


def _require_module(name, required=()):
    try:
        module = import_module(name)
    except ImportError as exc:
        raise SystemExit(
            f"FAILED: could not import {name}: {exc}\n"
            "Run the DYNAM-O_toolbox controlled native build "
            "(`./bootstrap.sh --yes` or `bootstrap.ps1 -Yes`); see README.md."
        ) from exc

    missing = [attr for attr in required if not hasattr(module, attr)]
    if missing:
        raise SystemExit(
            f"FAILED: {name} is missing {', '.join(missing)}. "
            "Rerun the DYNAM-O_toolbox controlled native build "
            "(`./bootstrap.sh --yes` or `bootstrap.ps1 -Yes`); see README.md."
        )
    return module


def main():
    pydynamo = _require_module(
        "pydynamo",
        ("run_dynamo", "load_recording", "load_settings", "write_run"),
    )
    # The kernel bindings pydynamo calls (checked by name in pydynamo._kernel).
    from pydynamo import _kernel

    _require_module("dynamo_rs", _kernel._REQUIRED)
    import dynamo_rs

    print(
        f"OK: pydynamo {pydynamo.__version__} on dynamo_rs "
        f"{dynamo_rs.build_info()['build_info']} (semantics {dynamo_rs.SEMANTICS_VERSION})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
