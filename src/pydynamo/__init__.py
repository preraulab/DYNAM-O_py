"""pydynamo: the DYNAM-O sleep-EEG analysis from Python.

Every number comes from the shared Rust kernel (``dynamo_rs``), the code
the DYNAM-O desktop app and CLI run, so a pydynamo run of a recording
matches an app run of it under the same settings.

    from pydynamo import load_recording, load_settings, run_dynamo, write_run

    rec = load_recording("night.edf", "C3-A2", "night_staging.csv")
    out = run_dynamo(rec.data, rec.fs, rec.stage_times, rec.stage_vals,
                     settings=load_settings("batch_settings.json"))
    write_run(out, "results", "night", "C3-A2")

Stage convention (DYNAM-O): 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake.
"""

from pydynamo.io.run_tree import write_run
from pydynamo.load import Recording, load_from_settings_file, load_recording
from pydynamo.pipeline import DynamoOutput, SOPHsResult, run_dynamo
from pydynamo.plot import summary_plot
from pydynamo.settings import (
    SEMANTICS_VERSION, default_settings, load_settings, merge_settings, migrate_settings,
)
from pydynamo.soph.paramfit import ParamFitResult, fit_paramfits
from pydynamo.soph.splinefit import SplineFitResult, SplineObject, fit_splinefits

__all__ = [
    "run_dynamo", "DynamoOutput", "SOPHsResult",
    "load_recording", "load_from_settings_file", "Recording",
    "load_settings", "default_settings", "merge_settings", "migrate_settings",
    "SEMANTICS_VERSION",
    "fit_paramfits", "ParamFitResult", "fit_splinefits", "SplineFitResult", "SplineObject",
    "summary_plot", "write_run",
]

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
    __version__ = _pkg_version("pydynamo")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.3.0"
