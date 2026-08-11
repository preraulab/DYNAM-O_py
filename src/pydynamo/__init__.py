"""pydynamo — minimal Python port of DYNAM-O.

Exposes:
    run_dynamo(data, fs, stage_times, stage_vals, ...) -> DynamoOutput

Stage convention (DYNAM-O): 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake — reversed from
most EDF stagers. Histograms default to NREM (stages 1, 2, 3).
"""

from pydynamo.defaults import DetectionOpts, BaselineOpts, SOPHOpts
from pydynamo.pipeline import run_dynamo, DynamoOutput
from pydynamo.plot import summary_plot
from pydynamo.soph.paramfit import ParamBasisOpts, ParamFitResult
from pydynamo.soph.splinefit import (
    SplineBasisOpts,
    SplineFitResult,
    SplineObject,
)
from pydynamo import matlab_api  # noqa: F401  — exposed for `py.pydynamo.matlab_api.*`

__all__ = ["run_dynamo", "summary_plot", "DynamoOutput", "DetectionOpts",
           "BaselineOpts", "SOPHOpts", "ParamBasisOpts", "ParamFitResult",
           "SplineBasisOpts", "SplineFitResult", "SplineObject", "matlab_api"]

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
    __version__ = _pkg_version("pydynamo")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.2.0"
