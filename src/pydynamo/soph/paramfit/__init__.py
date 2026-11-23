"""Parametric-basis fitting (SOPH dimensionality reduction).

A wrapper over the shared Rust kernel (`dynamo_rs.fit_paramfits`): the
seeding, mode loop, model selection and the per-mode table all run there,
the same code the DYNAM-O desktop app runs.
"""

from pydynamo.soph.paramfit.core import (
    PK_COLUMNS, ParamFitResult, fit_paramfits, paramfit_result_from_kernel,
    rebuild_phase_model,
)

__all__ = [
    "PK_COLUMNS",
    "ParamFitResult",
    "fit_paramfits",
    "paramfit_result_from_kernel",
    "rebuild_phase_model",
]
