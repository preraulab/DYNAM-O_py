"""Parametric-basis fitting (SOPH dimensionality reduction).

Port of DYNAM-O/toolbox/SOPH_dim_reduction/parametric_basis/. The
nonlinear fits themselves are the shared Rust kernels
(`dynamo_rs.fit_rotgauss` / `fit_vmgauss`); this package is the mode search,
seeding, and model-selection loop around them.
"""

from pydynamo.soph.paramfit.basis import (
    eval_modes, mode_overlap, rot_gauss, select_mode, vm_gauss,
)
from pydynamo.soph.paramfit.core import (
    ParamFitResult, fit_param_basis, fit_param_basis_axis,
)
from pydynamo.soph.paramfit.histpeaks import extract_hist_peaks, seeds_from_stats
from pydynamo.soph.paramfit.opts import ParamBasisOpts
from pydynamo.soph.paramfit.seed import residual_max_seed
from pydynamo.soph.paramfit.select import kneedle, select_iteration

__all__ = [
    "ParamBasisOpts",
    "ParamFitResult",
    "fit_param_basis",
    "fit_param_basis_axis",
    "extract_hist_peaks",
    "seeds_from_stats",
    "eval_modes",
    "mode_overlap",
    "rot_gauss",
    "vm_gauss",
    "select_mode",
    "residual_max_seed",
    "select_iteration",
    "kneedle",
]
