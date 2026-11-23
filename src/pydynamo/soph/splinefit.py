"""Spline-basis fits of a recording's SOPHs: a wrapper over the kernel.

The fits run in the shared Rust kernel, the code the DYNAM-O app runs:
inside :func:`pydynamo.run_dynamo`, or on SOPHs you already have through
:func:`fit_splinefits` (``dynamo_rs.fit_splinefits``). Options are the
``splinefit`` keys of a DYNAM-O settings section.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline

from pydynamo import _kernel


@dataclass
class SplineObject:
    """Portable numerical equivalent of MATLAB's tensor spline object.

    ``coefs`` uses the emitted ``(frequency_basis, feature_basis)`` layout,
    ``knots_x`` / ``knots_y`` are the augmented knot vectors. Calling the
    object evaluates the fitted surface on feature/frequency axes.
    """

    coefs: np.ndarray
    knots_x: np.ndarray
    knots_y: np.ndarray
    order: tuple[int, int] = (4, 4)
    form: str = "B-"
    dim: int = 1

    @property
    def number(self) -> tuple[int, int]:
        return (self.coefs.shape[1], self.coefs.shape[0])

    def __call__(self, feature_bins, freq_bins) -> np.ndarray:
        feature_bins = np.asarray(feature_bins, dtype=float).ravel()
        freq_bins = np.asarray(freq_bins, dtype=float).ravel()
        feature_basis = BSpline.design_matrix(
            feature_bins, self.knots_x, self.order[0] - 1,
        ).toarray()
        freq_basis = BSpline.design_matrix(
            freq_bins, self.knots_y, self.order[1] - 1,
        ).toarray()
        return feature_basis @ self.coefs.T @ freq_basis.T


@dataclass
class SplineFitResult:
    """One axis's spline fit, as the kernel returns it.

    ``splinefit`` is the fitted surface on the fit window,
    ``(n_fit_feature, n_fit_freq)``; ``fit_SOfeature_bins`` /
    ``fit_freq_bins`` are that window's bin centers. ``coefs`` is
    ``(frequency_basis, feature_basis)``.
    """

    splinefit: np.ndarray
    coefs: np.ndarray
    spline_obj: SplineObject
    knots_x_aug: np.ndarray
    knots_y_aug: np.ndarray
    fit_SOfeature_bins: np.ndarray
    fit_freq_bins: np.ndarray


def _nearest_indices(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """For each value, the index of the nearest grid point."""
    indices = []
    for value in values:
        indices.append(int(np.argmin(np.abs(grid - value))))
    return np.asarray(indices, dtype=int)


def spline_result_from_kernel(fit: dict | None, so_bins, freq_bins) -> SplineFitResult | None:
    """A `SplineFitResult` from one axis of the kernel's splinefit output.

    The kernel returns the fitted surface on the whole SOPH grid
    (``so_bins`` x ``freq_bins``, zero outside the fit window); the result
    keeps the fit-window part, as the DYNAM-O app's splinefit TIFF and
    MATLAB's ``spline_basis`` do.
    """
    if fit is None:
        return None
    coefs = np.asarray(fit["coefs"], dtype=float)
    knots_x = np.asarray(fit["knots_x_aug"], dtype=float)
    knots_y = np.asarray(fit["knots_y_aug"], dtype=float)
    fit_so_bins = np.asarray(fit["fit_so_bins"], dtype=float)
    fit_freq_bins = np.asarray(fit["fit_freq_bins"], dtype=float)
    full_surface = np.asarray(fit["splinefit"], dtype=float)
    so_idx = _nearest_indices(np.asarray(so_bins, dtype=float).ravel(), fit_so_bins)
    freq_idx = _nearest_indices(np.asarray(freq_bins, dtype=float).ravel(), fit_freq_bins)
    window_surface = full_surface[np.ix_(so_idx, freq_idx)]
    return SplineFitResult(
        splinefit=window_surface,
        coefs=coefs,
        spline_obj=SplineObject(coefs=coefs, knots_x=knots_x, knots_y=knots_y),
        knots_x_aug=knots_x,
        knots_y_aug=knots_y,
        fit_SOfeature_bins=fit_so_bins,
        fit_freq_bins=fit_freq_bins,
    )


def fit_splinefits(
    sopower_mat,
    sophase_mat,
    sopower_bins,
    sophase_bins,
    freq_bins,
    settings: dict | None = None,
):
    """Fit both SOPHs (``(n_feature, n_freq)``). Returns
    ``(power, phase, skipped)``: `SplineFitResult` or None per axis, and
    the kernel's reason for each declined axis. ``settings`` is a DYNAM-O
    settings section (dict) or None for the defaults."""
    def vector(values):
        return np.ascontiguousarray(np.asarray(values, dtype=float).ravel())

    out = _kernel.rs().fit_splinefits(
        np.ascontiguousarray(np.asarray(sopower_mat, dtype=float)),
        np.ascontiguousarray(np.asarray(sophase_mat, dtype=float)),
        vector(sopower_bins),
        vector(sophase_bins),
        vector(freq_bins),
        None if settings is None else json.dumps(settings),
    )
    power = spline_result_from_kernel(out["power"], sopower_bins, freq_bins)
    phase = spline_result_from_kernel(out["phase"], sophase_bins, freq_bins)
    return power, phase, dict(out["skipped"])


__all__ = [
    "SplineObject", "SplineFitResult", "fit_splinefits", "spline_result_from_kernel",
]
