"""Model selection across fit iterations.

Ports the `switch criterion` block at the end of param_basis_power.m (and the
identical block in param_basis_phase.m), plus kneedle.m.
"""

from __future__ import annotations

import warnings

import numpy as np


def kneedle(x, y, sensitivity=1.0):
    """Knee point of a concave curve — port of kneedle.m.

    Implements Satopaa et al. (2011). Returns ``(knee_xy, knee_idx, x_max)``.

    Deviation from MATLAB: kneedle.m interpolates with the Curve Fitting
    Toolbox `smoothingspline`, whose automatic smoothing parameter has no
    exact SciPy equivalent. We use `UnivariateSpline` with the default
    smoothing factor instead, so the knee can land on a slightly different
    iteration for borderline curves. This only matters for
    ``criterion='kneedle'``, which is not the default and which nothing in the
    toolbox selects.
    """
    from scipy.interpolate import UnivariateSpline

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if not 0 < sensitivity <= 1:
        raise ValueError("sensitivity must be in (0, 1]")

    if x.size > 3:
        k = 3
    else:
        k = max(1, x.size - 1)
    x_int = np.linspace(x[0], x[-1], 1000)
    try:
        spline = UnivariateSpline(x, y, k=k)
        y_int = spline(x_int)
    except Exception:
        y_int = np.interp(x_int, x, y)

    # Valid only if the curve bends downward somewhere (check_valid_curve).
    dy = np.gradient(y_int, x_int)
    ddy = np.gradient(dy, x_int)
    valid = bool(np.any(ddy < 0)) and x.size > 2

    if not valid:
        warnings.warn(
            "kneedle: curve has no negative curvature, picking max",
            RuntimeWarning, stacklevel=2,
        )
        idx = int(np.argmax(y))
        return (float(x[idx]), float(y[idx])), idx, float(x[idx])

    span_x = x_int.max() - x_int.min()
    span_y = y_int.max() - y_int.min()
    x_norm = (x_int - x_int.min()) / (span_x if span_x else 1.0)
    y_norm = (y_int - y_int.min()) / (span_y if span_y else 1.0)

    diff_curve = y_norm - x_norm
    threshold = sensitivity * diff_curve.max()
    knee_idx = int(np.argmax(diff_curve >= threshold))
    x_max = float(x_int[int(np.argmax(y_int))])
    return (float(x_int[knee_idx]), float(y_int[knee_idx])), knee_idx, x_max


def select_iteration(criterion, iter_numbers, iter_rsquared, *,
                     min_dr2=0.01, min_pctr2=0.01, kneedle_tol=0.01,
                     verbose=False):
    """Pick which fit iteration to keep.

    `iter_numbers` / `iter_rsquared` are the accepted ("good") iterations in
    order. Returns the chosen iteration number.
    """
    nums = np.asarray(iter_numbers, dtype=float).ravel()
    r2 = np.asarray(iter_rsquared, dtype=float).ravel()
    if nums.size == 0:
        raise ValueError("no accepted iterations to select from")
    if nums.size == 1:
        return int(nums[0])

    if criterion == "max":
        return int(nums[int(np.argmax(r2))])

    if criterion == "mindr2":
        # diff against a leading 0, so iteration 1's "jump" is its own r2.
        d = np.diff(np.concatenate(([0.0], r2)))
        hits = np.flatnonzero(d >= min_dr2)
        if hits.size == 0:
            return int(nums[0])
        return int(nums[hits[-1]])

    if criterion == "minpctr2":
        r2s = np.concatenate(([0.0], r2))
        abs_diff = np.abs(np.diff(r2s))
        prev = r2s[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = abs_diff / np.abs(prev)
        hits = np.flatnonzero(pct > min_pctr2)
        if hits.size == 0:
            return int(nums[0])
        return int(nums[hits[-1]])

    if criterion == "kneedle":
        knee, _, x_max = kneedle(nums, r2)
        knee_x = knee[0]
        if knee_x <= nums.min():
            return int(nums.min())
        if knee_x >= nums.max():
            return int(nums.max())
        after = np.flatnonzero(nums >= knee_x + kneedle_tol)
        if after.size == 0:
            return int(nums.max())
        fit_iteration = int(nums[after[0]])
        if fit_iteration > x_max and fit_iteration > 1:
            if verbose:
                print("Kneedle + 1 > max...")
            before = np.flatnonzero(nums < knee_x)
            if before.size:
                fit_iteration = int(nums[before[-1]])
        if verbose:
            print(f"Kneedle selects iteration {fit_iteration}")
        return fit_iteration

    raise ValueError(f"unknown criterion {criterion!r}")
