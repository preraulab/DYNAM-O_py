"""Basis-function evaluation and mode geometry.

Ports rotGauss.m, vmGauss.m, select_modes.m and mode_overlap.m. The fitting
itself lives in the Rust kernel (`dynamo_rs.fit_rotgauss` / `fit_vmgauss`);
these are the pure-evaluation pieces the outer loop needs between fits.

Every Gaussian factor here carries an explicit -1/2 in the exponent, which is
what makes the width parameters genuine standard deviations. The historical
form omitted it, so the fitted widths were sqrt(2) times the sigma they were
named after. These kernels must stay bit-for-bit equivalent to the Rust ones
in `rot_gauss.rs` / `vm_gauss.rs`: the fit runs in Rust and `core.py` then
re-evaluates the result here, so a disagreement corrupts the revert checks and
the returned `model_soph` without failing anything.
"""

from __future__ import annotations

import numpy as np

#: The sigma reparameterization factor. Historical Gaussian coefficients and
#: bounds are divided by this when preserving an existing fitted surface now
#: that the kernels carry the -1/2. Fresh estimators and priors are already in
#: the corrected parameter units. Never apply it to a von Mises `recikappa`.
SQRT2 = float(np.sqrt(2.0))


def rot_gauss(X, Y, amp, fmean, fstd, xmean, xstd, theta):
    """One rotated 2-D Gaussian — rotGauss.m:68.

    X is the SO-feature axis, Y is frequency. `fstd`/`xstd` are genuine
    standard deviations: the exponent carries the -1/2, so the surface one
    `fstd` from the center along the rotated frequency axis is exp(-1/2) of
    the peak. Both widths take the same convention here, unlike vm_gauss.
    """
    dy, dx = Y - fmean, X - xmean
    ct, st = np.cos(theta), np.sin(theta)
    a = (dy * ct + dx * st) / fstd
    b = (-dy * st + dx * ct) / xstd
    return amp * np.exp(-0.5 * (a ** 2 + b ** 2))


def vm_gauss(X, Y, amp, fmean, fstd, phasepref, recikappa, theta):
    """One von-Mises x Gaussian peak — vmGauss.m.

    Gaussian in frequency, von Mises in phase. Both widths are true standard
    deviations, but they get there by different routes and so are NOT
    symmetric under rescaling:

    * `fstd` is a sigma because of the explicit -1/2 below. Squaring the
      denominator (DYNAM-O_dev PR #71) was necessary but not sufficient.
    * `recikappa` (= 1/sqrt(kappa)) is ALREADY a sigma with no half needed,
      because the von Mises factor is its own small-angle Gaussian:
      exp(kappa*(cos d - 1)) -> exp(-d**2 / (2*recikappa**2)). Never rescale
      it alongside `fstd`.
    """
    dy = Y - fmean
    kappa = 1.0 / recikappa ** 2
    g = np.exp(-0.5 * (dy / fstd) ** 2)
    vm = np.exp(kappa * (np.cos(X - phasepref + dy * np.sin(theta)) - 1.0))
    return amp * g * vm


def eval_modes(params, x_bins, y_bins, kind="power", background=None,
               unit_row=False):
    """Evaluate a mode stack on the (y, x) grid.

    `params` is (N, 6); `background` is [xxx, yyy, zzz] or None. Returns an
    (n_y, n_x) array matching the SOPH orientation (freq down the rows).

    For `kind='phase'` the background is the sinusoidal baseline
    `xxx*sin(x + yyy) + zzz` rather than a plane.
    """
    X, Y = np.meshgrid(np.asarray(x_bins, float), np.asarray(y_bins, float))
    out = np.zeros_like(X, dtype=float)

    params = np.atleast_2d(np.asarray(params, dtype=float)) if params is not None \
        else np.empty((0, 6))
    fn = rot_gauss if kind == "power" else vm_gauss
    for row in params:
        out = out + fn(X, Y, *row[:6])

    if background is not None:
        xxx, yyy, zzz = (float(v) for v in background)
        if kind == "power":
            out = out + xxx * X + yyy * Y + zzz
        else:
            out = out + xxx * np.sin(X + yyy) + zzz

    if unit_row and kind == "phase":
        # normalized_vmGauss.m normalizes the assembled baseline + modes.
        # Rows that are entirely zero are left alone.
        rs = out.sum(axis=1, keepdims=True)
        out = np.divide(out, rs, out=np.zeros_like(out), where=rs != 0)
    return out


def select_mode(params, mode_idx, x_bins, y_bins, kind="power"):
    """Surface of a single mode with no background — select_modes.m.

    MATLAB zeroes every other coefficient (and the baseline) on the fit object
    and re-evaluates; evaluating the one mode directly is equivalent.
    """
    row = np.atleast_2d(np.asarray(params, dtype=float))[mode_idx]
    return eval_modes(row[None, :], x_bins, y_bins, kind=kind, background=None)


def mode_overlap(params, x_bins, y_bins, kind="power"):
    """Pairwise mode overlap — mode_overlap.m:42-49.

    overlap(p, q) = sum(min(Sp, Sq)) / sum(max(Sp, Sq)), i.e. intersection
    over union of the two mode surfaces. Strictly upper-triangular, matching
    MATLAB's `for q = p+1:N` loop; the lower triangle and diagonal stay 0.
    """
    params = np.atleast_2d(np.asarray(params, dtype=float))
    n = params.shape[0]
    overlap = np.zeros((n, n), dtype=float)
    if n < 2:
        return overlap

    surfaces = [select_mode(params, i, x_bins, y_bins, kind=kind)
                for i in range(n)]
    for p in range(n):
        for q in range(p + 1, n):
            lo = np.minimum(surfaces[p], surfaces[q]).sum()
            hi = np.maximum(surfaces[p], surfaces[q]).sum()
            overlap[p, q] = lo / hi if hi > 0 else 0.0
    return overlap


def min_pairwise_freq_diff(params):
    """Smallest |fmean_i - fmean_j| over all mode pairs (MATLAB `pdist`).

    Returns +inf for fewer than two modes so the caller's `< min_freq_diff`
    test is False, matching MATLAB's empty-pdist behaviour.
    """
    params = np.atleast_2d(np.asarray(params, dtype=float))
    if params.shape[0] < 2:
        return np.inf
    f = params[:, 1]
    d = np.abs(f[:, None] - f[None, :])
    np.fill_diagonal(d, np.inf)
    return float(d.min())
