"""Residual-max seeding — port of residual_max_seed.m.

Matching pursuit: once the watershed seeds are exhausted, each additional mode
is seeded at the argmax of the residual (SOPH - model), so it targets the
largest currently-unfit feature.
"""

from __future__ import annotations

import numpy as np

from pydynamo.soph.paramfit.basis import SQRT2


def residual_max_seed(soph_yx, model_yx, x_axis, y_axis, accepted_modes,
                      min_freq_diff, kind="power"):
    """Seed one new mode at the residual argmax.

    Returns ``(seed_row, found)`` where `seed_row` is
    ``[amp, fmean, fstd, xmean, xstd, theta]`` or None. `found` is False when
    no positive residual survives the frequency mask, which tells the caller
    to fall back to ``mean(B0i)``.

    Frequency bins within `min_freq_diff` of an accepted mode's center are
    masked out first; without that guard the residual peak beside an
    under-modelled mode gets picked and the new mode lands on top of the old
    one. `min_freq_diff = 0` disables the mask (the phase-axis convention).

    `kind` selects the width fallbacks, because slot 4 is a Gaussian sigma for
    ``'power'`` but `recikappa` for ``'phase'``, which was always a true sigma
    and takes no reparameterization rescale.
    """
    soph_yx = np.asarray(soph_yx, dtype=float)
    model_yx = np.asarray(model_yx, dtype=float)
    x_axis = np.asarray(x_axis, dtype=float).ravel()
    y_axis = np.asarray(y_axis, dtype=float).ravel()

    ny, nx = soph_yx.shape
    if model_yx.shape != (ny, nx):
        raise ValueError("model_yx must match soph_yx shape")
    if y_axis.size != ny or x_axis.size != nx:
        raise ValueError("axis lengths must match soph_yx dims")

    accepted = None
    if accepted_modes is not None:
        accepted = np.atleast_2d(np.asarray(accepted_modes, dtype=float))
        if accepted.size == 0:
            accepted = None

    excluded_y = np.zeros(ny, dtype=bool)
    if min_freq_diff > 0 and accepted is not None:
        fmeans = accepted[:, 1]
        excluded_y = np.any(
            np.abs(fmeans[None, :] - y_axis[:, None]) < min_freq_diff, axis=1
        )

    R = soph_yx - model_yx
    R = np.where(np.isfinite(R), R, -np.inf)
    R[excluded_y, :] = -np.inf

    lin = int(np.argmax(R))
    max_val = R.flat[lin]
    if not np.isfinite(max_val) or max_val <= 0:
        return None, False

    y_idx, x_idx = np.unravel_index(lin, (ny, nx))

    # Width priors: median of accepted-mode stds, with floors so the LM bounds
    # stay non-degenerate when there are no accepted modes yet. The medians
    # ride whatever the accepted params already carry, so only the literal
    # floors take the sigma rescale.
    fstd = 1.0 / SQRT2
    xstd = 5.0 / SQRT2 if kind == "power" else 5.0
    if accepted is not None:
        with np.errstate(all="ignore"):
            fstd_med = np.nanmedian(accepted[:, 2])
            xstd_med = np.nanmedian(accepted[:, 4])
        if np.isfinite(fstd_med) and fstd_med > 0:
            fstd = float(fstd_med)
        if np.isfinite(xstd_med) and xstd_med > 0:
            xstd = float(xstd_med)

    seed_row = np.array(
        [float(max_val), float(y_axis[y_idx]), fstd, float(x_axis[x_idx]), xstd, 0.0]
    )
    return seed_row, True
