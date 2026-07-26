"""Small MATLAB-semantics helpers used by the paramfit port."""

from __future__ import annotations

import numpy as np


def prctile(v, q):
    """MATLAB `prctile` — midpoint plotting positions, NaN omitted.

    MATLAB places the sorted values at percentiles ``100*(k-0.5)/n`` and
    interpolates linearly between them, clamping outside that range. NumPy's
    default ('linear' on ``(k-1)/(n-1)``) puts them somewhere else, so the two
    disagree in the tails — exactly where the 5th-percentile fit seed and the
    95th-percentile merge threshold live. Both of those feed the fit, so this
    difference is not cosmetic.

    ``q`` may be a scalar or array, in percent.
    """
    v = np.asarray(v, dtype=float).ravel()
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.full(np.shape(q), np.nan) if np.ndim(q) else np.nan
    if v.size == 1:
        return np.full(np.shape(q), v[0]) if np.ndim(q) else float(v[0])

    s = np.sort(v)
    n = s.size
    pos = 100.0 * (np.arange(n) + 0.5) / n
    out = np.interp(np.asarray(q, dtype=float), pos, s)  # np.interp clamps
    return out if np.ndim(q) else float(out)
