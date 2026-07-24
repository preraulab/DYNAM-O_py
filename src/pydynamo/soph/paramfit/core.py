"""The parametric-basis fitting loop — port of param_basis_power.m /
param_basis_phase.m.

Structure, following MATLAB:

1. Restrict to the valid analysis window (finite bins inside the limits).
2. Seed modes from a watershed over the histogram itself (`extract_hist_peaks`).
3. Iteratively add one mode at a time. Each iteration refits *all* modes
   jointly; modes past the watershed supply are seeded by matching pursuit on
   the residual.
4. After each fit, decide whether to keep the new mode ("revert" checks:
   R-squared gain, mode overlap, amplitude floor, frequency separation).
5. Choose which accepted iteration to return via `criterion`.

The nonlinear fit itself is the Rust kernel — `dynamo_rs.fit_rotgauss` for the
power axis, `fit_vmgauss` for phase — which ports MATLAB's bounded
trust-region solver. Everything else here is the search around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pydynamo.soph.paramfit.basis import (
    eval_modes, min_pairwise_freq_diff, mode_overlap,
)
from pydynamo.soph.paramfit.matlab_compat import prctile
from pydynamo.soph.paramfit.opts import ParamBasisOpts, resolve_bounds
from pydynamo.soph.paramfit.seed import residual_max_seed
from pydynamo.soph.paramfit.select import select_iteration

try:
    import dynamo_rs as _rs
except ImportError:  # pragma: no cover - the fit kernels are required
    _rs = None


# Background-plane bounds, from fit_rotGauss.m:111-113 / fit_vmGauss.m.
_BG_SLOPE_LIM = 0.1


def orient_soph(soph, x_bins, y_bins):
    """Return `soph` as (n_freq, n_feature), transposing if needed.

    Port of param_basis_power.m:120-127. The pipeline stores SOPH matrices as
    (n_feature_bins, n_freq_bins) — see SOpowerphaseHistogram.m:45-47 — while
    the fitter works on them as images with frequency down the rows, so one of
    the two callers always has to flip. MATLAB sniffs the shape rather than
    demanding a convention; we do the same so either layout is accepted.
    """
    soph = np.asarray(soph, dtype=float)
    x_bins = np.asarray(x_bins, dtype=float).ravel()
    y_bins = np.asarray(y_bins, dtype=float).ravel()
    if soph.shape == (x_bins.size, y_bins.size) and x_bins.size != y_bins.size:
        return soph.T
    if soph.shape == (y_bins.size, x_bins.size):
        return soph
    raise ValueError(
        f"incompatible SOPH dimensions: soph {soph.shape}, "
        f"x_bins {x_bins.size}, y_bins {y_bins.size}"
    )


@dataclass
class ParamFitResult:
    """One axis's parametric fit.

    `params` is (N, 6) — [amp, fmean, fstd, xmean, xstd, theta]. `model_soph`
    is evaluated on the *full* input grid, not just the analysis window, which
    is what MATLAB returns.
    """
    params: np.ndarray
    background: np.ndarray
    model_soph: np.ndarray
    gof: dict
    wshed_img: np.ndarray | None
    fit_iteration: int
    iter_numbers: list
    iter_rsquared: list
    n_wshed_modes: int


def _fit_once(soph_win, x_win, y_win, B0, LB, UB, opts):
    """One bounded nonlinear fit of the whole mode stack. Returns the raw
    dict from the Rust kernel."""
    if _rs is None:
        raise ImportError(
            "dynamo_rs is required for parametric basis fitting; build it with "
            "`maturin develop --release --features python -m "
            "../DYNAM-O_rs/rust/Cargo.toml`"
        )
    # fit_rotGauss.m:111 — 5th percentile of the *windowed* histogram, using
    # MATLAB's prctile convention (see matlab_compat.prctile).
    bg_initial = np.array([0.0, 0.0, float(prctile(soph_win, 5.0))])
    bg_lower = np.array([-_BG_SLOPE_LIM, -_BG_SLOPE_LIM, 0.0])
    bg_upper = np.array([_BG_SLOPE_LIM, _BG_SLOPE_LIM, float(np.max(soph_win))])

    if opts.kind == "power":
        return _rs.fit_rotgauss(
            np.ascontiguousarray(soph_win), x_win, y_win,
            np.ascontiguousarray(B0), np.ascontiguousarray(LB),
            np.ascontiguousarray(UB),
            bg_initial, bg_lower, bg_upper, opts.max_iters,
        )
    return _rs.fit_vmgauss(
        np.ascontiguousarray(soph_win), x_win, y_win,
        np.ascontiguousarray(B0), np.ascontiguousarray(LB),
        np.ascontiguousarray(UB),
        bg_initial, bg_lower, bg_upper, opts.max_iters, True,
    )


def _fit_background_only(soph_win, x_win, y_win, kind):
    """Bounded plane / sinusoid-only fit, for when no mode survives.

    MATLAB reaches this by calling fit_rotGauss with an empty B0
    (param_basis_power.m:511). The model is linear in its three coefficients,
    so a bounded linear least squares gives the same answer as the general
    solver without needing a mode stack.
    """
    from scipy.optimize import lsq_linear

    X, Y = np.meshgrid(np.asarray(x_win, float), np.asarray(y_win, float))
    if kind == "power":
        A = np.column_stack([X.ravel(), Y.ravel(), np.ones(X.size)])
    else:
        # xxx*sin(x + yyy) + zzz is nonlinear in yyy; hold the phase offset at
        # 0 so the remaining problem stays linear. MATLAB's solver would tune
        # yyy, but this path only runs when no mode fit at all, where the
        # baseline shape is not meaningful.
        A = np.column_stack([np.sin(X).ravel(), np.zeros(X.size), np.ones(X.size)])
    b = soph_win.ravel()
    lo = np.array([-_BG_SLOPE_LIM, -_BG_SLOPE_LIM, 0.0])
    hi = np.array([_BG_SLOPE_LIM, _BG_SLOPE_LIM, float(np.max(soph_win))])
    sol = lsq_linear(A, b, bounds=(lo, hi))
    return sol.x


def _gof_from(res_dict):
    return {k: res_dict[k] for k in
            ("sse", "rsquare", "adjrsquare", "rmse", "dfe", "dfm")}


def fit_param_basis_axis(soph, x_bins, y_bins, opts: ParamBasisOpts,
                         seed_modes=None, wshed_img=None) -> ParamFitResult:
    """Fit one axis. `soph` is (n_freq, n_feature) = (len(y_bins), len(x_bins)).

    `seed_modes` is the (N, 6) watershed seed stack; pass None to fit with a
    single synthetic seed (the MATLAB watershed-failure path).
    """
    soph = np.asarray(soph, dtype=float)
    x_bins = np.asarray(x_bins, dtype=float).ravel()
    y_bins = np.asarray(y_bins, dtype=float).ravel()
    soph = orient_soph(soph, x_bins, y_bins)
    verbose = bool(opts.verbose)

    # --- Valid analysis window (param_basis_power.m:151-156) ---------------
    valid_mat = np.isfinite(soph)
    invalid_freq = ~valid_mat.any(axis=1)
    valid_mat[invalid_freq, :] = True
    valid_x = ((x_bins >= opts.feature_limits[0])
               & (x_bins <= opts.feature_limits[1])
               & valid_mat.all(axis=0))
    valid_y = ((y_bins >= opts.freq_limits[0])
               & (y_bins <= opts.freq_limits[1])
               & ~invalid_freq)
    if not valid_x.any() or not valid_y.any():
        # Nothing fittable — e.g. a recording too short for
        # SOpower_min_time_in_bin, leaving every histogram bin NaN. runDYNAMO
        # guards this with `any(isfinite(SOPHs.SOpower_mat),'all')` and skips
        # the fit, so return an empty result rather than raising.
        if verbose:
            print("No valid bins in the analysis window; skipping the fit.")
        return ParamFitResult(
            params=np.zeros((0, 6)), background=np.zeros(3),
            model_soph=np.full_like(soph, np.nan), gof={}, wshed_img=None,
            fit_iteration=0, iter_numbers=[], iter_rsquared=[],
            n_wshed_modes=0,
        )

    x_win = x_bins[valid_x]
    y_win = y_bins[valid_y]
    soph_win = soph[np.ix_(valid_y, valid_x)]

    # --- Seeds --------------------------------------------------------------
    mode_params = (np.zeros((0, 6)) if seed_modes is None
                   else np.atleast_2d(np.asarray(seed_modes, dtype=float)))
    if mode_params.size == 0:
        mode_params = np.zeros((0, 6))
    amp0 = mode_params[:, 0] if mode_params.shape[0] else soph_win.ravel()

    lb_default, ub_default = resolve_bounds(opts, amp0, x_win, y_win)

    prefix = np.atleast_2d(np.asarray(opts.prefix_modes, dtype=float)) \
        if len(opts.prefix_modes) else np.zeros((0, 6))
    if opts.prefix_modes_order == 1:
        mode_params = np.vstack([prefix, mode_params])
    elif opts.prefix_modes_order == -1:
        mode_params = np.vstack([mode_params, prefix])
    else:
        mode_params = prefix

    n_wshed_modes = mode_params.shape[0]
    if n_wshed_modes < 1:
        # Synthetic single seed at the window center (param_basis_power.m:304).
        amp_seed = float(np.nanmax(soph_win))
        if opts.wshed_exp:
            amp_seed = float(np.log(amp_seed))
        mode_params = np.array([[
            amp_seed,
            (y_win.max() + y_win.min()) / 2.0,
            (y_win.max() - y_win.min()) / 4.0,
            (x_win.max() + x_win.min()) / 2.0,
            (x_win.max() - x_win.min()) / 4.0,
            0.0,
        ]])
        n_wshed_modes = 1

    max_peaks = opts.max_peaks if opts.max_peaks != -1 else n_wshed_modes

    # --- Iterative fitting --------------------------------------------------
    good_models, good_r2, good_nums = [], [], []
    B0i = np.zeros((0, 6))
    LBi = np.zeros((0, 6))
    UBi = np.zeros((0, 6))
    last_B0i = last_LBi = last_UBi = None
    model_soph = np.zeros_like(soph)
    last_ii = 0

    for ii in range(1, max_peaks + 1):
        last_ii = ii
        if ii <= n_wshed_modes:
            B0i = np.vstack([B0i, mode_params[ii - 1]])
        else:
            seed_row, found = residual_max_seed(
                soph_win, model_soph[np.ix_(valid_y, valid_x)],
                x_win, y_win, B0i, opts.min_freq_diff,
            )
            B0i = np.vstack([B0i, seed_row if found else B0i.mean(axis=0)])

        LBi = np.vstack([LBi, lb_default])
        UBi = np.vstack([UBi, ub_default])

        res = _fit_once(soph_win, x_win, y_win, B0i, LBi, UBi, opts)
        params = np.atleast_2d(res["params"])
        background = np.asarray(res["background"], dtype=float)
        model_soph = eval_modes(params, x_bins, y_bins, kind=opts.kind,
                                background=background,
                                unit_row=(opts.kind == "phase"))
        adjr2_i = float(res["adjrsquare"])
        n_modes = params.shape[0]

        if ii > 1:
            diffr2 = adjr2_i - good_r2[-1]
            diffr2_pct = diffr2 / abs(good_r2[-1]) if good_r2[-1] else 0.0
        else:
            diffr2 = diffr2_pct = 0.0

        ol_max = 0.0
        if n_modes > 1:
            ol_max = float(mode_overlap(params, x_bins, y_bins,
                                        kind=opts.kind).max())

        B0i = params

        # --- Revert checks (param_basis_power.m:445-491) --------------------
        revert = False
        if ii > 1:
            if opts.criterion == "mindr2" and diffr2 < opts.min_dr2:
                revert = True
                if verbose:
                    print(f"    R^2 change too small: {diffr2}")
            elif opts.criterion == "minpctr2" and diffr2_pct < opts.min_pctr2:
                revert = True
                if verbose:
                    print(f"    %R^2 change too small: {diffr2_pct}")
        if ol_max > opts.max_overlap:
            revert = True
            if verbose:
                print(f"    Overlap exceeds max: {ol_max}")
        if np.any(B0i[:, 0] < opts.min_amp):
            revert = True
            if verbose:
                print(f"    Peak amplitude below min amp: {B0i[:, 0]}")
        if opts.min_freq_diff > 0 and \
                min_pairwise_freq_diff(B0i) < opts.min_freq_diff:
            revert = True
            if verbose:
                print(f"    Peaks too close in frequency: {B0i[:, 1]}")

        if verbose:
            print(f"**********Iteration {ii}**********")
            print(f"   adjr2: {adjr2_i}  diff: {diffr2}  %diff: {diffr2_pct}"
                  f"  maxol: {ol_max}")

        if revert:
            if ii == 1:
                # No mode helped at all: return background only.
                bg = _fit_background_only(soph_win, x_win, y_win, opts.kind)
                model = eval_modes(np.zeros((0, 6)), x_bins, y_bins,
                                   kind=opts.kind, background=bg)
                return ParamFitResult(
                    params=np.zeros((0, 6)), background=bg, model_soph=model,
                    gof=_gof_from(res), wshed_img=wshed_img, fit_iteration=0,
                    iter_numbers=[], iter_rsquared=[],
                    n_wshed_modes=n_wshed_modes,
                )
            B0i, LBi, UBi = last_B0i, last_LBi, last_UBi
            if ii > n_wshed_modes:
                if verbose:
                    print("Additional modes will not improve fit. Terminating.")
                break
        else:
            good_models.append(res)
            good_r2.append(adjr2_i)
            good_nums.append(ii)
            last_B0i, last_LBi, last_UBi = B0i, LBi, UBi

    if not good_nums:
        bg = _fit_background_only(soph_win, x_win, y_win, opts.kind)
        model = eval_modes(np.zeros((0, 6)), x_bins, y_bins, kind=opts.kind,
                           background=bg)
        return ParamFitResult(
            params=np.zeros((0, 6)), background=bg, model_soph=model,
            gof={}, wshed_img=wshed_img, fit_iteration=0,
            iter_numbers=[], iter_rsquared=[], n_wshed_modes=n_wshed_modes,
        )

    fit_iteration = select_iteration(
        opts.criterion, good_nums, good_r2,
        min_dr2=opts.min_dr2, min_pctr2=opts.min_pctr2,
        kneedle_tol=opts.kneedle_tol, verbose=verbose,
    )
    best = good_models[good_nums.index(fit_iteration)]
    params = np.atleast_2d(best["params"])
    background = np.asarray(best["background"], dtype=float)
    model_soph = eval_modes(params, x_bins, y_bins, kind=opts.kind,
                            background=background,
                            unit_row=(opts.kind == "phase"))
    if verbose:
        print(f"Selected iteration {fit_iteration} of {good_nums}")

    return ParamFitResult(
        params=params, background=background, model_soph=model_soph,
        gof=_gof_from(best), wshed_img=wshed_img,
        fit_iteration=fit_iteration, iter_numbers=good_nums,
        iter_rsquared=good_r2, n_wshed_modes=n_wshed_modes,
    )


def fit_param_basis(soph, x_bins, y_bins, opts: ParamBasisOpts | None = None,
                    kind: str = "power") -> ParamFitResult:
    """Seed from a watershed over the histogram, then fit — fitParamBasis.m.

    `soph` is (n_freq, n_feature). `kind` picks the default option set when
    `opts` is None; an explicit `opts` wins.

    For the phase axis the histogram is Gaussian-smoothed before seeding when
    `opts.gauss_filt_std` is set, matching param_basis_phase.m:172-191. The
    smoothing affects seeding only — the fit always sees the raw histogram.
    """
    from scipy.ndimage import gaussian_filter

    from pydynamo.soph.paramfit.histpeaks import (
        extract_hist_peaks, seeds_from_stats,
    )

    if opts is None:
        opts = (ParamBasisOpts.power() if kind == "power"
                else ParamBasisOpts.phase())

    soph = orient_soph(soph, x_bins, y_bins)
    seed_img = np.exp(soph) if opts.wshed_exp else soph
    if opts.gauss_filt_std is not None and opts.kind == "phase":
        sig = np.asarray(opts.gauss_filt_std, dtype=float)
        seed_img = gaussian_filter(np.nan_to_num(seed_img), sigma=tuple(sig))

    merge_thresh, dur_min, bw_min, height_min, trim_vol = opts.watershed_params
    try:
        stats, wshed_img = extract_hist_peaks(
            seed_img, x_bins, y_bins, merge_thresh, dur_min, bw_min,
            height_min, trim_vol,
        )
    except ValueError:
        stats, wshed_img = None, None

    seeds = None
    if stats is not None and len(stats):
        seeds = seeds_from_stats(stats, opts.freq_limits, opts.min_freq_diff,
                                 wshed_exp=opts.wshed_exp)
        if seeds.shape[0] == 0:
            seeds = None
    if seeds is None and opts.verbose:
        print("Watershed produced no usable seeds; "
              "fitting from a synthetic seed instead.")

    return fit_param_basis_axis(soph, x_bins, y_bins, opts,
                                seed_modes=seeds, wshed_img=wshed_img)
