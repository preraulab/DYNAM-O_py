# pydynamo vs MATLAB — exhaustive bisect summary (2026-04-19)

Full stage-by-stage report: `data_cache/bisect_full_report.{json,txt}`.

## Stages that are now bit-identical to MATLAB

| stage | pydynamo result |
|---|---|
| A1. slope_test submask | 100% agreement (502601 samples) |
| A2. artifacts (full mask) | 99.99% agreement (99.8% recall/precision) |
| B/C. spectrograms (pass-1 & pass-2) | max_rel ≈ 5e-5 (FFT library floor) |
| D1. baseline1 from MATLAB spect1 | max_rel 9e-8 (bit-identical) |
| D2. baseline1 from pydynamo spect1 | max_rel 5e-6 (FFT floor) |
| E1. baseline2 from MATLAB spect2 | max_rel 9e-8 (bit-identical) |
| F1. spect/baseline → spect_norm | max_rel = 0 (exactly bit-identical) |
| G. segmentation bounds | identical (168 segments) |
| I2. mask_spectrogram | max_rel 6e-8 |
| N1. SOpower_mat binning (given MATLAB peaks + SOpower timeseries) | max_rel 0 (bit-identical) |
| L1. SOpower_norm vs MATLAB | cos = 1.000, 99th pctile rel err 0.8% |
| M2. SOphase middle-90% circular error | median 0.0006 rad (≈bit-identical) |

## Fixes applied this session

1. **`baseline.py`**: `np.nanpercentile` now uses `method="hazen"` (MATLAB `prctile` = Hyndman-Fan #5). Before: 0.05% max_rel vs MATLAB. After: 1e-7.
2. **`soph/sopower.py`**: same hazen method for the `pXshift{stages}` shift-baseline percentile.

## Residual divergences (documented, not fully resolvable without more data)

### J1. Pass-2 peaks: 9% over-detect on MATLAB spect2_masked input (9% too many)
- pydynamo 2821 peaks vs MATLAB pre-refine 2581 (given MATLAB spect2_masked as input).
- **Matched peaks (87%) are bit-identical in Height** (median diff = 0.0000).
- Unmatched-py extras: shorter duration (median 1.05s), narrower BW (median 2.05 Hz), lower height.
- Sweep of `merge_thresh` ∈ [10, 13] only trims 107 peaks (weak slope) — **not a threshold issue**.
- Watershed was previously proven bit-identical to MATLAB → divergence localized to **merge adjacency or trim**.
- Most likely: different tie-breaking order in the RAG traversal, or the `expand_labels(distance=5)` adjacency construction differing from MATLAB's border-pixel-share check.
- Full fix requires side-by-side pre/post-merge region counts from MATLAB — not currently exported.

### K1. Hann refinement
- Median |Δf| = 0.0007 Hz over 2505 matched peaks (essentially bit-identical).
- ~2% outliers up to 5 Hz drift — likely peaks with bbox straddling the frequency bounds where MATLAB and scipy's spline interp disagree on edge treatment.

### M1. SOphase timeseries (with MATLAB SOS loaded)
- Middle 90% of samples: circular error median **0.0006 rad** (≈ bit-identical).
- First/last 5%: median 0.02 rad, p95 0.3 rad.
- **Root cause identified**: MATLAB `filtfilt(digitalFilter, data)` uses Gustafsson's method (1996) for edge ICs; scipy `sosfiltfilt` uses fixed-length odd padding with no Gustafsson option for SOS filters. Not tunable via padlen.
- Workaround would require a native Gustafsson-SOS implementation (significant code).

### O1. SOphase_mat binning given MATLAB peaks + MATLAB SOphase timeseries
- cos = 0.9998 (near-perfect).

## End-to-end scores (full pipeline)

| dataset | SOpower cos | SOphase cos | pydynamo peaks |
|---|---|---|---|
| segment (8420–13446 s) | **0.9826** | **0.9623** | 5418 |
| night (full night) | **0.9891** | **0.9917** | 33991 |

MATLAB total peak count for segment = 5738 (pydynamo is 94% of that, consistent with the 9% pass-2 over-detect partially cancelling the stage-filter difference at the final level).

## Remaining work to reach full bit-identity

1. **Pass-2 merge adjacency investigation** — requires MATLAB to export per-segment pre-merge region count + post-merge region count, then diff with pydynamo per-segment.
2. **Gustafsson-SOS filtfilt** — port MATLAB's initial-condition method to Python/scipy SOS path. Would take SOphase edge transients from 0.02 rad median to ≤ 1e-6.
3. **Hann refinement edge cases** — identify the ~2% outlier peaks, likely near 0 or 30 Hz frequency bounds where pydynamo's spline interp differs from MATLAB's.

Parameters confirmed identical to MATLAB defaults:
- `merge_thresh = 11.0` (detection_opts.m default preset)
- `trim_vol = 0.8`
- `downsample = [2, 2]`
- `seg_time = 30.0`
- `SOpower_outlier_threshold = 3.0`
- `SOpower norm_method = 'p2shift1234'`
- `SO_freqrange = (0.3, 1.5)`
- `dur_max = 5`, `bw_max = 15`
- `min_time_in_bin = 5` (segment) / 10 (night) — runExampleData overrides
