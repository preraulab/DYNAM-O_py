# DYNAM-O_py

Python + Rust port of [DYNAM-O](https://github.com/preraulab/DYNAM-O): TF-peak
extraction (double watershed + merge + trim + Hann refinement), SO-power /
SO-phase histograms, and a MATLAB-style summary figure.
The repository is named **DYNAM-O_py**, while the Python package is named
`pydynamo`.

## Siblings

This repo is one of three coordinated implementations of DYNAM-O:

- **[DYNAM-O](https://github.com/preraulab/DYNAM-O)** — authoritative MATLAB implementation. Source-of-truth algorithm, File Manager GUI, full statistical testing suite.
- **[DYNAM-O_rs](https://github.com/preraulab/DYNAM-O_rs)** — shared pure-Rust kernel (`dynamo_rs`). The hot paths in both MATLAB (via MEX) and Python (via PyO3) delegate here.
- **[DYNAM-O_toolbox](https://github.com/preraulab/DYNAM-O_toolbox)** — parent bootstrap/orchestration repo that clones, aligns, and builds all three implementations.

## Rust acceleration

The `dynamo_rs` crate (lives in the sibling `DYNAM-O_rs` repo) accelerates the hot paths and now covers the full pipeline surface:

Extract / refine hot paths (the main speedup):
- `matlab_watershed` — bit-identical to MATLAB IPT `watershed` (Vincent-Soille + FIFO priority).
- `merge_segment` — port of `mergeWshedSegment` with the symmetric `edgeWeightEqual` rule.
- `trim_regions` — port of `trimWshedRegions`.
- `matlab_paint_labels` — 8-conn paint-in-label-order border filling; `pydynamo/tfpeaks/extract.py` now routes through this when `dynamo_rs` is available, which tightens pydynamo↔MATLAB peak-count parity.
- `mask_spectrogram`, `hann_event_spectra`, `refine_from_spectra`, `tfpeak_histogram`.

Time-series + metadata pipelines (ported this session — mirror pydynamo 1:1):
- `so_power_from_spectrogram` — post-MTS SO-power pipeline (band-integrate → pow2db → stage interp → outlier z-score → percentile-shift normalization → optional upsample). Bit-identical to `pydynamo.soph.sopower.compute_so_power`.
- `so_phase_from_eeg` — sosfiltfilt → hilbert → atan2 → unwrap → exclusion masking → stage interp. Bit-identical to `pydynamo.soph.sophase.compute_so_phase`.
- `detect_artifacts` — two-band (HF + BB) robust-z-score artifact detection. Bit-identical to `pydynamo.artifacts.detect_artifacts(..., slope_test=False)`. (Slope-test branch is a follow-up — needs multitaper.)
- `build_baseline_exclude`, `compute_baseline`, `subtract_baseline` — full baseline subpipeline.
- `hilbert`, `sosfiltfilt`, `movmean`, `unwrap` — raw primitives (pydynamo already picks `movmean` up for the artifact detrend).
- `fit_rotgauss`, `fit_vmgauss`, `fit_tensor_product_spline` — shared
  parametric and spline-basis fit kernels used by the default Python pipeline.
  Set `fit_param_basis=False` or `fit_spline_basis=False` to skip either fit
  family. Each spline result includes a callable `spline_obj` with the fitted
  coefficients, augmented knots, and spline order.

Multitaper spectrogram delegates to the existing
[`multitaper_rs`](https://github.com/preraulab/multitaper_toolbox) crate.

## Accuracy vs MATLAB reference

Validated end-to-end against `runDYNAMO` on the bundled example EEG, using the
real MATLAB pipeline defaults (`detection_opts()`, `baseline_opts()`,
`SOpowerphasehist_opts()`). Per-stage intermediates are compared via
`scripts/bisect/*.py` against ground-truth `.mat` files exported by
`scripts/export_*.m`.

| dataset | pydynamo peaks | MATLAB peaks | diff | **SOpower cos** | **SOphase cos** |
|---|---:|---:|---:|---:|---:|
| segment (~84 min) | 5 785 | 5 738 | **+0.8 %** | **0.9958** | **0.9822** |
| full night (~8.4 h) | 34 911 | 34 788 | **+0.4 %** | **0.9973** | **0.9960** |

Stage-by-stage verification:

| stage | agreement vs MATLAB |
|---|---|
| multitaper spectrogram | max_rel ≈ 5e-5 (FFT-library floor) |
| artifacts | 99.99 % sample agreement |
| baseline (hazen prctile) | max_rel ≈ 1e-7 (bit-identical) |
| spect / baseline division | exact |
| watershed | bit-identical to MATLAB IPT |
| merge (post-merge region counts) | +0.1 % |
| pass-2 mask (interior − 1-px perimeter) | matches `maskSpectrogram` exactly |
| SOpower timeseries | cos = 1.0000 |
| SOphase timeseries (MATLAB-exact SOS filter) | cos ≈ 0.987 (edge transient); mid-90 % bit-identical (0.0006 rad median) |
| SOpower / SOphase histogram binning | bit-identical given same stats table |

## Runtime (full night, 8.4 h recording, macOS x86_64)

| stage | pydynamo | MATLAB | speedup |
|---|---:|---:|---:|
| **total end-to-end** | **101 s** | **146 s** | **1.45×** |
| extract pass-1 | 20 s | 84 s | 4.2× |
| extract pass-2 | 21 s | 32 s | 1.5× |

## Install from scratch

The supported native build path is the
[`DYNAM-O_toolbox`](https://github.com/preraulab/DYNAM-O_toolbox) bootstrap.
It synchronizes the coordinated repositories, builds the Rust CLI, MATLAB MEX
files, and Python extensions with source-path remapping, and rejects native
artifacts that contain build-machine paths.

### Prerequisites

- **Git**
- **Python ≥ 3.9**
  On Apple Silicon, use a native arm64 interpreter; an x86_64 Python produces
  x86_64 extensions that run under Rosetta. Check with
  `python -c "import platform; print(platform.machine())"`.
- **Rust toolchain** (`cargo`, `rustc`) — install via [rustup](https://rustup.rs/):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source "$HOME/.cargo/env"
  ```
- **MATLAB and a configured MEX C compiler** are required only for the
  controlled native rebuild, because it also rebuilds DYNAM-O's MATLAB MEX
  artifacts. MATLAB is not required to run pydynamo after installation. It is
  otherwise needed only to
  regenerate ground-truth `.mat` intermediates under `data_cache/` (the
  `scripts/export_*.m` files). Accuracy benchmarking against MATLAB uses
  those exported files, which can be produced once and reused.

### Install

The bootstrap installs its pinned Maturin version into
`DYNAM-O_py/.venv`; do not install or invoke Maturin separately to produce
release artifacts.

macOS, Linux, WSL, or Git-Bash:

```bash
git clone https://github.com/preraulab/DYNAM-O_toolbox.git
cd DYNAM-O_toolbox
./bootstrap.sh --yes
source DYNAM-O_py/.venv/bin/activate
```

Windows PowerShell:

```powershell
git clone https://github.com/preraulab/DYNAM-O_toolbox.git
cd DYNAM-O_toolbox
powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1 -Yes
.\DYNAM-O_py\.venv\Scripts\Activate.ps1
```

For optional test dependencies, run
`python -m pip install -e '.[test]'` from `DYNAM-O_py` after activating that
environment.

The controlled workflow currently creates an editable pydynamo installation;
it does not produce a standalone pydynamo or `dynamo_rs` wheel for
distribution. Do not publish wheels produced through direct Maturin, pip, or
other PEP 517 build commands as controlled release artifacts.

Python runtime deps (installed automatically by `pip install -e .`):
numpy ≥ 1.24, scipy ≥ 1.11, scikit-image ≥ 0.22, matplotlib ≥ 3.7, pandas ≥ 2.0,
joblib ≥ 1.3, tqdm ≥ 4.65, colorcet ≥ 3.0, dynamo_rs ≥ 0.2.1 (built by the
bootstrap, not fetched from PyPI), tifffile ≥ 2023.7.10, h5py ≥ 3.9.

### Verify the install

The bootstrap already runs `scripts/check_install.py` (it checks that
`pydynamo`, `dynamo_rs`, and the multitaper Rust backend all import with
the expected surface). To re-check by hand from the activated venv:

```bash
python -c "import pydynamo; print(pydynamo.__version__)"
python -c "from pydynamo.io.stamp import current_stamp; print(current_stamp())"
python scripts/check_install.py     # from DYNAM-O_py/
```

The `current_stamp()` line shows the provenance every written artifact
gets: `writer_version` is this package's build and `kernel_version` is
the `dynamo_rs` build (`<semver>+<sha12>[.dirty]`). A `kernel_version`
of `unknown` means `dynamo_rs` did not import — rerun the bootstrap.

## Usage

### Recommended sampling frequency: 100 Hz

Pydynamo analyzes 0–30 Hz, so 100 Hz Nyquist is well above anything the
pipeline cares about. Resampling higher-rate recordings (128 / 200 /
256 / 500 / 1000 Hz) **down** to 100 Hz before calling `run_dynamo`
gives a ~2× end-to-end speedup with zero analytical change. The
multitaper-spectrogram NFFT is `2^nextpow2(Fs / mtm_dsfreqs)` (default
`mtm_dsfreqs = 0.1`), so anything above **Fs = 102.4 Hz** doubles NFFT
and spills the spectrogram past CPU L3 cache, costing 2–3× more on
every downstream stage. Resample with scipy:

```python
import numpy as np
from scipy.signal import resample_poly

target_fs = 100
if Fs != target_fs:
    from fractions import Fraction
    f = Fraction(target_fs, int(Fs)).limit_denominator(1000)
    data = resample_poly(data, f.numerator, f.denominator)
    Fs = target_fs
```

Zero analytical loss for sleep oscillations (slow oscillations 0.3–1.5 Hz,
spindles 11–16 Hz, alpha 8–12 Hz, beta 13–30 Hz are all far below 50 Hz
Nyquist). Skip the resample only if you specifically need spectral content
above 50 Hz (e.g., gamma analysis beyond pydynamo's analyzed band).

### Basic call

The bundled example EEG ships with the sibling MATLAB checkout: in the
`DYNAM-O_toolbox` layout it is `../DYNAM-O/example_data/example_data.mat`,
next to `runExampleData.m`. The three option values below reproduce that
script's `'segment'` preset (`time_range = [8420 13446]`,
`SOpower_min_time_in_bin = 5`, `SOphase_min_peak_at_freq = 10`); omit
them for the full-night defaults.

```python
import matplotlib.pyplot as plt
import scipy.io as sio
from pydynamo import run_dynamo

m = sio.loadmat("../DYNAM-O/example_data/example_data.mat", simplify_cells=True)
out = run_dynamo(
    m["data"].ravel(),
    float(m["Fs"]),
    m["stage_times"],
    m["stage_vals"],
    # MATLAB runExampleData.m 'segment' preset (omit for full-night defaults)
    time_range=(8420.0, 13446.0),
    min_time_in_bin=5,
    min_peak_at_freq=10,
)
plt.show()                          # out.fig is the MATLAB-style summary figure
print(out.stats_table.head())       # per-peak stats
print(out.SOPHs.SOpower_mat.shape)  # (SOpower_bins, freq_bins)
print(out.SOPHs.SOpower_paramfit.params_table)
print(out.provenance)               # which build computed the numbers
```

`run_dynamo` builds the summary figure by default (`plot=True`) and keeps
it as `out.fig` — `out.fig.savefig("summary.png", dpi=200)` writes it out.
To re-render a figure without re-running the pipeline, `summary_plot` is
importable directly and takes the arrays a run produces:

```python
import numpy as np
from pydynamo import summary_plot

t = np.arange(len(out.artifacts)) / float(m["Fs"]) + 8420.0
fig = summary_plot(
    out.spect, out.stimes, out.sfreqs, out.artifacts, t,
    m["stage_times"], m["stage_vals"], out.stats_table, out.SOPHs,
    time_range=(8420.0, 13446.0),
)
```

Each parametric-fit result keeps the legacy numeric `params` array and also
provides `params_table`, a pandas DataFrame with MATLAB-compatible named
columns such as `Volume`, power-mode `PrefPhase`/`Coupling`, and the per-mode
`Pk*` TF-peak summaries. Zero-mode fits retain the same named columns with no
rows.

### Batch example

Loop a cohort and write the canonical output tree (see "Reading and
writing the DYNAM-O output tree" below) so the results open in the
desktop app, `dynamo-cli`, and the MATLAB Results Browser. EDF inputs
load via `pydynamo.io_edf.read_edf` / `read_staging`:

```python
import pydynamo.io as pio
from pydynamo import run_dynamo
from pydynamo.io_edf import read_edf, read_staging

root, channel = "results", "C3"
for subject in ("S001", "S002"):
    edf = read_edf(f"/data/{subject}.edf", label=channel)
    times, vals = read_staging(f"/data/{subject}_stages.csv")
    out = run_dynamo(edf["data"], edf["fs"], times, vals, plot=False)
    s = out.SOPHs
    pio.write_stats_csv(out.stats_table,
                        pio.stats_csv_path(root, subject, channel),
                        out.provenance, subject_id=subject)
    pio.write_soph_tiff(pio.soph_tiff_path(root, subject, channel, "power"),
                        s.SOpower_mat, s.SOpower_bins, s.freq_bins,
                        "sopower", subject)
    pio.write_soph_tiff(pio.soph_tiff_path(root, subject, channel, "phase"),
                        s.SOphase_mat, s.SOphase_bins, s.freq_bins,
                        "sophase", subject)
    pio.append_run_item(root, subject=subject, channel=channel,
                        input_file=f"/data/{subject}.edf",
                        peaks=len(out.stats_table))
```

The same pattern extends to the paramfit CSVs, splinefit TIFFs, and the
auxiliary-data h5 (`pio.write_paramfit_csv`, `pio.write_splinefit_tiff`,
`pio.write_auxiliary_data_h5`).

## Reading and writing the DYNAM-O output tree

The numeric core is in-memory-only; the `pydynamo.io` package holds the
readers and writers for the canonical on-disk results tree shared by
`dynamo-cli`, the desktop app, and the MATLAB toolbox. The normative
spec is `documents/OUTPUT_FORMAT.md` in the DYNAM-O_DesktopApp repo —
§1-2 for the tree layout and filenames, §8 for the provenance stamp
every artifact carries (`format` / `writer` / `writer_version` /
`kernel_version`).

```python
import pydynamo.io as pio

out = run_dynamo(...)                       # out.provenance is the run's stamp
p = pio.stats_csv_path(root, "S001", "C3")  # <root>/C3/TFpeaks/S001_stats_table_C3.csv
pio.write_stats_csv(out.stats_table, p, out.provenance, subject_id="S001")
df, prov = pio.read_stats_csv(p)            # accepts formats 1/2/3
```

Writers exist for every per-(subject, channel) artifact: the stats CSV,
SOPH and splinefit TIFFs, paramfit CSVs, the auxiliary-data HDF5, and
the `_runs/*.jsonl` run index. Readers accept the legacy formats per
the spec's §8.3 tolerance rules and return the recovered provenance
alongside the data. When the `dynamo_rs` kernel is unavailable and a
pure-Python fallback computes the numbers, stamps report
`kernel_version='python-native'`.

## Stage convention

DYNAM-O uses `1=N3, 2=N2, 3=N1, 4=REM, 5=Wake` — reversed from most EDF
stagers. Pass stages in this convention or histograms will be empty or
inverted.

## Side-by-side figures

`scripts/compare_matlab_vs_pydynamo.py` renders MATLAB's output and
pydynamo's output through the identical `summary_plot` code, so any visual
difference is pure data difference (not color scaling or layout).

![segment comparison](data_cache/compare_segment_sidebyside.png)

![night comparison](data_cache/compare_night_sidebyside.png)

## Validation data

All ground-truth comparison uses `runDYNAMO` output from the DYNAM-O repo's
bundled example EEG (`example_data.mat`). Regenerate MATLAB intermediates
once from the DYNAM-O repo root:

```matlab
run('<path-to-DYNAM-O_py>/scripts/export_bisect_intermediates.m')
run('<path-to-DYNAM-O_py>/scripts/export_merge_diagnostics.m')
run('<path-to-DYNAM-O_py>/scripts/export_pass1_diagnostics.m')
```

These populate `data_cache/` with `.mat` / `.csv` files (not version-controlled).

## Troubleshooting

- **NaNs in `data`** — `run_dynamo` expects a finite EEG vector. NaN
  samples propagate through the multitaper spectrogram and poison the
  baseline percentile. Repair or crop the recording first (NaN gaps from
  hardware dropouts are usually detected by the artifact stage anyway
  once replaced with zeros, but explicit cropping via `time_range` is
  cleaner).
- **`stage_times` / `stage_vals` mismatch** — the two vectors must be the
  same length, in seconds since the start of `data` (not clock time), and
  cover the span you analyze. `run_dynamo` derives its default
  `time_range` from the first and last valid stage, so staging that does
  not overlap the recording produces `No valid stages found` or an empty
  analysis window.
- **Empty or inverted histograms** — almost always the stage convention.
  DYNAM-O uses `1=N3, 2=N2, 3=N1, 4=REM, 5=Wake` (see "Stage
  convention" above), reversed from most EDF stagers; histograms default
  to NREM stages `(1, 2, 3)`. `pydynamo.io_edf.read_staging` already
  returns DYNAM-O codes.
- **`dynamo_rs` missing** — most stages fall back to pure Python /
  scipy / scikit-image. The first fallback fires a single
  `RuntimeWarning` and every artifact written from that process is
  stamped `kernel_version='python-native'` instead of the Rust build id
  (the parametric and spline fits have no Python fallback and raise
  `ImportError`). Rerun the DYNAM-O_toolbox bootstrap to rebuild the
  kernel.

## Tests

```bash
pytest tests/
```

Smoke + unit coverage for spectrogram, artifacts, baseline, SOpower timeseries,
SOpower histogram binning from MATLAB peaks, slope_test, watershed-vs-MATLAB
equivalence, and an end-to-end run.
