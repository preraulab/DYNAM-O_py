# DYNAM-O_py

DYNAM-O from Python: TF-peak extraction, SO-power and SO-phase histograms
(SOPHs), and their parametric and spline fits, for sleep EEG. The repository
is named **DYNAM-O_py** and the Python package is named `pydynamo`.

pydynamo is a wrapper over the shared Rust kernel
[`dynamo_rs`](https://github.com/preraulab/DYNAM-O_rs), the same code the
DYNAM-O desktop app and `dynamo-cli` run. The whole run (resampling,
artifact detection, both multitaper passes, baseline, TF-peak extraction
and refinement, per-peak stage / SO-power / SO-phase, the SOPHs and both
fit families) is one kernel call, and pydynamo has no Python
reimplementation of any stage. So given the same recording and settings,
pydynamo and the app produce the same numbers by construction, and
pydynamo writes the same files.

## Siblings

- **[DYNAM-O](https://github.com/preraulab/DYNAM-O)**: the MATLAB implementation, with the File Manager GUI and the statistical testing suite.
- **[DYNAM-O_rs](https://github.com/preraulab/DYNAM-O_rs)**: the shared Rust kernel (`dynamo_rs`). MATLAB (through MEX), the desktop app and pydynamo (through PyO3) all call it. Its tests check it against MATLAB.
- **[DYNAM-O_DesktopApp](https://github.com/preraulab/DYNAM-O_DesktopApp)**: the desktop app and `dynamo-cli`. Its `documents/` folder holds the settings reference and the output-format spec.
- **[DYNAM-O_toolbox](https://github.com/preraulab/DYNAM-O_toolbox)**: the bootstrap repo that clones, aligns and builds the others.

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
  artifacts. MATLAB is not required to run pydynamo after installation.

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
numpy ≥ 1.24, scipy ≥ 1.11, matplotlib ≥ 3.7, pandas ≥ 2.0, dynamo_rs ≥ 0.2.2
(built by the bootstrap, not fetched from PyPI), tifffile ≥ 2023.7.10,
h5py ≥ 3.9.

### Verify the install

The bootstrap runs `scripts/check_install.py`, which checks that pydynamo
imports and that the installed `dynamo_rs` has every binding pydynamo
calls. To re-check by hand from the activated venv (in `DYNAM-O_py/`):

```bash
python scripts/check_install.py
```

It prints the pydynamo version, the kernel build
(`<semver>+<sha12>[.dirty]`) and the kernel's semantics version. Every file
pydynamo writes carries the same three values. An older kernel build that
lacks a binding is refused with a rebuild hint, so rerun the bootstrap.

## Usage

### One recording

```python
import pydynamo

settings = pydynamo.load_settings("batch_settings.json")   # or None for the defaults
rec = pydynamo.load_recording("night.edf", "C3-A2", "night_staging.csv")
out = pydynamo.run_dynamo(rec.data, rec.fs, rec.stage_times, rec.stage_vals,
                          settings=settings)

out.fig.savefig("summary.png", dpi=200)       # the summary figure
print(out.stats_table.head())                 # one row per TF-peak
print(out.SOPHs.SOpower_mat.shape)            # (SO-power bins, freq bins)
print(out.SOPHs.SOpower_paramfit.params_table)
pydynamo.write_run(out, "results", "night", "C3-A2")
```

`load_recording` loads one channel the way the app does: an EDF label or
expression (`C3`, `C3-A2`, `mean(C3, C4)`, `$[C3-A2 - B]$` for a literal
label that contains operators), named `references`, and the staging file's
layout (`time_col`, `stage_col`, `header_lines`, `delimiter`,
`stage_labels`). Staging clock times are aligned to the EDF start, and
without a staging file every epoch is N2. With a whole settings file,
`load_from_settings_file(document, edf, staging)` takes all of those from
the file.

`run_dynamo` arguments:

- `settings`: the `settings` section of a DYNAM-O settings file, as a dict.
  Keys it leaves out take the kernel's defaults, so a partial section such
  as `{"hist": {"soph_stages": [1]}}` works. The desktop app's
  `documents/SETTINGS_REFERENCE.md` lists every key.
- `time_range=(start, end)` in seconds limits the run to part of the night.
- `fit_param_basis=False` / `fit_spline_basis=False` skip those fits.
- `keep_spectrogram=True` also returns the pass-2 extraction spectrogram as
  `out.pass2_spectrogram` (over a gigabyte for a full night).
- `plot=False` skips the figure, `verbose=False` the progress lines.

The run resamples to `settings["resample_target_fs"]` (100 Hz by default,
as in the app), so `out.fs` is the rate the run used and `out.t` its
sample times. A fit the kernel declines on a degenerate SOPH is `None`,
with the reason in `out.skipped` and as a warning.

### Settings files

`load_settings(path_or_dict)` accepts a whole settings file
(`batch_settings.json`, a `dynamo-cli` manifest) or a bare section. A file
written under an older semantics version is migrated with the same steps as
`dynamo-cli migrate-settings`, and a warning lists them.
`default_settings()` returns the full default section, and
`merge_settings(base, overrides)` layers one section onto another key by key.

### A cohort

```python
import json
import pydynamo

document = json.load(open("batch_settings.json"))
settings = pydynamo.load_settings(document)
channel = document["channel_labels"][0]
for edf, staging, subject in zip(document["data_files"],
                                 document["staging_files"],
                                 document["subject_ids"]):
    rec = pydynamo.load_from_settings_file(document, edf, staging)
    out = pydynamo.run_dynamo(rec.data, rec.fs, rec.stage_times, rec.stage_vals,
                              settings=settings, plot=False)
    pydynamo.write_run(out, "results", subject, channel)
```

### Fits on SOPHs you already have

`fit_paramfits` and `fit_splinefits` run the kernel's fits on any pair of
SOPHs (`(n_feature, n_freq)` matrices and their bin centers), with the
`paramfit` or `splinefit` options of a settings section.
`fit_paramfits` also takes the TF-peak table and the baseline, which
the per-mode `Pk*` columns need. Each returns `(power, phase, skipped)`.

## The output tree

`write_run(out, root, subject, channel)` writes the stats CSV, both SOPH
TIFFs, the auxiliary-data HDF5, both paramfit CSVs and both splinefit TIFFs
where the app writes them, and honors the same `settings["output"]`
switches. A results root can mix pydynamo and app runs and opens in the
app. Figures are not written into the tree, since `out.fig` is pydynamo's
own summary figure.

The readers and writers are in `pydynamo.io`. The normative spec is
`documents/OUTPUT_FORMAT.md` in the DYNAM-O_DesktopApp repo: §1-2 for
the layout and file names, §8 for the provenance stamp every file carries
(`format`, `writer`, `writer_version`, `kernel_version`, `semantics`). The
stats CSV is written by the kernel's own writer, with the app's precision
rules. Readers accept the older formats and return the recovered
provenance with the data:

```python
import pydynamo.io as pio

df, prov = pio.read_stats_csv(pio.stats_csv_path("results", "night", "C3-A2"))
```

## Stage convention

DYNAM-O uses `1=N3, 2=N2, 3=N1, 4=REM, 5=Wake`, reversed from most stagers.
`load_recording` returns DYNAM-O codes. Stages you pass in yourself must
use this convention, or the SOPHs come out empty or inverted.

## Statistics

`pydynamo.stats` wraps the kernel's FDR and permutation tests
(`fdr_1d`, `fdr_2d`, `gpermtest`, `gpermtest2`), checked against
MATLAB fixtures in `tests/fixtures/`.

## Troubleshooting

- **`KernelUnavailable`**: `dynamo_rs` is missing or older than this
  pydynamo. Rerun the DYNAM-O_toolbox bootstrap.
- **`ValueError` from `run_dynamo`**: the kernel rejected the input or the
  settings (for example `stage_times` and `stage_vals` of different
  lengths, or a settings value of the wrong type). The message names the
  problem.
- **Empty or inverted SOPHs**: almost always the stage convention (see
  above).

## Tests

```bash
pytest tests/
```

The tests check that `run_dynamo` returns exactly what a direct
`dynamo_rs.run_dynamo` call returns on a seeded synthetic recording, that
settings reach the kernel, and the loaders, the settings helpers and every
writer and reader. The numbers themselves are tested in DYNAM-O_rs,
against MATLAB.
