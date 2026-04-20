# pydynamo

Minimal Python port of [DYNAM-O](https://github.com/preraulab/DYNAM-O) exposing TF-peak extraction, SO-power / SO-phase histograms, and a summary figure. No GUI, no parametric/spline fits, no statistical tests.

Companion Rust crate `dynamo_rs` (in `rust/`) provides the region-merging inner loop (`mergeWshedSegment` port). Multitaper spectrogram delegates to the existing `multitaper_rs` crate.

## Install (development)

```bash
cd ../pydynamo
# Build and install the Rust merge extension
maturin develop --release -m rust/Cargo.toml
# Install the Python package
pip install -e .
# Make sure multitaper_rs is installed too
pip install -e ../DYNAM-O/toolbox/helper_functions/multitaper_toolbox/rust
# or: (cd ../DYNAM-O/toolbox/helper_functions/multitaper_toolbox/rust && maturin develop --release)
```

## Usage

```python
import scipy.io as sio
from pydynamo import run_dynamo

m = sio.loadmat("../DYNAMO_dev/example_data/example_data.mat",
                simplify_cells=True)
out = run_dynamo(m["data"].ravel(), float(m["Fs"]), m["stage_times"], m["stage_vals"])
print(out.stats_table.head())
print(out.SOPHs.SOpower_mat.shape)   # (101, 151)
```

## Stage convention

DYNAM-O uses `1=N3, 2=N2, 3=N1, 4=REM, 5=Wake` — reversed from most EDF stagers. Pass stages in this convention or histograms will be empty or inverted.

## Equivalence

Designed to be bit-identical to the MATLAB reference (`runDYNAMO('segment')`) for:

- multitaper spectrogram
- artifact mask
- baseline-subtracted spectrogram
- SO-power and SO-phase timeseries
- SO-power and SO-phase histograms, **given the same `stats_table`**

The TF-peak extraction (watershed + merge + trim) is not bit-identical — `skimage.segmentation.watershed` uses different tie-breaking than MATLAB's `watershed`, and floating-point non-determinism in the merge loop gives small peak-count differences. See `tests/test_peaks.py` for the looser equivalence bar there.

## Validation data

- `../DYNAMO_dev/example_data/example_data.mat` — bundled example EEG (data, Fs, stage_times, stage_vals)
- `../DYNAMO_dev/segment_out.mat` — saved MATLAB `runDYNAMO('segment')` output
- `data_cache/segment_stats.csv` — MATLAB `stats_table` exported via `scripts/export_matlab_ground_truth.m` (must be generated in MATLAB first)
