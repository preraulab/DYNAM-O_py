"""Full pipeline smoke test on the bundled DYNAM-O segment data.

Runs run_dynamo on the `runDYNAMO('segment')` slice, checks outputs exist
and are plausibly shaped, writes the summary figure as PNG so it can be
eyeballed against the MATLAB output.
"""

from pathlib import Path

import numpy as np
import pytest

from pydynamo import SOPHOpts, run_dynamo
from pydynamo.soph.histogram import create_bins


SEGMENT_TIME_RANGE = (8420, 13446)


def _expected_soph_shape(opts: SOPHOpts, which: str) -> tuple[int, int]:
    """(n_c_bins, n_freq_bins) implied by `opts`, using MATLAB's bin methods.

    TFPeakHistogram.m:135-141 uses 'partial' for the frequency axis, 'extend'
    for a circular C axis (SO-phase) and 'partial' otherwise (SO-power).
    """
    _, freq_cbins = create_bins(opts.freq_range, *opts.freq_binsizestep, "partial")
    if which == "phase":
        _, c_cbins = create_bins(opts.SOphase_range, *opts.SOphase_binsizestep,
                                 "extend")
    else:
        _, c_cbins = create_bins(opts.SOpower_range, *opts.SOpower_binsizestep,
                                 "partial")
    return c_cbins.size, freq_cbins.size


def test_end_to_end_segment(example_data, tmp_path):
    fs = float(example_data["Fs"])
    data = np.asarray(example_data["data"]).ravel()
    stage_times = np.asarray(example_data["stage_times"]).ravel()
    stage_vals = np.asarray(example_data["stage_vals"]).ravel()

    out = run_dynamo(
        data, fs, stage_times, stage_vals,
        time_range=SEGMENT_TIME_RANGE,
        merge_thresh=0.0,
        plot=True,
        verbose=False,
    )

    # Outputs. Shapes are derived from the default SOPHOpts rather than
    # hardcoded, so a deliberate change to a MATLAB-matched default updates the
    # expectation with it instead of failing here.
    opts = SOPHOpts()
    pow_shape = _expected_soph_shape(opts, "power")
    pha_shape = _expected_soph_shape(opts, "phase")

    assert out.stats_table is not None
    assert out.SOPHs.SOpower_mat.shape == pow_shape, \
        f"unexpected SOpower_mat shape {out.SOPHs.SOpower_mat.shape}, want {pow_shape}"
    assert out.SOPHs.SOphase_mat.shape == pha_shape, \
        f"unexpected SOphase_mat shape {out.SOPHs.SOphase_mat.shape}, want {pha_shape}"
    assert out.SOPHs.SOpower_bins.size == pow_shape[0]
    assert out.SOPHs.SOphase_bins.size == pha_shape[0]
    assert out.SOPHs.freq_bins.size == pow_shape[1]

    # The frequency axis is the SOPH histogram range (2-18 Hz by default), not
    # the spectrogram's 0-30 Hz range. Bin centers accumulate step-sized
    # floating-point error, so compare with a tolerance well under one step.
    tol = 0.01 * opts.freq_binsizestep[1]
    assert out.SOPHs.freq_bins.min() >= opts.freq_range[0] - tol
    assert out.SOPHs.freq_bins.max() <= opts.freq_range[1] + tol

    # Peak count — MATLAB has 5738 in this segment; ours will differ but
    # should be in a reasonable range.
    print(f"\npeak count: {len(out.stats_table)}  (MATLAB: ~5738)")
    assert 500 < len(out.stats_table) < 20000, \
        f"peak count {len(out.stats_table)} out of ballpark"

    # Columns
    required = {"PeakTime", "PeakFrequency", "PeakStage", "SOpower",
                "SOphase", "Duration", "Bandwidth", "Volume"}
    assert required.issubset(set(out.stats_table.columns))

    # Save the figure so a human can visually compare to MATLAB's output.
    png_path = tmp_path / "pydynamo_summary.png"
    out.fig.savefig(png_path, dpi=120)
    print(f"summary figure written to {png_path}")
    assert png_path.exists()
