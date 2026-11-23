"""Write one run's outputs into the canonical DYNAM-O output tree.

:func:`write_run` writes a :class:`pydynamo.DynamoOutput` where the DYNAM-O
app writes the same run (OUTPUT_FORMAT.md §1-2), honoring the same
``settings["output"]`` switches, so a results root can mix pydynamo and
app runs and opens in the app. Figures are not written (the app's summary
and fit figures are its own; pydynamo's summary figure is ``out.fig``).
"""

from __future__ import annotations

from pathlib import Path

from pydynamo.io.aux_h5 import write_auxiliary_data_h5
from pydynamo.io.paramfit import write_paramfit_csv, write_paramfit_skipped_csv
from pydynamo.io.soph_tiff import write_soph_tiff
from pydynamo.io.splinefit_tiff import write_splinefit_tiff
from pydynamo.io.stamp import current_stamp
from pydynamo.io.stats import write_stats_csv
from pydynamo.io.tree import (
    aux_h5_path, paramfit_csv_path, soph_tiff_path, splinefit_tiff_path, stats_csv_path,
)


def write_run(out, root, subject_id: str, channel: str) -> dict:
    """Write ``out`` (a `DynamoOutput`) for ``subject_id`` / ``channel``
    under ``root``. ``channel`` is the output channel's name (the settings
    file's channel label, or the channel itself). Returns the paths
    written, by kind."""
    root = Path(root)
    output = out.settings["output"]
    stamp = current_stamp()
    sophs = out.SOPHs
    written = {}

    if output.get("write_csv", True):
        path = stats_csv_path(root, subject_id, channel)
        write_stats_csv(
            out.stats_table, path, stamp, subject_id, fs=out.fs, settings=out.settings,
        )
        written["stats"] = path

    if output.get("write_sophs", True):
        for axis, hist, bins in (
            ("power", sophs.SOpower_mat, sophs.SOpower_bins),
            ("phase", sophs.SOphase_mat, sophs.SOphase_bins),
        ):
            path = soph_tiff_path(root, subject_id, channel, axis)
            write_soph_tiff(path, hist, bins, sophs.freq_bins, f"so{axis}", subject_id, stamp)
            written[f"soph_{axis}"] = path

    if output.get("write_aux", True):
        path = aux_h5_path(root, subject_id, channel)
        sopower = out.settings["sopower"]
        write_auxiliary_data_h5(
            path,
            fs=out.fs,
            so_power_norm=out.SOpower_norm_native,
            so_power_t_start=out.SOpower_native_t_start,
            so_power_norm_method=sopower["norm_method"],
            so_power_window_params=tuple(sopower["window_params"]),
            so_power_freqrange=tuple(sopower["band"]),
            artifact_spans=out.artifact_spans,
            stage_times=out.stage_times,
            stage_vals=out.stage_vals,
            baseline=out.baseline,
            baseline_freqs=out.baseline_freqs,
            subject_id=subject_id,
            stamp=stamp,
        )
        written["aux"] = path

    if output.get("write_paramfits", True):
        for axis, fit, bins in (
            ("power", sophs.SOpower_paramfit, sophs.SOpower_bins),
            ("phase", sophs.SOphase_paramfit, sophs.SOphase_bins),
        ):
            path = paramfit_csv_path(root, subject_id, channel, axis)
            reason = out.skipped.get(f"paramfit_{axis}")
            if fit is not None:
                write_paramfit_csv(path, fit, axis, bins, sophs.freq_bins, subject_id, stamp)
            elif reason is not None:
                write_paramfit_skipped_csv(
                    path, axis, bins, sophs.freq_bins, reason, subject_id, stamp,
                )
            else:
                continue
            written[f"paramfit_{axis}"] = path

    if output.get("write_splinefits", True) and output.get("write_tiff", True):
        for axis, fit in (("power", sophs.SOpower_splinefit), ("phase", sophs.SOphase_splinefit)):
            if fit is None:
                continue
            path = splinefit_tiff_path(root, subject_id, channel, axis)
            write_splinefit_tiff(path, fit, axis, subject_id, stamp)
            written[f"splinefit_{axis}"] = path
    return written
