#!/usr/bin/env python3
"""Compare coverage-filtered per-spot CG and CH methylation rates."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from saturation import SAMPLE_COLORS


CONTEXT_COLUMNS = {
    "CG": ("mean_methylation", "cpg_site_count"),
    "CA": ("mean_ca_methylation", "ca_site_count"),
    "CC": ("mean_cc_methylation", "cc_site_count"),
    "CT": ("mean_ct_methylation", "ct_site_count"),
}
RATE_FIELDS = ["sample", "context", "mean_methylation"]
SUMMARY_FIELDS = [
    "sample",
    "context",
    "total_spot_count",
    "reads_threshold",
    "coverage_passing_spot_count",
    "plotted_spot_count",
    "median_methylation",
]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer sample-specific read thresholds and plot filtered per-spot "
            "CG, CA, CC, and CT methylation-rate distributions from multiple "
            "per_spot_summary.tsv files."
        )
    )
    parser.add_argument(
        "summary_files",
        nargs="+",
        type=Path,
        help=(
            "Input per_spot_summary.tsv files. The sample name is inferred "
            "from <sample>/summary/per_spot_summary.tsv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for the combined PNG and summary TSV.",
    )
    return parser.parse_args(argv)


def infer_sample_name(path: Path) -> str:
    if path.name != "per_spot_summary.tsv" or path.parent.name != "summary":
        raise ValueError(
            "expected a path ending in <sample>/summary/per_spot_summary.tsv: "
            f"{path}"
        )
    sample_name = path.parent.parent.name
    if not sample_name:
        raise ValueError(f"could not infer sample name from: {path}")
    return sample_name


def infer_reads_threshold(reads: np.ndarray) -> int:
    """Infer a high-coverage cutoff using Otsu separation on positive log reads."""
    positive_reads = reads[reads > 0]
    if positive_reads.size < 2 or np.unique(positive_reads).size < 2:
        raise ValueError(
            "cannot infer a coverage threshold from fewer than two positive "
            "read levels"
        )

    log_reads = np.log10(positive_reads)
    bin_count = min(256, max(32, int(np.sqrt(positive_reads.size))))
    counts, edges = np.histogram(log_reads, bins=bin_count)
    centers = (edges[:-1] + edges[1:]) / 2
    background_weight = np.cumsum(counts)[:-1]
    foreground_weight = counts.sum() - background_weight
    cumulative_moment = np.cumsum(counts * centers)[:-1]
    background_mean = np.divide(
        cumulative_moment,
        background_weight,
        out=np.zeros_like(cumulative_moment),
        where=background_weight > 0,
    )
    foreground_mean = np.divide(
        (counts * centers).sum() - cumulative_moment,
        foreground_weight,
        out=np.zeros_like(cumulative_moment),
        where=foreground_weight > 0,
    )
    between_class_variance = (
        background_weight
        * foreground_weight
        * np.square(background_mean - foreground_mean)
    )
    valid = (background_weight > 0) & (foreground_weight > 0)
    if not valid.any():
        raise ValueError("could not separate low- and high-coverage spots")
    between_class_variance[~valid] = -1
    split_index = int(np.argmax(between_class_variance))
    return max(1, int(np.ceil(10 ** edges[split_index + 1])))


def read_sample(
    path: Path,
) -> tuple[pd.DataFrame, list[dict[str, float | int | str]], int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"summary file not found: {path}")

    sample = infer_sample_name(path)
    input_columns = ["spot", "reads"]
    for rate_column, count_column in CONTEXT_COLUMNS.values():
        input_columns.extend((rate_column, count_column))
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=input_columns,
        dtype={"spot": "string"},
    )
    if frame.empty:
        raise ValueError(f"empty summary file: {path}")
    if frame["spot"].isna().any() or frame["spot"].str.strip().eq("").any():
        raise ValueError(f"summary contains missing or blank spot names: {path}")
    if frame["spot"].duplicated().any():
        raise ValueError(f"summary contains duplicate spot names: {path}")

    for column in input_columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame["reads"].isna().any():
        raise ValueError(f"summary contains missing or nonnumeric read counts: {path}")
    read_values = frame["reads"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(read_values).all()
        or (read_values < 0).any()
        or not np.equal(read_values, np.floor(read_values)).all()
    ):
        raise ValueError(f"read counts must be finite non-negative integers: {path}")

    reads_threshold = infer_reads_threshold(read_values)
    coverage_passing = frame["reads"].ge(reads_threshold)

    rate_columns = [columns[0] for columns in CONTEXT_COLUMNS.values()]
    complete_contexts = frame[rate_columns].notna().all(axis=1)
    retained = coverage_passing & complete_contexts
    retained_spot_count = int(retained.sum())
    if retained_spot_count < 2:
        raise ValueError(
            "summary has fewer than two complete methylation-rate spots after coverage "
            f"filtering: {path}"
        )

    rate_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, float | int | str]] = []
    for context, (rate_column, count_column) in CONTEXT_COLUMNS.items():
        rates = frame.loc[retained, rate_column].to_numpy(dtype=np.float64)
        site_counts = frame.loc[retained, count_column]
        if not np.isfinite(rates).all() or ((rates < 0) | (rates > 100)).any():
            raise ValueError(
                f"{context} methylation rates must be finite percentages in [0, 100]: "
                f"{path}"
            )
        if site_counts.isna().any() or site_counts.le(0).any():
            raise ValueError(
                f"summary contains a missing or non-positive {context} site count "
                f"with a rate: {path}"
            )

        rate_frames.append(
            pd.DataFrame(
                {
                    "sample": sample,
                    "context": context,
                    "mean_methylation": rates,
                },
                columns=RATE_FIELDS,
            )
        )

        summaries.append(
            {
                "sample": sample,
                "context": context,
                "total_spot_count": len(frame),
                "reads_threshold": reads_threshold,
                "coverage_passing_spot_count": int(coverage_passing.sum()),
                "plotted_spot_count": retained_spot_count,
                "median_methylation": float(np.median(rates)),
            }
        )
    return (
        pd.concat(rate_frames, ignore_index=True),
        summaries,
        int(coverage_passing.sum()),
        retained_spot_count,
    )


def atomic_write_table(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        frame.to_csv(
            temporary,
            sep="\t",
            index=False,
            float_format="%.6f",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_figure_atomic(figure: plt.Figure, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        figure.savefig(
            temporary,
            format="png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def grouped_violin_data(
    rates: pd.DataFrame, sample_names: list[str]
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    distributions: list[np.ndarray] = []
    positions: list[float] = []
    body_samples: list[str] = []
    offsets = np.linspace(-0.36, 0.36, len(sample_names))
    for context_index, context in enumerate(CONTEXT_COLUMNS, start=1):
        context_rates = rates.loc[rates["context"] == context]
        for offset, sample in zip(offsets, sample_names, strict=True):
            distributions.append(
                context_rates.loc[
                    context_rates["sample"] == sample, "mean_methylation"
                ].to_numpy()
            )
            positions.append(context_index + offset)
            body_samples.append(sample)
    return distributions, np.asarray(positions), body_samples


def draw_grouped_violins(
    axis: plt.Axes,
    distributions: list[np.ndarray],
    positions: np.ndarray,
    body_samples: list[str],
    violin_width: float,
) -> None:
    violins = axis.violinplot(
        distributions,
        positions=positions,
        widths=violin_width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method="scott",
    )
    for body, sample in zip(violins["bodies"], body_samples, strict=True):
        body.set_facecolor(SAMPLE_COLORS[sample])
        body.set_edgecolor("#333333")
        body.set_linewidth(0.4)
        body.set_alpha(0.92)

    for boundary in (1.5, 2.5, 3.5):
        axis.axvline(boundary, color="#D0D0D0", linewidth=0.7, zorder=0)
    axis.set_xlim(0.52, 4.48)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)


def write_plot(rates: pd.DataFrame, sample_names: list[str], output_dir: Path) -> None:
    distributions, positions, body_samples = grouped_violin_data(
        rates, sample_names
    )
    violin_width = min(0.052, 0.68 / len(sample_names))
    figure = plt.figure(figsize=(8, 3.5), layout="tight")
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(2.1, 1),
        width_ratios=(3, 1.4),
    )
    upper_axis = figure.add_subplot(grid[0, 0])
    lower_axis = figure.add_subplot(grid[1, 0], sharex=upper_axis)
    legend_axis = figure.add_subplot(grid[:, 1])
    for axis in (upper_axis, lower_axis):
        draw_grouped_violins(
            axis,
            distributions,
            positions,
            body_samples,
            violin_width,
        )

    cg_minimum = rates.loc[rates["context"] == "CG", "mean_methylation"].min()
    ch_maximum = rates.loc[
        rates["context"].isin(("CA", "CC", "CT")), "mean_methylation"
    ].max()
    lower_upper_limit = max(1.0, np.ceil(ch_maximum * 1.05 * 2) / 2)
    upper_lower_limit = max(
        lower_upper_limit + 1,
        np.floor(cg_minimum * 0.8 / 5) * 5,
    )
    if upper_lower_limit >= cg_minimum:
        upper_lower_limit = (lower_upper_limit + cg_minimum) / 2

    lower_axis.set_ylim(0, lower_upper_limit)
    upper_axis.set_ylim(upper_lower_limit, 100)
    upper_axis.spines["bottom"].set_visible(False)
    lower_axis.spines["top"].set_visible(False)
    upper_axis.tick_params(
        axis="x", which="both", bottom=False, labelbottom=False
    )
    upper_axis.tick_params(axis="y", labelsize=10)
    lower_axis.tick_params(axis="both", labelsize=10)
    lower_axis.set_xticks(
        np.arange(1, len(CONTEXT_COLUMNS) + 1),
        labels=list(CONTEXT_COLUMNS),
    )
    figure.supylabel("Mean methylation per spot (%)", fontsize=10)

    break_style = {
        "marker": [(-1, -0.5), (1, 0.5)],
        "markersize": 8,
        "linestyle": "none",
        "color": "#333333",
        "markeredgewidth": 0.8,
        "clip_on": False,
    }
    upper_axis.plot(
        (0, 1),
        (0, 0),
        transform=upper_axis.transAxes,
        **break_style,
    )
    lower_axis.plot(
        (0, 1),
        (1, 1),
        transform=lower_axis.transAxes,
        **break_style,
    )

    legend_handles = [
        Patch(
            facecolor=SAMPLE_COLORS[sample],
            edgecolor="#333333",
            linewidth=0.5,
            label=sample,
        )
        for sample in sample_names
    ]
    legend_axis.axis("off")
    legend_axis.legend(
        handles=legend_handles,
        title="Sample (left-to-right order)",
        loc="center",
        ncol=1,
        frameon=True,
        fontsize=8,
        title_fontsize=8,
        handlelength=1.5,
        labelspacing=0.35,
    )
    figure.suptitle("Methylation distributions", fontsize=12)
    save_figure_atomic(
        figure, output_dir / "combined_methylation_rate_violin.png"
    )
    plt.close(figure)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        sample_names: list[str] = []
        rate_frames: list[pd.DataFrame] = []
        summaries: list[dict[str, float | int | str]] = []
        for path in args.summary_files:
            sample = infer_sample_name(path)
            if sample in sample_names:
                raise ValueError(f"duplicate sample name: {sample}")
            if sample not in SAMPLE_COLORS:
                raise ValueError(f"SAMPLE_COLORS lacks an entry for: {sample}")
            rates, sample_summaries, coverage_spots, plotted_spots = read_sample(path)
            sample_names.append(sample)
            rate_frames.append(rates)
            summaries.extend(sample_summaries)
            cg_summary = sample_summaries[0]
            print(
                f"sample={sample} reads-threshold={cg_summary['reads_threshold']} "
                f"coverage-passing={coverage_spots} plotted-spots={plotted_spots}"
            )

        combined_rates = pd.concat(rate_frames, ignore_index=True)
        combined_summary = pd.DataFrame(summaries, columns=SUMMARY_FIELDS)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_table(
            combined_summary,
            args.output_dir / "methylation_rate_summary.tsv",
        )
        write_plot(combined_rates, sample_names, args.output_dir)
        print(f"[methylation] output-dir={args.output_dir}")
        return 0
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"[methylation] error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
