#!/usr/bin/env python3
"""Compare TAPS and TAPS-beta methylation rates and CG site counts."""

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

SAMPLE_COLORS = {
    "P35-M1-S1-TAPS": "#E64B35",
    "P35-M2-S2-beta": "#E64B35",
    "P60-M1-S1-TAPS": "#4DBBD5",
    "P60-M2-S2-beta": "#4DBBD5",
    "P60-M2-S11-TAPS": "#CC79A7",
    "P60-M2-S12-beta": "#CC79A7",
}
METHOD_HATCHES = {"TAPS": "", "TAPS-beta": "////"}


CONTEXT_COLUMNS = {
    "CG": ("mean_methylation", "cpg_site_count"),
    "CA": ("mean_ca_methylation", "ca_site_count"),
    "CC": ("mean_cc_methylation", "cc_site_count"),
    "CT": ("mean_ct_methylation", "ct_site_count"),
}
RATE_FIELDS = ["sample", "context", "mean_methylation", "site_count"]
SUMMARY_FIELDS = [
    "sample",
    "context",
    "total_spot_count",
    "tissue_spot_count",
    "plotted_spot_count",
    "median_methylation",
]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-spot CG, CA, CC, and CT methylation rates and CG site "
            "counts from multiple per_spot_summary.tsv files, with optional "
            "in-tissue filtering."
        )
    )
    parser.add_argument(
        "--summary",
        action="append",
        nargs=2,
        required=True,
        metavar=("SAMPLE", "FILE"),
        help=(
            "Sample name and input summary TSV. Repeat for multiple samples. "
            "Files may have any name and directory; SAMPLE must have an "
            "entry in SAMPLE_COLORS."
        ),
    )
    parser.add_argument(
        "--positions",
        action="append",
        nargs=2,
        default=[],
        metavar=("SAMPLE", "FILE"),
        help=(
            "Optional tissue_positions.tsv[.gz] for a SAMPLE supplied with "
            "--summary. Repeat for multiple samples. Match summary "
            "row_index/col_index to array_row/array_col and retain only "
            "in_tissue=1. "
            "Spots absent from the positions file are excluded."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for the methylation-rate and CG-count PNGs and TSV.",
    )
    return parser.parse_args(argv)


def tissue_mask(frame: pd.DataFrame, path: Path) -> pd.Series:
    """Match tissue flags by numeric array coordinates, not barcode names."""
    positions = pd.read_csv(path, sep="\t", dtype=str)
    required = {"array_row", "array_col", "in_tissue"}
    missing = required - set(positions.columns)
    if missing:
        raise ValueError(
            f"missing position columns in {path}: {', '.join(sorted(missing))}"
        )
    flags = pd.to_numeric(positions["in_tissue"], errors="coerce")
    if not flags.isin((0, 1)).all():
        raise ValueError(f"in_tissue must contain only 0 or 1: {path}")

    coordinate_keys: list[pd.MultiIndex] = []
    for table, columns in (
        (frame, ["row_index", "col_index"]),
        (positions, ["array_row", "array_col"]),
    ):
        coordinates = table[columns].apply(pd.to_numeric, errors="coerce")
        values = coordinates.to_numpy(dtype=np.float64)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or not np.equal(values, np.floor(values)).all()
        ):
            raise ValueError(
                "summary/position coordinates must be finite non-negative "
                f"integers when matching {path}"
            )
        keys = pd.MultiIndex.from_frame(coordinates)
        if keys.has_duplicates:
            raise ValueError(
                f"duplicate summary/position coordinates when matching {path}"
            )
        coordinate_keys.append(keys)

    summary_keys, position_keys = coordinate_keys
    tissue_keys = position_keys[flags.eq(1).to_numpy()]
    return pd.Series(summary_keys.isin(tissue_keys), index=frame.index)


def read_sample(
    path: Path,
    sample: str,
    positions_path: Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, float | int | str]], int]:
    if not path.is_file():
        raise FileNotFoundError(f"summary file not found: {path}")

    input_columns = ["spot"]
    for rate_column, count_column in CONTEXT_COLUMNS.values():
        input_columns.extend((rate_column, count_column))
    if positions_path is not None:
        input_columns.extend(("row_index", "col_index"))
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

    in_tissue = pd.Series(True, index=frame.index)
    if positions_path is not None:
        in_tissue = tissue_mask(frame, positions_path)
        if not in_tissue.any():
            raise ValueError(
                f"no summary spots match in-tissue positions: {positions_path}"
            )
    rate_columns = [columns[0] for columns in CONTEXT_COLUMNS.values()]
    complete_contexts = frame[rate_columns].notna().all(axis=1)
    retained = in_tissue & complete_contexts
    retained_spot_count = int(retained.sum())
    if retained_spot_count < 2:
        raise ValueError(
            "summary has fewer than two complete methylation-rate spots after "
            f"filtering: {path}"
        )

    rate_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, float | int | str]] = []
    for context, (rate_column, count_column) in CONTEXT_COLUMNS.items():
        rates = frame.loc[retained, rate_column].to_numpy(dtype=np.float64)
        site_counts = frame.loc[retained, count_column].to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(rates).all() or ((rates < 0) | (rates > 100)).any():
            raise ValueError(
                f"{context} methylation rates must be finite percentages in [0, 100]: "
                f"{path}"
            )
        if (
            not np.isfinite(site_counts).all()
            or (site_counts <= 0).any()
            or not np.equal(site_counts, np.floor(site_counts)).all()
        ):
            raise ValueError(
                f"{context} site counts must be finite positive integers "
                f"for retained spots: {path}"
            )

        rate_frames.append(
            pd.DataFrame(
                {
                    "sample": sample,
                    "context": context,
                    "mean_methylation": rates,
                    "site_count": site_counts,
                },
                columns=RATE_FIELDS,
            )
        )

        summaries.append(
            {
                "sample": sample,
                "context": context,
                "total_spot_count": len(frame),
                "tissue_spot_count": int(in_tissue.sum()),
                "plotted_spot_count": retained_spot_count,
                "median_methylation": float(np.median(rates)),
            }
        )
    return (
        pd.concat(rate_frames, ignore_index=True),
        summaries,
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
            pad_inches=0.02,
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
        body.set_hatch(METHOD_HATCHES["TAPS-beta" if sample.endswith("-beta") else "TAPS"])

    for boundary in (1.5, 2.5, 3.5):
        axis.axvline(boundary, color="#D0D0D0", linewidth=0.7, zorder=0)
    axis.set_xlim(0.52, 4.48)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"
            ],
            "axes.titlesize": 7,
            "axes.titlepad": 3,
            "axes.labelsize": 6,
            "axes.labelpad": 5,
        }
    )


def draw_sample_legend(axis: plt.Axes, sample_names: list[str]) -> None:
    handles = [
        Patch(
            facecolor=SAMPLE_COLORS[sample],
            edgecolor="#333333",
            linewidth=0.5,
            alpha=0.92,
            hatch=METHOD_HATCHES[
                "TAPS-beta" if sample.endswith("-beta") else "TAPS"
            ],
            label=sample,
        )
        for sample in sample_names
    ]
    axis.axis("off")
    axis.legend(
        handles=handles,
        title="Sample (left-to-right order)",
        loc="center",
        ncol=1,
        frameon=True,
        fontsize=5,
        title_fontsize=5,
        handlelength=1.5,
        labelspacing=0.35,
    )


def write_plot(rates: pd.DataFrame, sample_names: list[str], output_dir: Path) -> None:
    configure_plot_style()
    distributions, positions, body_samples = grouped_violin_data(
        rates, sample_names
    )
    violin_width = min(0.18, 0.68 / len(sample_names))
    figure = plt.figure(figsize=(4, 2), layout="tight")
    figure.get_layout_engine().set(pad=0.3, h_pad=0.2, w_pad=0.3)
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
    upper_axis.tick_params(axis="y", labelsize=5)
    lower_axis.tick_params(axis="both", labelsize=5)
    lower_axis.set_xticks(
        np.arange(1, len(CONTEXT_COLUMNS) + 1),
        labels=list(CONTEXT_COLUMNS),
    )
    figure.supylabel("Methylation rate per spot (%)", fontsize=6)

    break_style = {
        "marker": [(-1, -0.5), (1, 0.5)],
        "markersize": 4,
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

    draw_sample_legend(legend_axis, sample_names)
    figure.suptitle("Methylation rate", fontsize=7)
    save_figure_atomic(
        figure, output_dir / "combined_methylation_rate_violin.png"
    )
    plt.close(figure)


def write_cg_site_count_plot(
    rates: pd.DataFrame, sample_names: list[str], output_dir: Path
) -> None:
    configure_plot_style()
    cg_counts = rates.loc[rates["context"].eq("CG")]
    distributions = [
        cg_counts.loc[cg_counts["sample"].eq(sample), "site_count"].to_numpy()
        for sample in sample_names
    ]
    figure = plt.figure(figsize=(4, 2), layout="tight")
    figure.get_layout_engine().set(pad=0.3, w_pad=0.3)
    grid = figure.add_gridspec(1, 2, width_ratios=(3, 1.4))
    axis = figure.add_subplot(grid[0, 0])
    legend_axis = figure.add_subplot(grid[0, 1])
    positions = np.arange(1, len(sample_names) + 1)
    violins = axis.violinplot(
        distributions,
        positions=positions,
        widths=0.7,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        bw_method="scott",
    )
    for body, sample in zip(violins["bodies"], sample_names, strict=True):
        body.set_facecolor(SAMPLE_COLORS[sample])
        body.set_edgecolor("#333333")
        body.set_linewidth(0.4)
        body.set_alpha(0.92)
        body.set_hatch(
            METHOD_HATCHES[
                "TAPS-beta" if sample.endswith("-beta") else "TAPS"
            ]
        )
    violins["cmedians"].set_color("#222222")
    violins["cmedians"].set_linewidth(0.7)
    violins["cmedians"].set_zorder(3)
    for position, distribution in zip(positions, distributions, strict=True):
        median = float(np.median(distribution))
        axis.annotate(
            f"{median:,.1f}".removesuffix(".0"),
            xy=(position, median),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5,
            bbox={
                "facecolor": "white", "edgecolor": "none",
                "alpha": 0.85, "pad": 0.5,
            },
            zorder=4,
        )
    axis.set_xlim(0.5, len(sample_names) + 0.5)
    axis.set_ylim(bottom=0)
    axis.set_xticks([])
    axis.set_xlabel("CG")
    axis.set_ylabel("CG sites per spot")
    axis.tick_params(axis="y", labelsize=5)
    axis.ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0), useMathText=True
    )
    axis.yaxis.get_offset_text().set_fontsize(5)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    draw_sample_legend(legend_axis, sample_names)
    figure.suptitle("CG site count", fontsize=7)
    save_figure_atomic(figure, output_dir / "combined_cg_site_count_violin.png")
    plt.close(figure)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        positions_by_sample: dict[str, Path] = {}
        input_samples = {sample for sample, _ in args.summary}
        for sample, positions_path in args.positions:
            if sample not in input_samples:
                raise ValueError(f"--positions sample has no summary: {sample}")
            if sample in positions_by_sample:
                raise ValueError(f"duplicate --positions sample: {sample}")
            positions_by_sample[sample] = Path(positions_path)
        sample_names: list[str] = []
        rate_frames: list[pd.DataFrame] = []
        summaries: list[dict[str, float | int | str]] = []
        for sample, summary_path in args.summary:
            if sample in sample_names:
                raise ValueError(f"duplicate sample name: {sample}")
            if sample not in SAMPLE_COLORS:
                raise ValueError(f"SAMPLE_COLORS lacks an entry for: {sample}")
            rates, sample_summaries, plotted_spots = read_sample(
                Path(summary_path), sample, positions_by_sample.get(sample)
            )
            sample_names.append(sample)
            rate_frames.append(rates)
            summaries.extend(sample_summaries)
            cg_summary = sample_summaries[0]
            print(
                f"sample={sample} "
                f"tissue-spots={cg_summary['tissue_spot_count']} "
                f"plotted-spots={plotted_spots}"
            )

        combined_rates = pd.concat(rate_frames, ignore_index=True)
        combined_summary = pd.DataFrame(summaries, columns=SUMMARY_FIELDS)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_table(
            combined_summary,
            args.output_dir / "methylation_rate_summary.tsv",
        )
        write_plot(combined_rates, sample_names, args.output_dir)
        write_cg_site_count_plot(combined_rates, sample_names, args.output_dir)
        print(f"[methylation] output-dir={args.output_dir}")
        return 0
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"[methylation] error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
