#!/usr/bin/env python3
"""Compare genomic bins using row means and coverage-weighted methylation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


KEYS = ["chrom", "start"]
COUNTS = ["methylated", "unmethylated"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input1", type=Path, required=True,
        help="First CG coverage file, plotted on the x-axis.",
    )
    parser.add_argument(
        "--input2", type=Path, required=True,
        help="Second CG coverage file, plotted on the y-axis.",
    )
    parser.add_argument(
        "--x-label", default="Input 1 methylation (%)",
        help="Full x-axis title for input1.",
    )
    parser.add_argument(
        "--y-label", default="Input 2 methylation (%)",
        help="Full y-axis title for input2.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bin-size", type=int, default=10_000_000)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.bin_size < 1 or args.chunksize < 1:
        parser.error("bin-size and chunksize must be positive")
    return args


def aggregate_cov(
    path: Path, bin_size: int, chunksize: int,
) -> tuple[pd.DataFrame, int]:
    """Read 1-based cov sites; return 0-based half-open nominal bins."""
    totals = None
    row_count = 0
    columns = ["chrom", "position", "end", "row_rate", *COUNTS]
    reader = pd.read_csv(
        path, sep="\t", header=None, usecols=range(6),
        names=columns, chunksize=chunksize,
        dtype={
            "chrom": str, "row_rate": "float64",
            **{name: "int64" for name in ["position", "end", *COUNTS]},
        },
    )
    for chunk in reader:
        if (
            chunk.isna().any().any()
            or chunk["position"].lt(1).any()
            or chunk["position"].ne(chunk["end"]).any()
            or chunk[COUNTS].lt(0).any().any()
            or not chunk["row_rate"].between(0, 100).all()
        ):
            raise ValueError(f"Invalid site coordinates, rates or counts: {path}")
        chunk["start"] = (chunk["position"] - 1) // bin_size * bin_size
        grouped = chunk.groupby(KEYS, sort=False).agg(
            methylated=("methylated", "sum"),
            unmethylated=("unmethylated", "sum"),
            rate_sum=("row_rate", "sum"),
            row_count=("row_rate", "size"),
        )
        if totals is None:
            totals = grouped
        else:
            totals = pd.concat([totals, grouped]).groupby(
                level=KEYS, sort=False,
            ).sum()
        row_count += len(chunk)
        if row_count % (chunksize * 20) == 0:
            print(f"{path}: {row_count:,} rows", flush=True)
    if totals is None or totals.empty:
        raise ValueError(f"No cov records: {path}")
    totals["coverage"] = totals[COUNTS].sum(axis=1)
    totals["methylation_percent"] = (
        totals["rate_sum"] / totals["row_count"]
    )
    return totals, row_count


def draw_scatter(
    pairs: pd.DataFrame, output: Path, bin_size: int, rate: str,
    method: str, limits: tuple[float, float], x_label: str, y_label: str,
) -> dict:
    values = pairs[[f"{rate}_input1", f"{rate}_input2"]].dropna()
    x, y = values.iloc[:, 0], values.iloc[:, 1]
    pearson = None
    if len(values) > 1 and x.nunique() > 1 and y.nunique() > 1:
        pearson = float(x.corr(y))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans",
        ],
        "font.size": 5, "axes.titlesize": 7, "axes.labelsize": 6,
        "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 5,
    })
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.scatter(x, y, s=5, c="#4DBBD5", alpha=0.55, linewidths=0)
    ax.plot(limits, limits, "--", color="0.4", linewidth=0.6)
    ax.set(xlim=limits, ylim=limits)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, labelpad=5)
    ax.set_title(f"CG: {bin_size / 1e6:g} Mb bins\n{method}", pad=3)
    correlation = "NA" if pearson is None else f"{pearson:.3f}"
    ax.text(
        0.03, 0.97, f"n = {len(values):,}\nPearson r = {correlation}",
        transform=ax.transAxes, va="top", fontsize=5,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {
        "method": method,
        "paired_bins": len(values), "excluded_bins": len(pairs) - len(values),
        "pearson_r": pearson,
        "rmse_pp": float(np.sqrt(((y - x) ** 2).mean()))
        if len(values) else None,
    }


def write_outputs(
    pairs: pd.DataFrame, output_dir: Path, bin_size: int,
    x_label: str, y_label: str,
) -> None:
    """Derive both rates from pooled sums, using common plot limits."""
    methods = {
        "row_mean": ("methylation_percent", "Row mean"),
        "coverage_weighted": ("weighted_percent", "Coverage weighted"),
    }
    for label in ["input1", "input2"]:
        pairs[f"weighted_percent_{label}"] = (
            100 * pairs[f"methylated_{label}"]
            / pairs[f"coverage_{label}"].where(
                pairs[f"coverage_{label}"].gt(0)
            )
        )
    plotted = pd.concat([
        pairs[[f"{rate}_input1", f"{rate}_input2"]].dropna().stack()
        for rate, _ in methods.values()
    ], ignore_index=True)
    limits = (0, 100)
    if len(plotted):
        low, high = plotted.min(), plotted.max()
        padding = max((high - low) * 0.05, 0.1)
        limits = (max(0, low - padding), min(100, high + padding))
    pairs.to_csv(output_dir / "per_bin_comparison.tsv", sep="\t",
                 index=False, na_rep="NA")
    statistics = [
        draw_scatter(
            pairs, output_dir / f"CG_input1_vs_input2_{key}.png",
            bin_size,
            rate, title, limits, x_label, y_label,
        )
        for key, (rate, title) in methods.items()
    ]
    print(json.dumps(statistics, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    # A fresh directory protects results from accidental replacement.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    input1_bins, input1_rows = aggregate_cov(
        args.input1, args.bin_size, args.chunksize,
    )
    print(f"Input 1: {input1_rows:,} sites, {len(input1_bins):,} bins",
          flush=True)
    input2_bins, input2_rows = aggregate_cov(
        args.input2, args.bin_size, args.chunksize,
    )
    print(
        f"Input 2: {input2_rows:,} sites, "
        f"{len(input2_bins):,} bins", flush=True,
    )
    pairs = input1_bins.add_suffix("_input1").join(
        input2_bins.add_suffix("_input2"), how="outer",
    ).reset_index().sort_values(KEYS)
    pairs.insert(2, "end", pairs["start"] + args.bin_size)
    write_outputs(
        pairs, args.output_dir, args.bin_size, args.x_label, args.y_label,
    )


if __name__ == "__main__":
    main()
