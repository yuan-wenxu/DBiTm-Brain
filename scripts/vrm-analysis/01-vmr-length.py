#!/usr/bin/env python3
"""Plot VMR length distributions from MethSCAn matrix column names."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import re
import statistics
from pathlib import Path
from typing import TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_MATRIX_NAME = "methylation_fractions.csv.gz"
CONTEXT_ORDER = ("CA", "CC", "CG", "CT")
VMR_PATTERN = re.compile(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read VMR coordinates from MethSCAn matrix headers and write a "
            "length-distribution figure plus detailed and summary TSV files."
        )
    )
    parser.add_argument(
        "--methscan_dir",
        type=Path,
        help="MethSCAn output root containing CONTEXT/matrix directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory (default: METHSCAN_DIR/vmr_length_distribution)."
        ),
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        help="Contexts to plot (default: auto-detect, ordered as CA CC CG CT).",
    )
    parser.add_argument(
        "--matrix-name",
        default=DEFAULT_MATRIX_NAME,
        help=f"Matrix filename used for its header (default: {DEFAULT_MATRIX_NAME}).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of shared histogram bins (default: 100).",
    )
    parser.add_argument(
        "--x-scale",
        choices=("log", "linear"),
        default="linear",
        help="Histogram x-axis scale (default: linear).",
    )
    parser.add_argument(
        "--max-percentile",
        type=float,
        default=98.0,
        help=(
            "Percentile used as the histogram maximum; longer VMRs are "
            "clipped into the final bin (default: 98)."
        ),
    )
    args = parser.parse_args()

    if args.bins < 1:
        parser.error("--bins must be at least 1")
    if not 0 < args.max_percentile <= 100:
        parser.error("--max-percentile must be greater than 0 and at most 100")
    return args


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open(mode="r", encoding="utf-8", newline="")


def matrix_path(methscan_dir: Path, context: str, matrix_name: str) -> Path:
    return methscan_dir / context / "matrix" / matrix_name


def discover_contexts(methscan_dir: Path, matrix_name: str) -> list[str]:
    contexts = [
        path.name
        for path in methscan_dir.iterdir()
        if path.is_dir() and matrix_path(methscan_dir, path.name, matrix_name).is_file()
    ]
    order = {context: index for index, context in enumerate(CONTEXT_ORDER)}
    return sorted(contexts, key=lambda context: (order.get(context, len(order)), context))


def read_vmrs(path: Path, context: str) -> list[dict[str, str | int]]:
    with open_text(path) as handle:
        try:
            header = next(csv.reader(handle))
        except StopIteration as error:
            raise ValueError(f"Matrix is empty: {path}") from error

    if len(header) < 2:
        raise ValueError(f"Matrix header has no VMR columns: {path}")

    vmrs: list[dict[str, str | int]] = []
    seen: set[str] = set()
    for column_number, vmr in enumerate(header[1:], start=2):
        match = VMR_PATTERN.fullmatch(vmr)
        if match is None:
            raise ValueError(
                f"Invalid VMR coordinate in column {column_number} of {path}: {vmr!r}"
            )

        start = int(match.group("start"))
        end = int(match.group("end"))
        if end <= start:
            raise ValueError(
                f"VMR end must be greater than start in {path}: {vmr!r}"
            )
        if vmr in seen:
            raise ValueError(f"Duplicate VMR coordinate in {path}: {vmr!r}")
        seen.add(vmr)

        vmrs.append(
            {
                "context": context,
                "vmr": vmr,
                "chrom": match.group("chrom"),
                "start": start,
                "end": end,
                "length_bp": end - start,
            }
        )
    return vmrs


def percentile(sorted_values: list[int], probability: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return sorted_values[lower] + fraction * (
        sorted_values[upper] - sorted_values[lower]
    )


def summarize(
    context: str,
    vmrs: list[dict[str, str | int]],
    max_percentile: float = 98.0,
) -> dict[str, str | int | float]:
    lengths = sorted(int(vmr["length_bp"]) for vmr in vmrs)
    return {
        "context": context,
        "vmr_count": len(lengths),
        "min_bp": lengths[0],
        "q1_bp": round(percentile(lengths, 0.25), 1),
        "median_bp": statistics.median(lengths),
        "mean_bp": round(statistics.fmean(lengths), 1),
        "q3_bp": round(percentile(lengths, 0.75), 1),
        "cutoff_bp": round(percentile(lengths, max_percentile / 100.0), 1),
        "max_bp": lengths[-1],
    }


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open(mode="w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def shared_bin_edges(
    all_lengths: list[int],
    bin_count: int,
    x_scale: str,
    max_percentile: float,
) -> list[float]:
    sorted_lengths = sorted(all_lengths)
    minimum = sorted_lengths[0]
    maximum = percentile(sorted_lengths, max_percentile / 100.0)
    if minimum == maximum:
        return [minimum - 0.5, maximum + 0.5]
    if x_scale == "log":
        ratio = (maximum / minimum) ** (1 / bin_count)
        return [minimum * ratio**index for index in range(bin_count + 1)]
    width = (maximum - minimum) / bin_count
    return [minimum + index * width for index in range(bin_count + 1)]


def plot_distributions(
    path: Path,
    vmrs_by_context: dict[str, list[dict[str, str | int]]],
    bin_count: int,
    x_scale: str,
    max_percentile: float,
) -> None:
    contexts = list(vmrs_by_context)
    column_count = min(2, len(contexts))
    row_count = math.ceil(len(contexts) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(column_count * 5, row_count * 4),
        squeeze=False,
        sharex=True,
    )

    all_lengths = [
        int(vmr["length_bp"])
        for vmrs in vmrs_by_context.values()
        for vmr in vmrs
    ]
    bins = shared_bin_edges(all_lengths, bin_count, x_scale, max_percentile)

    for axis, context in zip(axes.flat, contexts):
        lengths = [int(vmr["length_bp"]) for vmr in vmrs_by_context[context]]
        clipped_lengths = [min(length, bins[-1]) for length in lengths]
        weights = [100.0 / len(clipped_lengths)] * len(clipped_lengths)
        axis.hist(
            clipped_lengths,
            bins=bins,
            weights=weights,
            color="#8491B4",
            edgecolor="white",
            linewidth=0.35,
        )
        axis.set_title(f"{context} (n = {len(lengths):,} VMRs)", fontsize=12)
        axis.set_xscale(x_scale)
        axis.set_ylabel("VMRs per bin (%)", fontsize=10)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)

    for axis in axes.flat[len(contexts) :]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            scale_label = ", log scale" if x_scale == "log" else ""
            axis.set_xlabel(f"VMR length (bp{scale_label}; end - start)", fontsize=10)

    figure.suptitle(
        "MethSCAn VMR length distributions\n"
        f"x-axis capped at the P{max_percentile:g} length; longer VMRs are "
        "grouped in the final bin",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    methscan_dir = args.methscan_dir.expanduser().resolve()
    if not methscan_dir.is_dir():
        raise SystemExit(f"MethSCAn directory does not exist: {methscan_dir}")

    contexts = args.contexts or discover_contexts(methscan_dir, args.matrix_name)
    if not contexts:
        raise SystemExit(
            f"No context matrices named {args.matrix_name!r} found under {methscan_dir}"
        )
    if len(contexts) != len(set(contexts)):
        raise SystemExit("--contexts contains duplicate values")

    vmrs_by_context: dict[str, list[dict[str, str | int]]] = {}
    for context in contexts:
        path = matrix_path(methscan_dir, context, args.matrix_name)
        if not path.is_file():
            raise SystemExit(f"Matrix does not exist: {path}")
        try:
            vmrs_by_context[context] = read_vmrs(path, context)
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            raise SystemExit(str(error)) from error

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else methscan_dir / "vmr_length_distribution"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = [vmr for vmrs in vmrs_by_context.values() for vmr in vmrs]
    summary_rows = [
        summarize(context, vmrs, args.max_percentile)
        for context, vmrs in vmrs_by_context.items()
    ]
    detail_path = output_dir / "vmr_lengths.tsv"
    summary_path = output_dir / "vmr_length_summary.tsv"
    figure_path = output_dir / "vmr_length_distribution.png"

    write_tsv(
        detail_path,
        detail_rows,
        ["context", "vmr", "chrom", "start", "end", "length_bp"],
    )
    write_tsv(
        summary_path,
        summary_rows,
        [
            "context",
            "vmr_count",
            "min_bp",
            "q1_bp",
            "median_bp",
            "mean_bp",
            "q3_bp",
            "cutoff_bp",
            "max_bp",
        ],
    )
    plot_distributions(
        figure_path,
        vmrs_by_context,
        args.bins,
        args.x_scale,
        args.max_percentile,
    )

    print(f"Contexts: {', '.join(contexts)}")
    print(f"VMRs: {sum(len(vmrs) for vmrs in vmrs_by_context.values()):,}")
    print(f"Figure: {figure_path}")
    print(f"Summary: {summary_path}")
    print(f"Details: {detail_path}")


if __name__ == "__main__":
    main()
