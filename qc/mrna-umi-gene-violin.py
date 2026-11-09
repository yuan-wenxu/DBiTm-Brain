#!/usr/bin/env python3
"""Plot UMI and detected-gene violins from mRNA H5ADs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


QC_COLOR = "#4DBBD5"
SAMPLE_COLORS = {
    "P7-M1-S3-mRNA": "#FFE082",
    "P35-M1-S3-mRNA": "#E64B35",
    "P60-M1-S3-mRNA": "#4DBBD5",
    "P60-M3-S2-mRNA": "#CC79A7",
}
OUTPUT_NAME = "mrna-umi-gene-violin.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", nargs=2, action="append", required=True,
        metavar=("SAMPLE", "FILE"),
        help="Sample name and mRNA H5AD. Repeat for multiple samples.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for the combined QC PNG.",
    )
    return parser.parse_args()


def sample_sort_key(name: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", name)
    )


def read_count_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"Input H5AD does not exist: {path}")
    source = ad.read_h5ad(path, backed="r")
    try:
        if source.n_obs == 0 or source.n_vars == 0:
            raise ValueError(f"Input H5AD is empty: {path}")
        if not source.obs_names.is_unique or not source.var_names.is_unique:
            raise ValueError(f"Spot and gene IDs must be unique: {path}")
        if "sample" in source.obs and source.obs["sample"].nunique() > 1:
            raise ValueError(
                f"Each input must contain one unintegrated sample: {path}"
            )
        if "counts" in source.layers:
            matrix = source.layers["counts"]
        else:
            matrix = source.X[:]
        counts = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
        counts.sum_duplicates()
        if not np.isfinite(counts.data).all() or (counts.data < 0).any():
            raise ValueError(f"Counts must be finite and nonnegative: {path}")
        if not np.equal(counts.data, np.floor(counts.data)).all():
            raise ValueError(
                f"Expected integer raw counts in layers['counts'] or X: {path}"
            )
        counts.eliminate_zeros()
        umis = np.asarray(counts.sum(axis=1)).ravel()
        genes = np.diff(counts.indptr).astype(np.float64)
    finally:
        source.file.close()
    return umis, genes


def median_label(value: float) -> str:
    return f"{value:,.0f}" if value.is_integer() else f"{value:,.1f}"


def plot_violins(
    metrics: dict[str, tuple[np.ndarray, np.ndarray]], output: Path,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans",
        ],
        "axes.titlesize": 7, "axes.labelsize": 6,
        "xtick.labelsize": 5, "ytick.labelsize": 5,
        "legend.fontsize": 5,
    })
    names = sorted(metrics, key=sample_sort_key)
    labels = names
    rotate_labels = len(names) > 8 or any(len(label) > 12 for label in labels)
    figure, axes = plt.subplots(2, 1, figsize=(4, 2), sharex=True)
    try:
        for metric, (axis, ylabel) in enumerate(zip(
            axes, ("Number of UMIs", "Number of genes"), strict=True,
        )):
            maximum = 0.0
            for position, sample in enumerate(names):
                color = SAMPLE_COLORS.get(sample, QC_COLOR)
                values = metrics[sample][metric]
                maximum = max(maximum, float(values.max()))
                if len(values) > 1 and np.ptp(values) > 0:
                    violin = axis.violinplot(
                        values, positions=[position], widths=0.5,
                        showextrema=False, points=200,
                    )
                    for body in violin["bodies"]:
                        body.set_facecolor(color)
                        body.set_edgecolor("none")
                        body.set_alpha(1)
                else:
                    axis.plot([position - 0.25, position + 0.25],
                              [values[0], values[0]], color=color,
                              linewidth=1)
                median = float(np.median(values))
                axis.hlines(
                    median, position - 0.125, position + 0.125,
                    colors="#222222", linewidths=0.7, zorder=3,
                )
                axis.annotate(
                    median_label(median), (position, median),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=5,
                    bbox={
                        "facecolor": "white", "edgecolor": "none",
                        "alpha": 0.85, "pad": 0.5,
                    },
                    zorder=4,
                )
            locator = MaxNLocator(nbins=4)
            upper = locator.tick_values(0, maximum * 1.1 or 1.0)[-1]
            axis.set_ylim(0, upper)
            axis.yaxis.set_major_locator(locator)
            axis.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:,.0f}")
            )
            axis.set_ylabel(ylabel, labelpad=5)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_linewidth(0.6)
            axis.tick_params(width=0.6, length=2, pad=2)
            axis.set_xlim(-0.6, len(names) - 0.4)
        axes[0].tick_params(axis="x", bottom=False, labelbottom=False)
        axes[1].set_xticks(range(len(names)), labels,
                          rotation=45 if rotate_labels else 0,
                          ha="right" if rotate_labels else "center")
        figure.subplots_adjust(
            left=0.18, right=0.98,
            bottom=0.16 if rotate_labels else 0.09, top=0.98, hspace=0.12,
        )
        figure.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.02)
    finally:
        plt.close(figure)


def run(args: argparse.Namespace) -> None:
    samples = [
        (name, Path(path).expanduser().resolve()) for name, path in args.input
    ]
    names = [name for name, _ in samples]
    if any(not name.strip() for name in names):
        raise ValueError("Sample names must not be blank")
    if len(set(names)) != len(names):
        raise ValueError("Sample names must be unique")
    output_dir = args.output_dir.expanduser().resolve()
    output = output_dir / OUTPUT_NAME
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    metrics = {}
    for name, path in samples:
        umis, genes = read_count_metrics(path)
        metrics[name] = (umis, genes)
        print(f"{name}: {len(umis):,} spots; "
              f"median UMI={median_label(float(np.median(umis)))}; "
              f"median genes={median_label(float(np.median(genes)))}",
              flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_violins(metrics, output)
    print(f"Saved {output}", flush=True)


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
