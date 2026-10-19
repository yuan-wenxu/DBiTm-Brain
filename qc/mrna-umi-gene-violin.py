#!/usr/bin/env python3
"""Plot UMI and detected-gene violins from mRNA H5ADs."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


QC_COLOR = "#4DBBD5"
AGE_COLORS = {
    "p4": "#A73030",
    "p7": "#E64B35",
    "p35": "#2F5597",
    "p60": "#4DBBD5",
}
OUTPUT_NAME = "mrna-umi-gene-violin.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", nargs="+", type=Path, required=True,
        help="mRNA H5ADs; sample names come from filenames.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for the combined QC PNG.",
    )
    return parser.parse_args()


def infer_sample_name(path: Path) -> str:
    name = path.name.removesuffix(".h5ad").removesuffix(".qc")
    name = name.replace("-mrna-", "-")
    if not name.strip():
        raise ValueError(f"Cannot infer sample name from {path.name}")
    return name


def sample_sort_key(name: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", name)
    )


def sample_labels(names: list[str]) -> list[str]:
    matches = [re.fullmatch(r"p(\d+)-rep(\d+)", name, re.I) for name in names]
    ages = Counter(match.group(1) for match in matches if match is not None)
    labels = []
    for name, match in zip(names, matches, strict=True):
        if match is None:
            labels.append(name)
        elif ages[match.group(1)] == 1:
            labels.append(f"P{match.group(1)}")
        else:
            labels.append(f"P{match.group(1)}-rep{match.group(2)}")
    return labels


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
    labels = sample_labels(names)
    rotate_labels = len(names) > 8 or any(len(label) > 12 for label in labels)
    figure, axes = plt.subplots(2, 1, figsize=(4, 3), sharex=True)
    try:
        for metric, (axis, ylabel) in enumerate(zip(
            axes, ("Number of UMIs", "Number of genes"), strict=True,
        )):
            maximum = 0.0
            for position, sample in enumerate(names):
                age = re.match(r"(p\d+)(?:[-_.]|$)", sample, re.I)
                color = AGE_COLORS.get(
                    age.group(1).lower() if age else "", QC_COLOR,
                )
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
                axis.plot(
                    position, median, "o", markersize=3,
                    markerfacecolor="white", markeredgecolor="#2F5597",
                    markeredgewidth=0.5, zorder=3,
                )
                axis.annotate(
                    median_label(median), (position, median),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=5,
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
    paths = [path.expanduser().resolve() for path in args.inputs]
    names = [infer_sample_name(path) for path in paths]
    if len(set(names)) != len(names):
        raise ValueError("Input H5AD filenames must identify unique samples")
    output_dir = args.output_dir.expanduser().resolve()
    output = output_dir / OUTPUT_NAME
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    metrics = {}
    for name, path in zip(names, paths, strict=True):
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
