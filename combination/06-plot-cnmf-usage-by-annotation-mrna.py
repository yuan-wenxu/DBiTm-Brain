#!/usr/bin/env python3
"""Plot mean cNMF usage by age within each current annotation."""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import PercentFormatter


SAMPLE_ORDER = ("p7-rep1", "p35-rep1", "p60-rep1")
AGE_LABELS = ("P7", "P35", "P60")
FIGURE_SIZE = (7, 3)
OUTPUT_NAME = "program-usage-by-annotation.png"
ANNOTATION_TITLES = {
    "Astrocytes": "Astro",
    "Vascular and Leptomeningeal Cells": "VLMC",
    "Cajal Retzius cells": "CR",
    "Oligodendrocytes": "Oligo",
    "Striatal spiny projection neurons": "Striat.",
    "Ependymal": "Epen.",
    "Unassigned": "Unass.",
}
COLOR_MAP = LinearSegmentedColormap.from_list(
    "usage", ("#4DBBD5", "#FFFFFF", "#E64B35"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="H5AD containing sample and annotation labels.")
    parser.add_argument("--usage", type=Path, required=True,
                        help="Normalized usage CSV indexed by spot_id.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for the PNG heatmap only.")
    return parser.parse_args()


def read_data(
    input_path: Path, usage_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    source = ad.read_h5ad(input_path, backed="r")
    try:
        required = {"sample", "annotation"}
        if not required <= set(source.obs.columns):
            raise ValueError("H5AD requires sample and annotation labels")
        if not source.obs_names.is_unique:
            raise ValueError("H5AD spot IDs must be unique")
        metadata = source.obs[["sample", "annotation"]].copy()
    finally:
        source.file.close()
    if metadata.empty or metadata.isna().any().any():
        raise ValueError("Spot labels must be nonempty and nonmissing")
    if set(metadata["sample"].astype(str)) != set(SAMPLE_ORDER):
        raise ValueError("Expected P7, P35 and P60 rep1 samples")
    labels = metadata["annotation"]
    if labels.astype(str).str.strip().eq("").any():
        raise ValueError("H5AD annotation labels must not be blank")
    if isinstance(labels.dtype, pd.CategoricalDtype):
        order = labels.cat.remove_unused_categories().cat.categories.tolist()
    else:
        order = labels.drop_duplicates().tolist()
    usage = pd.read_csv(
        usage_path, index_col="spot_id", dtype={"spot_id": str, "sample": str},
    )
    if not usage.index.is_unique or usage.index.isna().any():
        raise ValueError("Usage spot IDs must be unique and nonmissing")
    if set(usage.index) != set(metadata.index):
        raise ValueError("Usage and H5AD must contain exactly the same spots")
    usage = usage.loc[metadata.index]
    if "sample" not in usage or not np.array_equal(
        usage["sample"].to_numpy(), metadata["sample"].astype(str).to_numpy(),
    ):
        raise ValueError("Usage and H5AD sample labels must match by spot ID")
    programs = [name for name in usage if name.startswith("program_")]
    if not programs or not all(
        re.fullmatch(r"program_[1-9][0-9]*", name) for name in programs
    ):
        raise ValueError("Usage needs program_1, program_2, ... columns")
    programs.sort(key=lambda name: int(name.split("_")[1]))
    values = usage[programs].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError("Expected nonnegative usage summing to 1 per spot")
    # Group using current H5AD labels, rather than old IDs in the usage CSV.
    aligned_usage = pd.DataFrame(values, index=usage.index, columns=programs)
    return metadata, aligned_usage, order


def summarize_usage(
    metadata: pd.DataFrame, usage: pd.DataFrame, annotations: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    if not metadata.index.equals(usage.index):
        raise ValueError("Metadata and usage spot order must match")
    combined = usage.copy()
    combined["annotation"] = metadata["annotation"].to_numpy()
    combined["sample"] = metadata["sample"].astype(str).to_numpy()
    index = pd.MultiIndex.from_product(
        [annotations, SAMPLE_ORDER], names=["annotation", "sample"],
    )
    grouped = combined.groupby(["annotation", "sample"], observed=True)
    means = grouped[usage.columns.tolist()].mean().reindex(index)
    counts = grouped.size().reindex(index, fill_value=0)
    if int(counts.sum()) != len(metadata):
        raise ValueError("Some spots fall outside the requested annotations")
    valid = counts > 0
    np.testing.assert_allclose(means.loc[valid].sum(axis=1), 1, atol=1e-6)
    if not means.loc[~valid].isna().all().all():
        raise ValueError("Missing annotation-age groups must have missing means")
    return means, counts


def plot_heatmaps(
    means: pd.DataFrame, annotations: list[str], output: Path,
) -> None:
    programs = means.columns.tolist()
    upper = float(np.nanmax(means.to_numpy())) or 1.0
    normalization = Normalize(vmin=0, vmax=upper)
    color_map = COLOR_MAP.copy()
    color_map.set_bad("#F2F2F2")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [
            "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans",
        ], "axes.titlesize": 7, "axes.labelsize": 6,
        "xtick.labelsize": 5, "ytick.labelsize": 5,
    })
    figure, grid = plt.subplots(
        1, len(annotations), figsize=FIGURE_SIZE, sharey=True, squeeze=False,
    )
    try:
        for index, (axis, annotation) in enumerate(zip(
            grid.ravel(), annotations, strict=True,
        )):
            matrix = means.loc[annotation].reindex(SAMPLE_ORDER).to_numpy().T
            mesh = axis.pcolormesh(
                np.arange(4), np.arange(len(programs) + 1),
                np.ma.masked_invalid(matrix), cmap=color_map,
                norm=normalization, shading="flat",
                edgecolors="#444444", linewidth=0.35, antialiased=False,
            )
            for row, column in np.argwhere(~np.isfinite(matrix)):
                axis.text(column + 0.5, row + 0.5, "×", ha="center",
                          va="center", fontsize=5, color="#777777")
            axis.set_xticks(np.arange(3) + 0.5, AGE_LABELS, rotation=45,
                            ha="right", rotation_mode="anchor")
            axis.set_yticks(
                np.arange(len(programs)) + 0.5,
                [name.replace("program_", "P") for name in programs],
            )
            axis.set(xlim=(0, 3), ylim=(0, len(programs)))
            axis.tick_params(length=0, pad=1, labelleft=index == 0)
            for spine in axis.spines.values():
                spine.set_linewidth(0.5)
            title = ANNOTATION_TITLES.get(annotation, annotation)
            axis.set_title(textwrap.fill(title, 8), pad=3)
        figure.subplots_adjust(
            left=0.045, right=0.905, bottom=0.21, top=0.8, wspace=0.22,
        )
        figure.suptitle(
            f"cNMF programs (K={len(programs)}): mean usage by age "
            "within annotation", fontsize=7, y=0.9,
        )
        color_axis = figure.add_axes((0.94, 0.25, 0.012, 0.48))
        colorbar = figure.colorbar(
            mesh, cax=color_axis, ticks=np.linspace(0, upper, 4),
            format=PercentFormatter(xmax=1, decimals=0),
        )
        colorbar.ax.set_title("Mean\nusage", fontsize=5, pad=3)
        colorbar.ax.tick_params(labelsize=5, length=2, width=0.5, pad=2)
        colorbar.outline.set_linewidth(0.5)
        figure.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.02)
    finally:
        plt.close(figure)


def run(args: argparse.Namespace) -> None:
    for name in ("input", "usage", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    output = args.output_dir / OUTPUT_NAME
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise ValueError("Output directory is a file")
    if output.exists():
        raise ValueError("Output exists; choose a new output directory")
    metadata, usage, annotations = read_data(args.input, args.usage)
    means, counts = summarize_usage(metadata, usage, annotations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmaps(means, annotations, output)
    print(f"{len(annotations)} annotations; {len(usage.columns)} programs; "
          f"{len(metadata):,} spots", flush=True)
    print(f"Missing annotation-age groups: {int((counts == 0).sum())}; "
          f"shared color maximum: {means.max().max():.1%}", flush=True)
    print(f"Saved {output}")


def main() -> None:
    try:
        run(parse_args())
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
