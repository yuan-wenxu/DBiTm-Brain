#!/usr/bin/env python3
"""Plot LR pattern fractions, representative pairs and annotation means."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PALETTE = [
    "#A73030", "#E64B35", "#2F5597", "#4DBBD5", "#CC79A7",
    "#7E57C2", "#C5A3E0", "#D6A500", "#FFE082", "#2CA02C",
    "#264653", "#E7298A", "#98DF8A", "#C49C94",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: INPUT_DIR/plots; must not exist.")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir / "plots"
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a new directory")
    return args


def configure_plots() -> LinearSegmentedColormap:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                            "DejaVu Sans"],
        "font.size": 5, "axes.titlesize": 7, "axes.labelsize": 6,
        "xtick.labelsize": 5, "ytick.labelsize": 5,
        "legend.fontsize": 5, "axes.linewidth": 0.5,
        "axes.titlepad": 3, "ytick.major.width": 0.5,
        "xtick.major.width": 0.5,
    })
    colormap = LinearSegmentedColormap.from_list(
        "cyan_white_red", ["#4DBBD5", "#FFFFFF", "#E64B35"])
    colormap.set_bad("#D9D9D9")
    return colormap


def save_figure(figure: plt.Figure, destination: Path) -> None:
    figure.savefig(destination, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def spatial_geometry(
    observations: pd.DataFrame, samples: list[str]
) -> tuple[dict, float, float]:
    geometry = {}
    width, height = 0.0, 0.0
    for sample in samples:
        mask = observations["sample"].eq(sample).to_numpy()
        coordinates = observations.loc[mask, ["x_um", "y_um"]].to_numpy()
        neighbors, _ = cKDTree(coordinates).query(coordinates, k=2)
        pitch = float(np.median(neighbors[:, 1]))
        if not np.isfinite(pitch) or pitch <= 0:
            raise ValueError(f"Invalid spot spacing for {sample}")
        center = (coordinates.min(axis=0) + coordinates.max(axis=0)) / 2
        centered = coordinates - center
        span = np.ptp(coordinates, axis=0)
        width = max(width, float(span[0]) + 2 * pitch)
        height = max(height, float(span[1]) + 2 * pitch)
        side = pitch * 0.90
        geometry[sample] = {
            "mask": mask,
            "patches": [Rectangle((x - side / 2, y - side / 2), side, side)
                        for x, y in centered],
        }
    return geometry, width, height


def plot_spatial(
    usage: pd.DataFrame, patterns: list[str], samples: list[str],
    colormap: LinearSegmentedColormap, args: argparse.Namespace,
) -> None:
    geometry, width, height = spatial_geometry(usage, samples)
    for pattern in patterns:
        field = pattern.replace("pattern_", "fraction_")
        values = usage[field].to_numpy()
        maximum = float(np.nanmax(values))
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError(f"No positive fraction for {pattern}")
        norm = Normalize(0, maximum)
        figure = plt.figure(figsize=(5, 2))
        grid = figure.add_gridspec(
            1, len(samples) + 1, width_ratios=[1] * len(samples) + [0.045],
            left=0.02, right=0.91, bottom=0.18, top=0.83, wspace=0.10)
        for column, sample in enumerate(samples):
            axis = figure.add_subplot(grid[0, column])
            data = geometry[sample]
            collection = PatchCollection(
                data["patches"], cmap=colormap, norm=norm,
                edgecolor="#D9D9D9", linewidth=0.08)
            collection.set_array(np.ma.masked_invalid(values[data["mask"]]))
            axis.add_collection(collection)
            axis.set_xlim(-width / 2, width / 2)
            axis.set_ylim(height / 2, -height / 2)
            axis.set_aspect("equal")
            axis.axis("off")
            axis.set_title(sample.split("-")[0].upper(), pad=3)
            if column == 0:
                length = (500 if width >= 1000
                          else 10 ** np.floor(np.log10(width / 3)))
                x_start, y_start = width / 2 - length * 1.25, height / 2
                axis.plot([x_start, x_start + length], [y_start, y_start],
                          color="#333333", linewidth=0.6, clip_on=False)
                axis.text(x_start + length / 2, y_start,
                          f"{length:g} μm", va="bottom", ha="center",
                          fontsize=5)
        colorbar = figure.colorbar(collection,
                                  cax=figure.add_subplot(grid[0, -1]))
        colorbar.set_ticks([0, norm.vmax / 2, norm.vmax])
        colorbar.ax.set_yticklabels([
            "0", f"{norm.vmax / 2:.2g}", f"{norm.vmax:.2g}"])
        colorbar.ax.tick_params(labelsize=5, length=1.5, pad=1)
        colorbar.outline.set_linewidth(0.3)
        figure.suptitle(pattern.replace("_", " ").capitalize(),
                        fontsize=7, y=0.98)
        save_figure(figure, args.output_dir / f"{pattern}_spatial.png")


def plot_top_pairs(
    loadings: pd.DataFrame, patterns: list[str], args: argparse.Namespace,
) -> None:
    for index, pattern in enumerate(patterns):
        selected = loadings.loc[loadings["nmf_included"]].nlargest(
            10, pattern).iloc[::-1]
        figure, axis = plt.subplots(figsize=(2, 2))
        axis.barh(selected["lr_pair"], selected[pattern],
                  color=PALETTE[index % len(PALETTE)])
        axis.set_xlabel("Pair loading (sum across all pairs = 1)")
        axis.set_title(pattern.replace("_", " ").capitalize())
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2, pad=2)
        figure.subplots_adjust(left=0.30, right=0.99, bottom=0.23, top=0.86)
        save_figure(figure, args.output_dir / f"{pattern}_top_pairs.png")


def plot_usage_by_annotation(
    usage: pd.DataFrame, patterns: list[str], samples: list[str],
    colormap: LinearSegmentedColormap, output: Path,
) -> None:
    """Plot mean spot fractions by sample within each annotation."""
    if "annotation" not in usage or usage["annotation"].isna().any():
        raise ValueError("Annotation labels must be present and nonmissing")
    annotations = usage["annotation"].unique().tolist()
    fields = [name.replace("pattern_", "fraction_") for name in patterns]
    groups = pd.MultiIndex.from_product(
        [annotations, samples], names=["annotation", "sample"])
    means = usage.groupby(
        ["annotation", "sample"], sort=False, observed=True,
    )[fields].mean().reindex(groups)
    maximum = float(np.nanmax(means.to_numpy()))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("No positive pattern fractions for annotation means")
    normalization = Normalize(0, maximum)
    figure, axes = plt.subplots(
        1, len(annotations), figsize=(3, 2), sharey=True, squeeze=False)
    for index, (axis, annotation) in enumerate(zip(
        axes.ravel(), annotations, strict=True,
    )):
        matrix = means.loc[annotation].reindex(samples).to_numpy().T
        mesh = axis.pcolormesh(
            np.arange(len(samples) + 1), np.arange(len(patterns) + 1),
            np.ma.masked_invalid(matrix), cmap=colormap,
            norm=normalization, shading="flat", edgecolors="#444444",
            linewidth=0.35, antialiased=False)
        for row, column in np.argwhere(~np.isfinite(matrix)):
            axis.text(column + 0.5, row + 0.5, "×", ha="center",
                      va="center", fontsize=5, color="#777777")
        axis.set_xticks(
            np.arange(len(samples)) + 0.5,
            [sample.split("-")[0].upper() for sample in samples],
            rotation=45, ha="right", rotation_mode="anchor")
        axis.set_yticks(
            np.arange(len(patterns)) + 0.5,
            [name.replace("pattern_", "P") for name in patterns])
        axis.set(xlim=(0, len(samples)), ylim=(0, len(patterns)))
        axis.tick_params(length=0, pad=1, labelleft=index == 0)
        axis.set_title(str(annotation), pad=3)
    figure.subplots_adjust(
        left=0.075, right=0.85, bottom=0.19, top=0.80, wspace=0.16)
    figure.suptitle(
        f"LR patterns (K={len(patterns)}): mean fraction by annotation",
        fontsize=7, y=0.99)
    color_axis = figure.add_axes((0.89, 0.24, 0.018, 0.48))
    colorbar = figure.colorbar(
        mesh, cax=color_axis, ticks=np.linspace(0, maximum, 4),
        format=PercentFormatter(xmax=1, decimals=0))
    colorbar.ax.set_title("Mean\nfraction", fontsize=5, pad=3)
    colorbar.ax.tick_params(labelsize=5, length=2, width=0.5, pad=2)
    colorbar.outline.set_linewidth(0.5)
    save_figure(figure, output)


def main() -> None:
    args = parse_args()
    usage = pd.read_csv(args.input_dir / "nmf_spot_usage.csv",
                        dtype={"spot_id": str})
    loadings = pd.read_csv(args.input_dir / "nmf_pair_loadings.csv")
    patterns = sorted(
        [name for name in usage if name.startswith("pattern_")
         and name.removeprefix("pattern_").isdigit()],
        key=lambda name: int(name.removeprefix("pattern_")))
    if not patterns:
        raise ValueError("No pattern activity columns in the input table")
    activity = usage[patterns].to_numpy(dtype=float)
    totals = activity.sum(axis=1, keepdims=True)
    fraction_names = [name.replace("pattern_", "fraction_")
                      for name in patterns]
    usage[fraction_names] = np.divide(
        activity, totals, out=np.full_like(activity, np.nan),
        where=totals > 0)
    samples = usage["sample"].unique().tolist()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    colormap = configure_plots()
    plot_spatial(usage, patterns, samples, colormap, args)
    plot_top_pairs(loadings, patterns, args)
    plot_usage_by_annotation(
        usage, patterns, samples, colormap,
        args.output_dir / "pattern-usage-by-annotation.png")
    print(f"Saved {2 * len(patterns) + 1} PNGs to {args.output_dir}")


if __name__ == "__main__":
    main()
