#!/usr/bin/env python3
"""Export full/cropped tissue overlays and array-position PNGs per RCTD type."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

COORDINATE_COLUMNS = ["pxl_row_in_fullres", "pxl_col_in_fullres"]
ARRAY_COLUMNS = ["array_row", "array_col"]
WEIGHT_COLORS = ["#4DBBD5", "#FFFFFF", "#E64B35"]
WEIGHT_CMAP = LinearSegmentedColormap.from_list("rctd_project_red_blue", WEIGHT_COLORS)
WEIGHT_CMAP.set_bad("#BDBDBD")
FIGURE_DPI = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="RCTD CSV directory.")
    parser.add_argument("--h5ad", type=Path, required=True, help="H5AD providing only the tissue image and its scale metadata; spot coordinates come from spot_metadata.csv.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; existing paths are refused.")
    parser.add_argument("--value-table", choices=("weights", "proportions"), default="weights", help="Raw weights (default) or normalized proportions.")
    parser.add_argument("--spot-size-um", type=float, default=50)
    parser.add_argument("--margin-um", type=float, default=500, help="Margin around spots in cropped figures (default: 500 um).")
    return parser.parse_args()


def validate_values(values: pd.DataFrame) -> None:
    if values.empty or values.index.has_duplicates or values.columns.has_duplicates:
        raise ValueError("Weight table must be nonempty with unique barcodes and cell types")
    array = values.to_numpy(dtype=float)
    if np.isinf(array).any() or (array[np.isfinite(array)] < 0).any():
        raise ValueError("Weights must be nonnegative finite numbers or missing")
    finite = array[np.isfinite(array)]
    if not len(finite):
        raise ValueError("No finite weights to plot")


def load_spatial_plot_data(
    metadata: pd.DataFrame, h5ad_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    missing = set(COORDINATE_COLUMNS).difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata coordinate columns: {sorted(missing)}")
    coordinates = metadata[COORDINATE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Metadata pixel coordinates must be finite")
    with h5py.File(h5ad_path, "r") as handle:
        spatial = handle["uns/spatial"]
        if len(spatial) != 1:
            raise ValueError("Expected exactly one tissue-image library under uns['spatial']")
        library = spatial[next(iter(spatial))]
        image = np.asarray(library["images/hires"])
        scale = float(library["scalefactors/tissue_hires_scalef"][()])
        pixel_size = float(library["metadata/hires_pixel_size_um"][()])
    if not np.isfinite([scale, pixel_size]).all() or scale <= 0 or pixel_size <= 0:
        raise ValueError("Image scale and pixel size must be finite and greater than zero")
    if image.ndim not in (2, 3) or not image.shape[0] or not image.shape[1]:
        raise ValueError("Expected a nonempty tissue image")
    x = coordinates[:, 1] * scale
    y = coordinates[:, 0] * scale
    return image, x, y, pixel_size


def add_weight_squares(
    axis: plt.Axes, values: np.ndarray, vmax: float,
    x: np.ndarray, y: np.ndarray, side: float,
) -> PatchCollection:
    squares = [Rectangle((cx - side / 2, cy - side / 2), side, side) for cx, cy in zip(x, y, strict=True)]
    spots = PatchCollection(squares, cmap=WEIGHT_CMAP, norm=Normalize(vmin=0, vmax=vmax), edgecolors="none")
    spots.set_array(np.ma.masked_invalid(values))
    axis.add_collection(spots)
    return spots


def add_scale_bar(
    axis: plt.Axes, view_width_pixels: float,
    view_height_pixels: float, pixel_size_um: float,
) -> None:
    target_length_um = view_width_pixels * pixel_size_um / 5
    magnitude = 10 ** np.floor(np.log10(target_length_um))
    scale_bar_um = max(
        multiplier * magnitude
        for multiplier in (1, 2, 5)
        if multiplier * magnitude <= target_length_um
    )
    scale_bar = AnchoredSizeBar(
        axis.transData, scale_bar_um / pixel_size_um, f"{scale_bar_um:g} µm",
        loc="lower left", pad=0.4, borderpad=0.5, sep=3, frameon=True,
        size_vertical=max(2, view_height_pixels * 0.003), color="white",
    )
    scale_bar.patch.set_facecolor("black")
    scale_bar.patch.set_alpha(0.6)
    scale_bar.patch.set_edgecolor("none")
    axis.add_artist(scale_bar)


def draw_map(
    axis: plt.Axes, values: np.ndarray, vmax: float, image: np.ndarray,
    x: np.ndarray, y: np.ndarray, pixel_size: float, spot_size: float,
    bounds: tuple[float, float, float, float],
) -> PatchCollection:
    axis.imshow(image, cmap="gray", origin="upper")
    spots = add_weight_squares(axis, values, vmax, x, y, spot_size / pixel_size)
    left, right, top, bottom = bounds
    axis.set(xlim=(left, right), ylim=(bottom, top), aspect="equal")
    axis.axis("off")
    add_scale_bar(axis, right - left, bottom - top, pixel_size)
    return spots


def add_weight_colorbar(
    figure: plt.Figure, axis: plt.Axes, spots: PatchCollection,
    vmax: float, metric: str,
) -> None:
    colorbar = figure.colorbar(spots, ax=axis, fraction=0.035, pad=0.025)
    colorbar.set_label(metric, fontsize=10)
    colorbar.set_ticks(np.linspace(0, vmax, 5))
    colorbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3g"))


def plot_array_positions(
    output_dir: Path, stem: str, cell_type: str, values: np.ndarray,
    vmax: float, rows: np.ndarray, columns: np.ndarray, metric: str,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    spots = add_weight_squares(axis, values, vmax, columns, rows, 0.8)
    padding = 0.75
    axis.set_xlim(columns.max() + padding, columns.min() - padding)
    axis.set_ylim(rows.max() + padding, rows.min() - padding)
    axis.set_aspect("equal")
    axis.set_title(cell_type, fontsize=12)
    axis.set_axis_off()
    add_weight_colorbar(figure, axis, spots, vmax, metric)
    figure.tight_layout()
    save_figure(figure, output_dir, f"{stem}_array_positions")


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise ValueError(f"Output directory already exists: {args.output_dir}")
    if not np.isfinite([args.spot_size_um, args.margin_um]).all() or args.spot_size_um <= 0 or args.margin_um < 0:
        raise ValueError("Invalid spot size or margin")
    input_dir = args.input_dir.expanduser().resolve()
    table_path = input_dir / f"rctd_{args.value_table}.csv"
    metadata = pd.read_csv(input_dir / "spot_metadata.csv", index_col="barcode")
    values = pd.read_csv(table_path, index_col="barcode")
    validate_values(values)
    if metadata.index.has_duplicates or set(metadata.index) != set(values.index):
        raise ValueError("Metadata and weight table must have the same unique barcodes")
    values = values.loc[metadata.index]
    missing_array = set(ARRAY_COLUMNS).difference(metadata.columns)
    if missing_array:
        raise ValueError(f"Missing metadata array coordinates: {sorted(missing_array)}")
    array_coordinates = metadata[ARRAY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(array_coordinates).all():
        raise ValueError("Metadata array coordinates must be finite")
    rows, columns = array_coordinates[:, 0], array_coordinates[:, 1]
    image, x, y, pixel_size = load_spatial_plot_data(metadata, args.h5ad)
    if not np.isfinite([*x, *y]).all() or (x < 0).any() or (y < 0).any() or (x >= image.shape[1]).any() or (y >= image.shape[0]).any():
        raise ValueError("Spot coordinates fall outside the tissue image")
    margin = (args.margin_um + args.spot_size_um / 2) / pixel_size
    full_bounds = (0.0, float(image.shape[1]), 0.0, float(image.shape[0]))
    cropped_bounds = (max(0, x.min() - margin), min(image.shape[1], x.max() + margin), max(0, y.min() - margin), min(image.shape[0], y.max() + margin))
    metric = "RCTD raw weight" if args.value_table == "weights" else "RCTD proportion"

    args.output_dir.mkdir(parents=True)

    for index, cell_type in enumerate(values.columns, start=1):
        weights = values[cell_type].to_numpy(dtype=float)
        finite = weights[np.isfinite(weights)]
        type_max = float(finite.max()) if len(finite) else np.nan
        vmax = type_max if np.isfinite(type_max) and type_max > 0 else 1.0
        stem = f"{index:02d}_{re.sub(r'[^A-Za-z0-9]+', '_', cell_type).strip('_')}"
        for view, bounds in (("full", full_bounds), ("cropped", cropped_bounds)):
            figure, axis = plt.subplots(figsize=(7, 6))
            spots = draw_map(axis, weights, vmax, image, x, y, pixel_size, args.spot_size_um, bounds)
            axis.set_title(cell_type, fontsize=12)
            add_weight_colorbar(figure, axis, spots, vmax, metric)
            figure.tight_layout()
            save_figure(figure, args.output_dir, f"{stem}_{view}")

        plot_array_positions(args.output_dir, stem, cell_type, weights, vmax, rows, columns, metric)

        print(f"[{index}/{len(values.columns)}] {cell_type}: max = {type_max:.4g}", flush=True)

    print(f"Full, cropped, and array-position PNG figures (300 DPI): {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
