#!/usr/bin/env python3
"""Overlay per-spot CG site counts on the tissue image stored in an H5AD."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from utils import spatial_plot_data


CG_COLORMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "qc_blue_white_red", ("#4DBBD5", "#FFFFFF", "#E64B35")
)
OUTPUT_NAMES = (
    "cg-sites-spatial.png",
    "cg-sites-spatial-cropped.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True,
                        help="Input per_spot_summary.tsv with cpg_site_count.")
    parser.add_argument("--input", type=Path, required=True,
                        help="H5AD providing spot coordinates and the tissue image.")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to cg-sites-spatial/ beside the H5AD.")
    parser.add_argument("--spot-size-um", type=float, default=50.0,
                        help="Spot-square side length in micrometers (default: 50).")
    parser.add_argument("--crop-margin-um", type=float, default=500.0,
                        help="Margin around the cropped view in micrometers (default: 500).")
    parser.add_argument("--alpha", type=float, default=0.85,
                        help="Spot opacity over the unmodified image (default: 0.85).")
    parser.add_argument("--vmax-quantile", type=float, default=0.95,
                        help="Upper color limit quantile; 1 uses the maximum (default: 0.95).")
    args = parser.parse_args()
    if not np.isfinite(args.spot_size_um) or args.spot_size_um <= 0:
        parser.error("--spot-size-um must be finite and > 0")
    if not np.isfinite(args.crop_margin_um) or args.crop_margin_um < 0:
        parser.error("--crop-margin-um must be finite and >= 0")
    if not 0 < args.alpha <= 1:
        parser.error("--alpha must be in (0, 1]")
    if not 0 < args.vmax_quantile <= 1:
        parser.error("--vmax-quantile must be in (0, 1]")
    return args


def integer_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError(f"{column} must contain finite, nonnegative integers")
    return values.astype(np.int64)


def match_counts(summary_path: Path, obs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Join array coordinates exactly, preserving H5AD spot order and barcodes."""
    summary = pd.read_csv(summary_path, sep="\t")
    for frame, columns in (
        (summary, ("row_index", "col_index", "cpg_site_count")),
        (obs, ("array_row", "array_col")),
    ):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    summary_keys = pd.MultiIndex.from_arrays([
        integer_values(summary, "row_index"), integer_values(summary, "col_index")
    ])
    obs_keys = pd.MultiIndex.from_arrays([
        integer_values(obs, "array_row"), integer_values(obs, "array_col")
    ])
    if summary_keys.has_duplicates or obs_keys.has_duplicates:
        raise ValueError("Duplicate row/column coordinate pairs in the summary or H5AD")
    counts = pd.Series(integer_values(summary, "cpg_site_count"), index=summary_keys)
    matched = counts.reindex(obs_keys)
    if matched.isna().any():
        raise ValueError(f"{matched.isna().sum()} H5AD spots have no summary entry")
    if obs.empty:
        raise ValueError("The H5AD contains no spots")
    table = pd.DataFrame({
        "barcode": obs.index.astype(str),
        "row_index": obs_keys.get_level_values(0),
        "col_index": obs_keys.get_level_values(1),
        "cpg_site_count": matched.to_numpy(dtype=np.int64),
    })
    return table, len(summary)


def add_scale_bar(axis: plt.Axes, width: float, height: float, pixel_size_um: float) -> None:
    target_um = width * pixel_size_um / 5
    magnitude = 10 ** np.floor(np.log10(target_um))
    length_um = max(value * magnitude for value in (1, 2, 5) if value * magnitude <= target_um)
    bar = AnchoredSizeBar(
        axis.transData, length_um / pixel_size_um, f"{length_um:g} µm",
        loc="lower left", pad=0.4, borderpad=0.5, sep=3, frameon=True,
        size_vertical=max(2, height * 0.003), color="white",
    )
    bar.patch.set_facecolor("black")
    bar.patch.set_alpha(0.6)
    bar.patch.set_edgecolor("none")
    axis.add_artist(bar)


def plot_counts(
    path: Path,
    image: np.ndarray,
    table: pd.DataFrame,
    pixel_size_um: float,
    args: argparse.Namespace,
    norm: matplotlib.colors.Normalize,
    cropped: bool,
) -> None:
    x = table["x_hires"].to_numpy()
    y = table["y_hires"].to_numpy()
    values = table["cpg_site_count"].to_numpy()
    side = args.spot_size_um / pixel_size_um
    left, right, top, bottom = 0.0, float(image.shape[1]), 0.0, float(image.shape[0])
    if cropped:
        margin = (args.crop_margin_um + args.spot_size_um / 2) / pixel_size_um
        left, right = max(left, x.min() - margin), min(right, x.max() + margin)
        top, bottom = max(top, y.min() - margin), min(bottom, y.max() + margin)
    if left >= right or top >= bottom:
        raise ValueError("Spot coordinates do not overlap the tissue image")
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(image, cmap="gray", origin="upper")
    collection = PatchCollection(
        [Rectangle((xc - side / 2, yc - side / 2), side, side)
         for xc, yc in zip(x, y, strict=True)],
        cmap=CG_COLORMAP, norm=norm, edgecolors="none", alpha=args.alpha,
    )
    collection.set_array(values)
    axis.add_collection(collection)
    axis.set_xlim(left, right)
    axis.set_ylim(bottom, top)
    axis.set_aspect("equal")
    axis.set_title("CG site counts on tissue image", fontsize=12)
    axis.set_axis_off()
    add_scale_bar(axis, right - left, bottom - top, pixel_size_um)
    color_axis = make_axes_locatable(axis).append_axes("right", size="4%", pad="2%")
    # Keep the legend opaque: translucent spots blend with the local tissue image.
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=CG_COLORMAP), cax=color_axis,
        format=FuncFormatter(lambda value, _: f"{value:,.0f}"),
    )
    colorbar.set_label("CG sites / spot", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    if (values > norm.vmax).any():
        colorbar.ax.set_title(f"P{args.vmax_quantile * 100:g} cap", fontsize=8, pad=10)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    output_dir = (args.output_dir or input_path.parent / "cg-sites-spatial").expanduser().resolve()
    existing = [output_dir / name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if existing:
        raise ValueError(f"Refusing to overwrite existing output: {existing[0]}")
    adata = ad.read_h5ad(input_path, backed="r")
    try:
        table, summary_count = match_counts(summary_path, adata.obs)
        image, x, y, pixel_size_um = spatial_plot_data(adata)
    finally:
        adata.file.close()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Spatial coordinates must be finite")
    if ((x < 0) | (x >= image.shape[1]) | (y < 0) | (y >= image.shape[0])).any():
        raise ValueError("Spot centers must lie within the stored tissue image")
    table["x_hires"] = x
    table["y_hires"] = y
    values = table["cpg_site_count"].to_numpy()
    vmin = float(values.min())
    vmax = float(np.quantile(values, args.vmax_quantile))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1), clip=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for cropped, name in ((False, OUTPUT_NAMES[0]), (True, OUTPUT_NAMES[1])):
        plot_counts(output_dir / name, image, table, pixel_size_um, args, norm, cropped)

    print(f"Matched {len(table):,}/{len(table):,} H5AD spots; summary contains {summary_count:,} spots.")
    print(f"CG sites: {values.min():,}–{values.max():,}; color upper limit: {norm.vmax:,.1f}.")
    print(f"Saved full-image and cropped overlays to {output_dir}")


if __name__ == "__main__":
    main()
