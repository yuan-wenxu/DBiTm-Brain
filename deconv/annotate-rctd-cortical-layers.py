#!/usr/bin/env python3
"""Annotate cortical layers using absolute RCTD weights and same-layer neighbors."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


LAYER_TYPES = {
    "L2/3": ["L2/3 IT CTX Glut"],
    "L4/5": ["L4/5 IT CTX Glut"],
    "L5": ["L5 ET CTX Glut", "L5 IT CTX Glut"],
    "L6": ["L6 CT CTX Glut", "L6 IT CTX Glut"],
    "L6b": ["L6b CTX Glut"],
}
LAYER_CATEGORIES = [*LAYER_TYPES, "none"]
LAYER_COLORS = ["#A73030", "#E64B35", "#2F5597", "#4DBBD5", "#CC79A7", "#F0F0F0"]
LABEL_COLUMN = "rctd_cortical_layer"
CLUSTER_COLORS = [
    "#7E57C2", "#C5A3E0", "#D6A500", "#FFE082", "#2CA02C", "#264653", "#E7298A",
    "#98DF8A", "#C49C94",
]
FIGURE_DPI = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="RCTD directory containing rctd_weights.csv and spot_metadata.csv.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="New output directory; default: <input-dir>/cortical_layer_annotations.",
    )
    parser.add_argument(
        "--min-layer-weight", type=float, default=0.5,
        help=(
            "Minimum summed raw weight for a layer (default: 0.5). "
            "Exactly one qualifying layer requires a same-layer neighbor; "
            "conflicts, isolated assignments and zero qualifying layers are none."
        ),
    )
    parser.add_argument(
        "--min-layer-margin", type=float, default=0,
        help=(
            "Minimum absolute raw-weight difference between the top two layers (default: 0). "
            "Insufficient margins and ties among qualifying layers are none."
        ),
    )
    parser.add_argument(
        "--h5ad", type=Path,
        help=(
            "Plot every H5AD spot using its image and cluster labels; never export H5AD. "
            "RCTD spot coordinates come from metadata, remaining coordinates from H5AD obs."
        ),
    )
    parser.add_argument(
        "--cluster-key", default="leiden",
        help="H5AD obs column containing numeric cluster IDs (default: leiden).",
    )
    parser.add_argument("--spot-size-um", type=float, default=50)
    parser.add_argument("--crop-margin-um", type=float, default=500)
    args = parser.parse_args()
    for name in ("min_layer_weight", "min_layer_margin"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if (
        not np.isfinite([args.spot_size_um, args.crop_margin_um]).all()
        or args.spot_size_um <= 0 or args.crop_margin_um < 0
    ):
        parser.error("Spot size must be positive and crop margin must be nonnegative")
    return args


def read_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(input_dir / "spot_metadata.csv", index_col="barcode")
    weights = pd.read_csv(input_dir / "rctd_weights.csv", index_col="barcode")
    for name, table in (("metadata", metadata), ("weights", weights)):
        if table.empty or table.index.has_duplicates or table.index.isna().any():
            raise ValueError(f"{name} must contain nonempty, unique barcodes")
        if (table.index.astype(str).str.strip() == "").any():
            raise ValueError(f"{name} contains an empty barcode")
    if set(metadata.index) != set(weights.index):
        raise ValueError("Metadata and weights must have the same barcode set")
    return metadata, weights.loc[metadata.index]


def annotate_layers(
    weights: pd.DataFrame, metadata: pd.DataFrame,
    min_layer_weight: float = 0.5, min_layer_margin: float = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {cell_type for types in LAYER_TYPES.values() for cell_type in types}
    missing = required.difference(weights.columns)
    if missing:
        raise ValueError(f"Missing cortical reference cell types: {sorted(missing)}")
    if not weights.index.equals(metadata.index):
        raise ValueError("Weights and metadata must be aligned by barcode")
    values = weights.to_numpy(dtype=float)
    if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
        raise ValueError("Weights must be nonnegative finite numbers or missing")
    valid = np.isfinite(values).all(axis=1)
    grouped = pd.DataFrame(
        {layer: weights[types].sum(axis=1, skipna=False) for layer, types in LAYER_TYPES.items()},
        index=weights.index,
    )
    grouped.loc[~valid] = np.nan
    grouped.index.name = "barcode"
    maxima = grouped.max(axis=1, skipna=False)
    annotations = pd.DataFrame(
        {LABEL_COLUMN: "none", "rctd_best_layer_weight": maxima}, index=weights.index,
    )
    annotations.index.name = "barcode"
    sorted_values = np.sort(grouped.to_numpy(dtype=float), axis=1)
    margin = sorted_values[:, -1] - sorted_values[:, -2]
    fitted = np.ones(len(weights), dtype=bool)
    if "rctd_status" in metadata:
        fitted = metadata["rctd_status"].eq("fitted").to_numpy()
    qualifying = (grouped.ge(min_layer_weight) & grouped.gt(0)).sum(axis=1).to_numpy()
    single_layer = valid & fitted & (qualifying == 1)
    ambiguous = single_layer & (
        (margin < min_layer_margin) | np.isclose(margin, 0, atol=1e-12, rtol=0)
    )
    assigned = single_layer & ~ambiguous
    annotations.loc[assigned, LABEL_COLUMN] = grouped.loc[assigned].idxmax(axis=1)
    annotations = filter_spatial_neighbors(annotations, metadata)
    annotations[LABEL_COLUMN] = pd.Categorical(
        annotations[LABEL_COLUMN], categories=LAYER_CATEGORIES, ordered=True,
    )
    return annotations, grouped


def filter_spatial_neighbors(annotations: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Require one same-layer spot among the eight immediate array neighbors.

    Check all initial weight-based labels simultaneously. Missing grid positions
    provide no support; none labels are retained. Scores are unchanged.
    """
    if not annotations.index.equals(metadata.index):
        raise ValueError("Annotations and metadata must be aligned by barcode")
    coordinate_columns = ["array_row", "array_col"]
    missing = set(coordinate_columns).difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata grid coordinates: {sorted(missing)}")
    coordinates = metadata[coordinate_columns].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all() or (coordinates != np.floor(coordinates)).any():
        raise ValueError("Array coordinates must be finite integers")
    if metadata.duplicated(subset=coordinate_columns).any():
        raise ValueError("Array coordinates must identify unique spots")
    positions = [tuple(position) for position in coordinates.astype(np.int64)]
    labels = annotations[LABEL_COLUMN].to_numpy(dtype=str)
    grid_labels = dict(zip(positions, labels, strict=True))
    offsets = [(row, col) for row in (-1, 0, 1) for col in (-1, 0, 1) if row or col]
    isolated = np.zeros(len(labels), dtype=bool)
    for index, ((row, col), layer) in enumerate(zip(positions, labels, strict=True)):
        if layer in LAYER_TYPES:
            isolated[index] = not any(
                grid_labels.get((row + dr, col + dc)) == layer for dr, dc in offsets
            )
    result = annotations.copy()
    result.loc[isolated, LABEL_COLUMN] = "none"
    return result


def load_plot_data(
    path: Path, metadata: pd.DataFrame, annotations: pd.DataFrame, cluster_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, float]:
    coordinate_columns = ["pxl_row_in_fullres", "pxl_col_in_fullres", "array_row", "array_col"]
    missing = set(coordinate_columns).difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata coordinate columns: {sorted(missing)}")
    with h5py.File(path.expanduser().resolve(), "r") as handle:
        obs = ad.io.read_elem(handle["obs"])
        spatial = handle["uns/spatial"]
        if len(spatial) != 1:
            raise ValueError("Expected exactly one tissue-image library")
        library = spatial[next(iter(spatial))]
        image = np.asarray(library["images/hires"])
        scale = float(library["scalefactors/tissue_hires_scalef"][()])
        pixel_size = float(library["metadata/hires_pixel_size_um"][()])
    if obs.index.has_duplicates or not metadata.index.isin(obs.index).all():
        raise ValueError("Metadata barcodes must occur exactly once in the H5AD")
    if cluster_key not in obs:
        raise ValueError(f"Missing H5AD cluster column: {cluster_key}")
    plot_metadata = pd.DataFrame(index=obs.index, columns=coordinate_columns, dtype=float)
    plot_metadata.loc[metadata.index] = metadata[coordinate_columns].to_numpy(dtype=float)
    extra = ~obs.index.isin(metadata.index)
    if extra.any():
        missing = set(coordinate_columns).difference(obs.columns)
        if missing:
            raise ValueError(f"Missing H5AD coordinates for non-RCTD spots: {sorted(missing)}")
        plot_metadata.loc[extra] = obs.loc[extra, coordinate_columns].to_numpy(dtype=float)
    coordinates = plot_metadata.to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Plot coordinates must be finite for every H5AD spot")
    clusters = pd.to_numeric(obs[cluster_key], errors="raise").to_numpy(dtype=float)
    if (
        not np.isfinite(clusters).all() or (clusters < 0).any()
        or (clusters != np.floor(clusters)).any()
    ):
        raise ValueError("Cluster IDs must be nonnegative integers")
    if not np.isfinite([scale, pixel_size]).all() or scale <= 0 or pixel_size <= 0:
        raise ValueError("Image scale and pixel size must be positive and finite")
    if image.ndim not in (2, 3) or not image.shape[0] or not image.shape[1]:
        raise ValueError("Expected a nonempty tissue image")
    x, y = coordinates[:, 1] * scale, coordinates[:, 0] * scale
    if (x < 0).any() or (y < 0).any() or (x >= image.shape[1]).any() or (y >= image.shape[0]).any():
        raise ValueError("Spot coordinates fall outside the tissue image")
    table = annotations.reindex(obs.index)
    table.index.name = "barcode"
    table[LABEL_COLUMN] = table[LABEL_COLUMN].fillna("none")
    table["cluster"] = clusters.astype(np.int64).astype(str)
    table["plot_label"] = table[LABEL_COLUMN].astype(str)
    fallback = table[LABEL_COLUMN].eq("none")
    table.loc[fallback, "plot_label"] = "Cluster " + table.loc[fallback, "cluster"]
    return table, plot_metadata, image, x, y, pixel_size


def plot_colors(table: pd.DataFrame) -> dict[str, str]:
    colors = dict(zip(LAYER_CATEGORIES, LAYER_COLORS, strict=True))
    cluster_count = int(table["cluster"].astype(int).max()) + 1
    palette = CLUSTER_COLORS.copy()
    if cluster_count > len(palette):
        extra_count = cluster_count - len(palette)
        palette.extend(
            matplotlib.colors.to_hex(plt.get_cmap("turbo")(i / max(1, extra_count - 1)))
            for i in range(extra_count)
        )
    colors.update({f"Cluster {index}": color for index, color in enumerate(palette)})
    return colors


def add_annotation_squares(
    axis: plt.Axes, labels: np.ndarray, colors: dict[str, str],
    x: np.ndarray, y: np.ndarray, side: float, alpha: float = 1,
) -> None:
    squares = [
        Rectangle((cx - side / 2, cy - side / 2), side, side)
        for cx, cy in zip(x, y, strict=True)
    ]
    axis.add_collection(PatchCollection(
        squares, facecolors=[colors[label] for label in labels],
        edgecolors="none", alpha=alpha,
    ))


def add_scale_bar(axis: plt.Axes, width: float, height: float, pixel_size: float) -> None:
    target_um = width * pixel_size / 5
    magnitude = 10 ** np.floor(np.log10(target_um))
    length_um = max(
        multiplier * magnitude for multiplier in (1, 2, 5)
        if multiplier * magnitude <= target_um
    )
    scale_bar = AnchoredSizeBar(
        axis.transData, length_um / pixel_size, f"{length_um:g} µm",
        loc="lower left", pad=0.4, borderpad=0.5, sep=3, frameon=True,
        size_vertical=max(2, height * 0.003), color="white",
    )
    scale_bar.patch.set_facecolor("black")
    scale_bar.patch.set_alpha(0.6)
    scale_bar.patch.set_edgecolor("none")
    axis.add_artist(scale_bar)


def plot_tissue_overlay(
    path: Path, labels: np.ndarray, colors: dict[str, str],
    image: np.ndarray, x: np.ndarray, y: np.ndarray, pixel_size: float,
    spot_size: float, bounds: tuple[float, float, float, float],
    title: str, handles: list[Patch], highlight: np.ndarray | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(image, cmap="gray", origin="upper")
    side = spot_size / pixel_size
    if highlight is None:
        add_annotation_squares(axis, labels, colors, x, y, side)
    else:
        other = ~highlight
        add_annotation_squares(
            axis, np.full(other.sum(), "other"), {"other": "#D9D9D9"},
            x[other], y[other], side, alpha=0.3,
        )
        if highlight.any():
            add_annotation_squares(
                axis, labels[highlight], colors, x[highlight], y[highlight], side,
            )
    left, right, top, bottom = bounds
    axis.set(xlim=(left, right), ylim=(bottom, top), aspect="equal")
    axis.set_title(title, fontsize=12)
    add_scale_bar(axis, right - left, bottom - top, pixel_size)
    axis.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_annotations(
    output_dir: Path, table: pd.DataFrame,
    image: np.ndarray, x: np.ndarray, y: np.ndarray, pixel_size: float,
    spot_size: float, crop_margin: float,
) -> None:
    labels = table["plot_label"].to_numpy(dtype=str)
    colors = plot_colors(table)
    overview = table[LABEL_COLUMN].isin(LAYER_TYPES).to_numpy()
    legend_labels = [label for label in LAYER_TYPES if label in labels]
    handles = [
        Patch(facecolor=colors[label], edgecolor="none", label=label) for label in legend_labels
    ]
    margin = (crop_margin + spot_size / 2) / pixel_size
    bounds = {
        "full": (0.0, float(image.shape[1]), 0.0, float(image.shape[0])),
        "cropped": (
            max(0, x.min() - margin), min(image.shape[1], x.max() + margin),
            max(0, y.min() - margin), min(image.shape[0], y.max() + margin),
        ),
    }
    for view, view_bounds in bounds.items():
        plot_tissue_overlay(
            output_dir / f"cortical_annotations_{view}.png", labels[overview], colors,
            image, x[overview], y[overview], pixel_size, spot_size, view_bounds,
            "Cortical layer annotations", handles,
        )
    layer_dir = output_dir / "layers"
    layer_dir.mkdir()
    for layer in LAYER_TYPES:
        highlight = table[LABEL_COLUMN].eq(layer).to_numpy()
        layer_handles = [
            Patch(
                facecolor=colors[layer], edgecolor="none",
                label=f"{layer} ({highlight.sum()} spots)",
            ),
            Patch(facecolor="#D9D9D9", edgecolor="none", alpha=0.3, label="Other spots"),
        ]
        stem = layer.replace("/", "_")
        for view, view_bounds in bounds.items():
            plot_tissue_overlay(
                layer_dir / f"cortical_layer_{stem}_{view}_merge.png", labels, colors,
                image, x, y, pixel_size, spot_size, view_bounds,
                f"{layer} cortical annotation", layer_handles, highlight,
            )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir or input_dir / "cortical_layer_annotations"
    ).expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    metadata, weights = read_inputs(input_dir)
    annotations, grouped = annotate_layers(
        weights, metadata, args.min_layer_weight, args.min_layer_margin,
    )
    table = annotations.copy()
    if args.h5ad:
        table, _, image, x, y, pixel_size = load_plot_data(
            args.h5ad, metadata, annotations, args.cluster_key,
        )
    labels = table[LABEL_COLUMN]
    summary = pd.DataFrame([
        {
            "layer": layer, "n_spots": int(labels.eq(layer).sum()),
            "fraction_of_spots": float(labels.eq(layer).mean()), "color": color,
        }
        for layer, color in zip(LAYER_CATEGORIES, LAYER_COLORS, strict=True)
    ])

    output_dir.mkdir(parents=True)
    table.to_csv(output_dir / "spot_cortical_annotations.csv")
    grouped.reindex(table.index).to_csv(output_dir / "cortical_layer_weights.csv")
    summary.to_csv(output_dir / "cortical_layer_summary.csv", index=False)

    if args.h5ad:
        plot_annotations(
            output_dir, table, image, x, y, pixel_size, args.spot_size_um, args.crop_margin_um,
        )
        print(
            f"Loaded {len(table)} H5AD spots; "
            f"overview shows {table[LABEL_COLUMN].ne('none').sum()} "
            "cortical annotations."
        )
        print("Saved full and cropped tissue overlays, including five individual layers (300 DPI).")
    print(summary[["layer", "n_spots", "fraction_of_spots"]].to_string(index=False))
    print(f"Annotations: {output_dir}")


if __name__ == "__main__":
    main()
