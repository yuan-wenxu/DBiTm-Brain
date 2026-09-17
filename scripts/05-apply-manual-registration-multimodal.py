#!/usr/bin/env python3
"""Apply manual hires-pixel registration and plot clusters on TAPS-beta."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import anndata as ad
import matplotlib
import mudata as md
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle


md.set_options(pull_on_update=False)

FIGURE_DPI = 300
REFERENCE_ID = "taps_beta"
REFERENCE_SPOT_COLOR = "#4DBBD5"
MOVING_SPOT_COLOR = "#E64B35"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrna", type=Path, required=True)
    parser.add_argument("--taps", type=Path, required=True)
    parser.add_argument("--taps-beta", dest="taps_beta", type=Path, required=True)
    parser.add_argument("--mrna-offsets", type=Path, required=True)
    parser.add_argument("--taps-offsets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--spot-side-um",
        type=float,
        default=50.0,
        help="Spatial spot square side length in micrometers (default: 50).",
    )
    args = parser.parse_args()
    if args.spot_side_um <= 0:
        parser.error("--spot-side-um must be greater than zero")
    return args


def read_modality(
    path: Path,
    modality_id: str,
    label: str,
    cluster_key: str,
    cluster_prefix: str,
) -> tuple[ad.AnnData, dict[str, object]]:
    print(f"Reading {label}: {path}", flush=True)
    adata = ad.read_h5ad(path)
    library = next(iter(adata.uns["spatial"].values()))
    image = np.asarray(library["images"]["hires"]).copy()
    scale = float(library["scalefactors"]["tissue_hires_scalef"])
    pixel_size_um = float(library["metadata"]["hires_pixel_size_um"])
    clusters = adata.obs[cluster_key]
    source_cluster_labels = [str(value) for value in clusters.cat.categories]
    cluster_colors = [
        str(value)
        for value in np.asarray(adata.uns[f"{cluster_key}_colors"]).tolist()
    ]
    if len(cluster_colors) != len(source_cluster_labels):
        raise ValueError(
            f"Expected one {cluster_key!r} color per cluster in {path}, found "
            f"{len(cluster_colors)} colors for {len(source_cluster_labels)} clusters"
        )
    cluster_numbers = []
    for cluster_label in source_cluster_labels:
        numeric_label = (
            cluster_label[1:] if cluster_label.startswith("D") else cluster_label
        )
        try:
            cluster_number = int(numeric_label)
        except ValueError as error:
            raise ValueError(
                f"Expected numeric cluster labels in {path}, found {cluster_label!r}"
            ) from error
        cluster_numbers.append(cluster_number)
    cluster_labels = [
        f"{cluster_prefix}{cluster_number}"
        for cluster_number in cluster_numbers
    ]
    adata.obs[cluster_key] = clusters.cat.rename_categories(cluster_labels)
    clusters = adata.obs[cluster_key]
    result = {
        "id": modality_id,
        "label": label,
        "image": image,
        "x": adata.obs["pxl_col_in_fullres"].to_numpy(dtype=float) * scale,
        "y": adata.obs["pxl_row_in_fullres"].to_numpy(dtype=float) * scale,
        "clusterCodes": clusters.cat.codes.to_numpy(dtype=int),
        "clusterLabels": cluster_labels,
        "clusterColors": cluster_colors,
        "pixelSizeUm": pixel_size_um,
        "hiresScale": scale,
    }
    print(
        f"  {len(result['x']):,} spots; hires image "
        f"{image.shape[1]:,} x {image.shape[0]:,}",
        flush=True,
    )
    return adata, result


def read_offsets(path: Path, moving_id: str) -> tuple[float, float, float]:
    values = json.loads(path.read_text(encoding="utf-8"))[
        f"{moving_id}_to_{REFERENCE_ID}"
    ]
    return (
        float(values["row_offset_hires_pixels"]),
        float(values["col_offset_hires_pixels"]),
        float(values["rotation_degrees"]),
    )


def register_coordinates(
    layer: dict[str, object],
    row_offset: float,
    col_offset: float,
    rotation_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = layer["image"].shape[:2]
    center_col = width / 2
    center_row = height / 2
    radians = np.deg2rad(rotation_degrees)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    centered_col = layer["x"] - center_col
    centered_row = layer["y"] - center_row
    registered_col = (
        cosine * centered_col
        - sine * centered_row
        + center_col
        + col_offset
    )
    registered_row = (
        sine * centered_col
        + cosine * centered_row
        + center_row
        + row_offset
    )
    return registered_col, registered_row


def replace_spatial_coordinates(
    adata: ad.AnnData,
    registered_col: np.ndarray,
    registered_row: np.ndarray,
    reference_hires_scale: float,
    offsets: tuple[float, float, float],
) -> None:
    row_offset, col_offset, rotation_degrees = offsets
    adata.obs["pxl_row_in_fullres"] = registered_row / reference_hires_scale
    adata.obs["pxl_col_in_fullres"] = registered_col / reference_hires_scale
    del adata.uns["spatial"]
    adata.uns["manual_registration"] = {
        "reference": REFERENCE_ID,
        "row_offset_hires_pixels": row_offset,
        "col_offset_hires_pixels": col_offset,
        "rotation_degrees": rotation_degrees,
        "reference_tissue_hires_scalef": reference_hires_scale,
    }


def add_nearest_reference_spot(
    adata: ad.AnnData,
    registered_col: np.ndarray,
    registered_row: np.ndarray,
    reference_adata: ad.AnnData,
    reference_col: np.ndarray,
    reference_row: np.ndarray,
    spot_side_pixels: float,
) -> None:
    reference_coordinates = np.column_stack((reference_col, reference_row))
    moving_coordinates = np.column_stack((registered_col, registered_row))
    nearest_indices = cKDTree(reference_coordinates).query(
        moving_coordinates,
        k=1,
    )[1]
    coordinate_differences = np.abs(
        moving_coordinates - reference_coordinates[nearest_indices]
    )
    overlaps = np.all(coordinate_differences < spot_side_pixels, axis=1)
    overlapping_indices = nearest_indices[overlaps]
    if np.unique(overlapping_indices).size != overlapping_indices.size:
        raise ValueError(
            "Multiple moving spots overlap the same TAPS-beta spot after registration"
        )

    reference_spots = reference_adata.obs_names.to_numpy(dtype=str)
    match_codes = np.full(adata.n_obs, -1, dtype=int)
    match_codes[overlaps] = overlapping_indices
    adata.obs["nearest_taps_beta_spot"] = pd.Categorical.from_codes(
        match_codes,
        categories=reference_spots,
    )
    nearest_array_rows = pd.array([pd.NA] * adata.n_obs, dtype="Int64")
    nearest_array_cols = pd.array([pd.NA] * adata.n_obs, dtype="Int64")
    nearest_array_rows[overlaps] = reference_adata.obs["array_row"].to_numpy(
        dtype=int
    )[overlapping_indices]
    nearest_array_cols[overlaps] = reference_adata.obs["array_col"].to_numpy(
        dtype=int
    )[overlapping_indices]
    adata.obs["nearest_taps_beta_array_row"] = nearest_array_rows
    adata.obs["nearest_taps_beta_array_col"] = nearest_array_cols


def write_mudata(mdata: md.MuData, output_path: Path) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        print(f"Writing {output_path} ...", flush=True)
        mdata.write_h5mu(temporary_path, compression="gzip")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def plot_clusters(
    path: Path,
    layer: dict[str, object],
    reference: dict[str, object],
    registered_col: np.ndarray,
    registered_row: np.ndarray,
    spot_side_um: float,
) -> None:
    image = reference["image"]
    spot_side_pixels = spot_side_um / reference["pixelSizeUm"]
    half_side = spot_side_pixels / 2
    squares = [
        Rectangle(
            (col - half_side, row - half_side),
            spot_side_pixels,
            spot_side_pixels,
        )
        for col, row in zip(registered_col, registered_row, strict=True)
    ]
    colors = layer["clusterColors"]
    cluster_codes = layer["clusterCodes"]

    figure_width = 5
    figure_height = 6
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    axis.imshow(image, cmap="gray", origin="upper")
    axis.add_collection(
        PatchCollection(
            squares,
            facecolors=[colors[code] for code in cluster_codes],
            edgecolors="none",
            alpha=0.88,
        )
    )
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.set_aspect("equal")
    axis.set_title(
        f"{layer['label']} clusters in TAPS-beta hires space",
        fontsize=12,
    )
    axis.legend(
        handles=[
            Patch(facecolor=color, edgecolor="none", label=label)
            for label, color in zip(
                layer["clusterLabels"],
                colors,
                strict=True,
            )
        ],
        title="Cluster",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=False,
    )
    axis.set_axis_off()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def plot_spot_overlap(
    path: Path,
    reference: dict[str, object],
    registered: dict[str, tuple[np.ndarray, np.ndarray]],
    spot_side_um: float,
) -> None:
    image = reference["image"]
    spot_side_pixels = spot_side_um / reference["pixelSizeUm"]
    half_side = spot_side_pixels / 2
    reference_col, reference_row = registered[REFERENCE_ID]

    panel_width = 6
    figure_height = 6
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(panel_width * 2, figure_height),
    )
    for axis, modality_id, label in zip(
        axes,
        ("mrna", "taps"),
        ("mRNA", "TAPS"),
        strict=True,
    ):
        moving_col, moving_row = registered[modality_id]
        reference_squares = [
            Rectangle(
                (col - half_side, row - half_side),
                spot_side_pixels,
                spot_side_pixels,
            )
            for col, row in zip(reference_col, reference_row, strict=True)
        ]
        moving_squares = [
            Rectangle(
                (col - half_side, row - half_side),
                spot_side_pixels,
                spot_side_pixels,
            )
            for col, row in zip(moving_col, moving_row, strict=True)
        ]

        axis.imshow(image, cmap="gray", origin="upper")
        axis.add_collection(
            PatchCollection(
                reference_squares,
                facecolor=REFERENCE_SPOT_COLOR,
                edgecolor="none",
                alpha=0.45,
            )
        )
        axis.add_collection(
            PatchCollection(
                moving_squares,
                facecolor=MOVING_SPOT_COLOR,
                edgecolor="none",
                alpha=0.45,
            )
        )
        axis.set_xlim(0, image.shape[1])
        axis.set_ylim(image.shape[0], 0)
        axis.set_aspect("equal")
        axis.set_title(f"{label} and TAPS-beta spot overlap", fontsize=12)
        axis.set_axis_off()

    figure.legend(
        handles=[
            Patch(
                facecolor=REFERENCE_SPOT_COLOR,
                edgecolor="none",
                alpha=0.65,
                label="TAPS-beta spots",
            ),
            Patch(
                facecolor=MOVING_SPOT_COLOR,
                edgecolor="none",
                alpha=0.65,
                label="mRNA / TAPS spots",
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    figure.subplots_adjust(top=0.9, wspace=0.03)
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"Error: output directory already exists: {output_dir}")

    try:
        loaded = {
            "mrna": read_modality(
                args.mrna.expanduser().resolve(),
                "mrna",
                "mRNA",
                "leiden",
                "M",
            ),
            "taps": read_modality(
                args.taps.expanduser().resolve(),
                "taps",
                "TAPS",
                "cluster",
                "T",
            ),
            REFERENCE_ID: read_modality(
                args.taps_beta.expanduser().resolve(),
                REFERENCE_ID,
                "TAPS-beta",
                "cluster",
                "B",
            ),
        }
        adatas = {
            modality_id: value[0] for modality_id, value in loaded.items()
        }
        layers = {
            modality_id: value[1] for modality_id, value in loaded.items()
        }
        offsets = {
            "mrna": read_offsets(args.mrna_offsets.expanduser().resolve(), "mrna"),
            "taps": read_offsets(args.taps_offsets.expanduser().resolve(), "taps"),
        }
        registered = {
            modality_id: register_coordinates(layers[modality_id], *offset)
            for modality_id, offset in offsets.items()
        }
        registered[REFERENCE_ID] = (
            layers[REFERENCE_ID]["x"],
            layers[REFERENCE_ID]["y"],
        )

        output_dir.mkdir(parents=True)
        reference_hires_scale = layers[REFERENCE_ID]["hiresScale"]
        spot_side_pixels = args.spot_side_um / layers[REFERENCE_ID]["pixelSizeUm"]
        for modality_id in ("mrna", "taps"):
            registered_col, registered_row = registered[modality_id]
            replace_spatial_coordinates(
                adatas[modality_id],
                registered_col,
                registered_row,
                reference_hires_scale,
                offsets[modality_id],
            )
            add_nearest_reference_spot(
                adatas[modality_id],
                registered_col,
                registered_row,
                adatas[REFERENCE_ID],
                registered[REFERENCE_ID][0],
                registered[REFERENCE_ID][1],
                spot_side_pixels,
            )
        adatas[REFERENCE_ID].uns["manual_registration"] = {
            "reference": REFERENCE_ID,
            "row_offset_hires_pixels": 0.0,
            "col_offset_hires_pixels": 0.0,
            "rotation_degrees": 0.0,
            "reference_tissue_hires_scalef": reference_hires_scale,
        }
        mdata = md.MuData(adatas)
        write_mudata(mdata, output_dir / "registered-multimodal.h5mu")

        for modality_id in ("mrna", "taps", REFERENCE_ID):
            registered_col, registered_row = registered[modality_id]
            file_id = modality_id.replace("_", "-")
            plot_clusters(
                output_dir / f"{file_id}-clusters-on-taps-beta.png",
                layers[modality_id],
                layers[REFERENCE_ID],
                registered_col,
                registered_row,
                args.spot_side_um,
            )
        plot_spot_overlap(
            output_dir / "spot-overlap-on-taps-beta.png",
            layers[REFERENCE_ID],
            registered,
            args.spot_side_um,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(f"Wrote registered H5MU and cluster plots to {output_dir}")


if __name__ == "__main__":
    main()
