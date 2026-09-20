#!/usr/bin/env python3
"""Apply raw-image registration transforms and calculate spot overlaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle


REFERENCE_ID = "taps_beta"
POSITION_COLUMNS = (
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
)
REFERENCE_COLOR = "#4DBBD5"
MOVING_COLOR = "#E64B35"
FIGURE_DPI = 300
PLOT_MAX_IMAGE_EDGE = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrna-image", type=Path, required=True)
    parser.add_argument("--taps-image", type=Path, required=True)
    parser.add_argument(
        "--taps-beta-image",
        dest="taps_beta_image",
        type=Path,
        required=True,
    )
    parser.add_argument("--mrna-positions", type=Path, required=True)
    parser.add_argument("--taps-positions", type=Path, required=True)
    parser.add_argument(
        "--taps-beta-positions",
        dest="taps_beta_positions",
        type=Path,
        required=True,
    )
    parser.add_argument("--mrna-offsets", type=Path, required=True)
    parser.add_argument("--taps-offsets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--spot-side-um",
        type=float,
        default=50.0,
        help="Spot square side length in micrometers (default: 50).",
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=0.294,
        help="Full-resolution image pixel size in micrometers (default: 0.294).",
    )
    args = parser.parse_args()
    if args.spot_side_um <= 0:
        parser.error("--spot-side-um must be greater than zero")
    if args.pixel_size_um <= 0:
        parser.error("--pixel-size-um must be greater than zero")
    return args


def image_size(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise ValueError(f"Image does not exist: {path}")
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError(f"Expected grayscale image at {path}, found {image.mode}")
        return image.size


def read_positions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"Position table does not exist: {path}")
    positions = pd.read_csv(path, sep="\t", compression="infer", dtype=str)
    missing = [column for column in POSITION_COLUMNS if column not in positions]
    if missing:
        raise ValueError(f"Missing position columns in {path}: {', '.join(missing)}")
    positions = positions.loc[:, POSITION_COLUMNS].copy()
    for column in POSITION_COLUMNS[1:]:
        try:
            positions[column] = pd.to_numeric(positions[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Position column {column!r} is not numeric in {path}"
            ) from error
    if positions["barcode"].duplicated().any():
        raise ValueError(f"Duplicate barcode in {path}")
    if positions.duplicated(["array_row", "array_col"]).any():
        raise ValueError(f"Duplicate array position in {path}")
    positions = positions.loc[positions["in_tissue"] == 1].copy()
    if positions.empty:
        raise ValueError(f"No selected spots in {path}")
    return positions.reset_index(drop=True)


def read_offsets(path: Path, moving_id: str) -> tuple[float, float, float]:
    if not path.is_file():
        raise ValueError(f"Offset JSON does not exist: {path}")
    values = json.loads(path.read_text(encoding="utf-8"))[
        f"{moving_id}_to_{REFERENCE_ID}"
    ]
    required = (
        "row_offset_fullres_pixels",
        "col_offset_fullres_pixels",
        "rotation_degrees",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(
            f"Missing full-resolution offset fields in {path}: {', '.join(missing)}"
        )
    return tuple(float(values[key]) for key in required)


def register_coordinates(
    positions: pd.DataFrame,
    image_width: int,
    image_height: int,
    offsets: tuple[float, float, float],
) -> np.ndarray:
    row_offset, col_offset, rotation_degrees = offsets
    columns = positions["pxl_col_in_fullres"].to_numpy(dtype=float)
    rows = positions["pxl_row_in_fullres"].to_numpy(dtype=float)
    center_col = image_width / 2
    center_row = image_height / 2
    radians = np.deg2rad(rotation_degrees)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    centered_col = columns - center_col
    centered_row = rows - center_row
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
    return np.column_stack((registered_col, registered_row))


def spot_ids(rows: pd.Series, columns: pd.Series) -> pd.Series:
    """Format numeric array coordinates as the pipeline's row_column spot ID."""
    row_ids = rows.map(lambda value: f"{int(value):02d}")
    column_ids = columns.map(lambda value: f"{int(value):02d}")
    return row_ids + "_" + column_ids


def calculate_matches(
    moving: pd.DataFrame,
    registered_coordinates: np.ndarray,
    reference: pd.DataFrame,
    spot_side_pixels: float,
    moving_id: str,
) -> pd.DataFrame:
    reference_coordinates = reference[
        ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    ].to_numpy(dtype=float)
    distances, nearest_indices = cKDTree(reference_coordinates).query(
        registered_coordinates,
        k=1,
    )
    differences = registered_coordinates - reference_coordinates[nearest_indices]
    overlaps = np.all(np.abs(differences) < spot_side_pixels, axis=1)
    overlapping_reference_indices = nearest_indices[overlaps]
    duplicate_count = len(overlapping_reference_indices) - len(
        np.unique(overlapping_reference_indices)
    )
    if duplicate_count:
        raise ValueError(
            f"{duplicate_count} {moving_id} overlap assignment(s) target an "
            "already matched TAPS-beta spot; refine the registration"
        )

    nearest_reference = reference.iloc[nearest_indices].reset_index(drop=True)
    moving_spot_ids = spot_ids(moving["array_row"], moving["array_col"])
    reference_spot_ids = spot_ids(
        nearest_reference["array_row"],
        nearest_reference["array_col"],
    )
    result = pd.DataFrame(
        {
            "moving_barcode": moving["barcode"].to_numpy(),
            "moving_spot_id": moving_spot_ids.to_numpy(),
            "moving_array_row": moving["array_row"].map(
                lambda value: f"{int(value):02d}"
            ),
            "moving_array_col": moving["array_col"].map(
                lambda value: f"{int(value):02d}"
            ),
            "registered_col_fullres": registered_coordinates[:, 0],
            "registered_row_fullres": registered_coordinates[:, 1],
            "nearest_taps_beta_barcode": nearest_reference["barcode"].to_numpy(),
            "nearest_taps_beta_spot_id": reference_spot_ids.to_numpy(),
            "nearest_taps_beta_array_row": nearest_reference["array_row"].map(
                lambda value: f"{int(value):02d}"
            ),
            "nearest_taps_beta_array_col": nearest_reference["array_col"].map(
                lambda value: f"{int(value):02d}"
            ),
            "col_difference_fullres_pixels": differences[:, 0],
            "row_difference_fullres_pixels": differences[:, 1],
            "overlap": overlaps,
        }
    )
    return result


def spot_pair_table(matches: pd.DataFrame, moving_id: str) -> pd.DataFrame:
    """Return overlap-only spot pairs using MethSCAn-compatible cell IDs."""
    return matches.loc[
        matches["overlap"],
        ["moving_spot_id", "nearest_taps_beta_spot_id"],
    ].rename(
        columns={
            "moving_spot_id": f"{moving_id}_cell",
            "nearest_taps_beta_spot_id": "beta_cell",
        }
    ).reset_index(drop=True)


def three_modality_overlap(
    reference: pd.DataFrame,
    mrna_matches: pd.DataFrame,
    taps_matches: pd.DataFrame,
) -> pd.DataFrame:
    reference_table = reference[
        ["barcode", "array_row", "array_col"]
    ].rename(
        columns={
            "barcode": "taps_beta_barcode",
            "array_row": "taps_beta_array_row",
            "array_col": "taps_beta_array_col",
        }
    )
    for column in ("taps_beta_array_row", "taps_beta_array_col"):
        reference_table[column] = reference_table[column].map(
            lambda value: f"{int(value):02d}"
        )
    result = reference_table.merge(
        mrna_matches.loc[
            mrna_matches["overlap"],
            ["nearest_taps_beta_barcode", "moving_barcode"],
        ].rename(columns={"moving_barcode": "mrna_barcode"}),
        left_on="taps_beta_barcode",
        right_on="nearest_taps_beta_barcode",
        how="inner",
    ).drop(columns="nearest_taps_beta_barcode")
    result = result.merge(
        taps_matches.loc[
            taps_matches["overlap"],
            ["nearest_taps_beta_barcode", "moving_barcode"],
        ].rename(columns={"moving_barcode": "taps_barcode"}),
        left_on="taps_beta_barcode",
        right_on="nearest_taps_beta_barcode",
        how="inner",
    ).drop(columns="nearest_taps_beta_barcode")
    return result


def downsample_reference_image(
    path: Path,
    max_edge: int,
) -> tuple[np.ndarray, float, float]:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as source:
        width, height = source.size
        scale = min(1.0, max_edge / max(width, height))
        display = source.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            resample=Image.Resampling.LANCZOS,
            reducing_gap=3.0,
        )
    image = np.asarray(display)
    return image, image.shape[1] / width, image.shape[0] / height


def plot_overlap(
    path: Path,
    reference_image_path: Path,
    reference: pd.DataFrame,
    registered: dict[str, np.ndarray],
    spot_side_pixels: float,
    max_image_edge: int,
) -> None:
    image, x_scale, y_scale = downsample_reference_image(
        reference_image_path,
        max_image_edge,
    )
    reference_coordinates = reference[
        ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    ].to_numpy(dtype=float)
    side_x = spot_side_pixels * x_scale
    side_y = spot_side_pixels * y_scale

    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    for axis, modality_id, label in zip(
        axes,
        ("mrna", "taps"),
        ("mRNA", "TAPS"),
        strict=True,
    ):
        moving_coordinates = registered[modality_id]
        reference_squares = [
            Rectangle(
                (column * x_scale - side_x / 2, row * y_scale - side_y / 2),
                side_x,
                side_y,
            )
            for column, row in reference_coordinates
        ]
        moving_squares = [
            Rectangle(
                (column * x_scale - side_x / 2, row * y_scale - side_y / 2),
                side_x,
                side_y,
            )
            for column, row in moving_coordinates
        ]
        axis.imshow(image, cmap="gray", origin="upper")
        axis.add_collection(
            PatchCollection(
                reference_squares,
                facecolor=REFERENCE_COLOR,
                edgecolor="none",
                alpha=0.42,
            )
        )
        axis.add_collection(
            PatchCollection(
                moving_squares,
                facecolor=MOVING_COLOR,
                edgecolor="none",
                alpha=0.42,
            )
        )
        axis.set_xlim(0, image.shape[1])
        axis.set_ylim(image.shape[0], 0)
        axis.set_aspect("equal")
        axis.set_title(f"{label} and TAPS-beta spot overlap", fontsize=12)
        axis.set_axis_off()
    figure.legend(
        handles=[
            Patch(facecolor=REFERENCE_COLOR, edgecolor="none", label="TAPS-beta"),
            Patch(facecolor=MOVING_COLOR, edgecolor="none", label="mRNA / TAPS"),
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
        image_paths = {
            "mrna": args.mrna_image.expanduser().resolve(),
            "taps": args.taps_image.expanduser().resolve(),
            REFERENCE_ID: args.taps_beta_image.expanduser().resolve(),
        }
        position_paths = {
            "mrna": args.mrna_positions.expanduser().resolve(),
            "taps": args.taps_positions.expanduser().resolve(),
            REFERENCE_ID: args.taps_beta_positions.expanduser().resolve(),
        }
        positions = {
            modality_id: read_positions(path)
            for modality_id, path in position_paths.items()
        }
        image_sizes = {
            modality_id: image_size(path)
            for modality_id, path in image_paths.items()
        }
        offsets = {
            "mrna": read_offsets(args.mrna_offsets.expanduser().resolve(), "mrna"),
            "taps": read_offsets(args.taps_offsets.expanduser().resolve(), "taps"),
        }
        registered = {
            modality_id: register_coordinates(
                positions[modality_id],
                *image_sizes[modality_id],
                offsets[modality_id],
            )
            for modality_id in ("mrna", "taps")
        }
        spot_side_pixels = args.spot_side_um / args.pixel_size_um
        matches = {
            modality_id: calculate_matches(
                positions[modality_id],
                registered[modality_id],
                positions[REFERENCE_ID],
                spot_side_pixels,
                modality_id,
            )
            for modality_id in ("mrna", "taps")
        }
        shared = three_modality_overlap(
            positions[REFERENCE_ID],
            matches["mrna"],
            matches["taps"],
        )

        output_dir.mkdir(parents=True)
        for modality_id, table in matches.items():
            file_id = modality_id.replace("_", "-")
            table.to_csv(
                output_dir / f"{file_id}-to-taps-beta-nearest-spots.tsv",
                index=False,
                sep="\t",
            )
            spot_pair_table(table, modality_id).to_csv(
                output_dir / f"{file_id}-to-taps-beta-spot-pairs.tsv",
                index=False,
                sep="\t",
            )
        shared.to_csv(
            output_dir / "three-modality-overlap.tsv",
            index=False,
            sep="\t",
        )

        plot_overlap(
            output_dir / "spot-overlap-on-taps-beta.png",
            image_paths[REFERENCE_ID],
            positions[REFERENCE_ID],
            registered,
            spot_side_pixels,
            PLOT_MAX_IMAGE_EDGE,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(
        f"mRNA overlaps: {int(matches['mrna']['overlap'].sum()):,}; "
        f"TAPS overlaps: {int(matches['taps']['overlap'].sum()):,}; "
        f"three modalities: {len(shared):,}"
    )
    print(f"Wrote spot-overlap results to {output_dir}")


if __name__ == "__main__":
    main()
