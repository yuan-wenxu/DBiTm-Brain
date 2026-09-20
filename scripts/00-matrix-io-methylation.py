#!/usr/bin/env python3
"""Combine DBiT residual and methylation Matrix Market data in one AnnData."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from utils import (
    read_features,
    read_nonempty_lines,
    read_positions,
    read_image,
    validate_output,
    write_anndata)


SOURCE_PIXEL_SIZE_UM = 0.294
HIRES_PIXEL_SIZE_UM = 2.94
IMAGE_SCALE_FACTOR = SOURCE_PIXEL_SIZE_UM / HIRES_PIXEL_SIZE_UM
MATRIX_DIRECTORIES = {
    "residuals": "mean_shrunken_residuals",
    "methylation": "methylation_fractions",
}
MATRIX_FILENAMES = {
    "matrix": "matrix.mtx.gz",
    "features": "features.tsv.gz",
    "barcodes": "barcodes.tsv.gz",
}
SPATIAL_FILENAMES = {
    "positions": "tissue_positions.tsv.gz",
    "image": "tissue_raw_image.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read residual and methylation sparse-matrix folders and write one "
            "h5ad containing both matrices, shared positions, and a shared image."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing mean_shrunken_residuals/ and "
            "methylation_fractions/ matrix folders plus tissue_positions.tsv.gz "
            "and tissue_raw_image.png."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output h5ad path; the file must not already exist.",
    )
    parser.add_argument(
        "--library-id",
        required=True,
        help="Library identifier used as the key in adata.uns['spatial']." \
        "5mc, 5hmc, taps, and taps-beta libraries are all valid identifiers.",
    )
    return parser.parse_args()


def resolve_inputs(input_dir: Path) -> dict[str, dict[str, Path]]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    paths = {
        layer: {
            key: input_dir / directory / filename
            for key, filename in MATRIX_FILENAMES.items()
        }
        for layer, directory in MATRIX_DIRECTORIES.items()
    }
    paths["spatial"] = {
        key: input_dir / filename for key, filename in SPATIAL_FILENAMES.items()
    }
    missing = [
        path
        for layer_paths in paths.values()
        for path in layer_paths.values()
        if not path.is_file()
    ]
    if missing:
        raise ValueError(
            "Missing required input file(s): " + ", ".join(str(path) for path in missing)
        )
    return paths


def parse_matrix_barcode(barcode: str) -> tuple[int, int]:
    """Parse ``RRCC`` or ``RR_CC`` as fixed two-digit row/column indices."""
    if len(barcode) == 5 and barcode[2] == "_":
        digits = barcode[:2] + barcode[3:]
    elif len(barcode) == 4:
        digits = barcode
    else:
        digits = ""
    if len(digits) != 4 or not digits.isascii() or not digits.isdigit():
        raise ValueError(
            f"Invalid matrix barcode {barcode!r}; expected RRCC or RR_CC "
            "with two-digit row and column indices"
        )
    return int(digits[:2]), int(digits[2:])


def match_positions(matrix_barcodes: list[str], positions: pd.DataFrame) -> pd.DataFrame:
    """Match fixed-width numeric matrix barcodes to position row/column indices."""
    coordinates = [parse_matrix_barcode(barcode) for barcode in matrix_barcodes]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("Matrix barcodes contain duplicate row/column coordinates")

    coordinate_lookup = positions.copy()
    coordinate_lookup.index = pd.MultiIndex.from_arrays(
        [
            pd.to_numeric(coordinate_lookup["array_row"]),
            pd.to_numeric(coordinate_lookup["array_col"]),
        ],
        names=["array_row", "array_col"],
    )
    missing = [
        (barcode, coordinate)
        for barcode, coordinate in zip(matrix_barcodes, coordinates, strict=True)
        if coordinate not in coordinate_lookup.index
    ]
    if missing:
        preview = ", ".join(
            f"{barcode} -> ({row}, {column})"
            for barcode, (row, column) in missing[:5]
        )
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(
            f"{len(missing)} matrix barcode(s) have no position: {preview}{suffix}"
        )

    matched = coordinate_lookup.loc[coordinates].copy()
    matched.index = pd.Index(matrix_barcodes, name="matrix_barcode")
    return matched


def read_matrix(
    path: Path,
    expected_shape: tuple[int, int],
    layer: str,
) -> sparse.csr_matrix:
    print(f"Reading {layer} sparse matrix ...", file=sys.stderr, flush=True)
    matrix = mmread(path, spmatrix=True)
    if not sparse.issparse(matrix):
        matrix = sparse.coo_matrix(matrix)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"{layer} matrix shape {matrix.shape} does not match "
            f"{expected_shape[0]} features x {expected_shape[1]} barcodes"
        )
    matrix = matrix.astype(np.float32).transpose().tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"{layer} matrix contains non-finite values")
    return matrix


def build_anndata(
    paths: dict[str, dict[str, Path]], library_id: str
) -> ad.AnnData:
    residual_paths = paths["residuals"]
    feature_frame = read_features(residual_paths["features"])
    matrix_barcodes = read_nonempty_lines(residual_paths["barcodes"])
    if len(matrix_barcodes) != len(set(matrix_barcodes)):
        raise ValueError(f"Duplicate matrix barcode in {residual_paths['barcodes']}")

    expected_shape = (len(feature_frame), len(matrix_barcodes))
    matrices = {
        layer: read_matrix(paths[layer]["matrix"], expected_shape, layer)
        for layer in MATRIX_DIRECTORIES
    }

    spatial_paths = paths["spatial"]
    all_positions = read_positions(spatial_paths["positions"])
    matched = match_positions(matrix_barcodes, all_positions)

    obs = matched[
        [
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ]
    ].copy()
    obs.index = pd.Index(matched["barcode"], name="barcode")
    obs["in_tissue"] = obs["in_tissue"].astype(np.int8)
    for column in ("pxl_row_in_fullres", "pxl_col_in_fullres"):
        obs[column] = obs[column].astype(np.int32)

    print("Reading and downsampling grayscale image ...", file=sys.stderr, flush=True)
    image = read_image(spatial_paths["image"], IMAGE_SCALE_FACTOR)

    adata = ad.AnnData(
        X=matrices["residuals"],
        obs=obs,
        var=feature_frame,
        layers={"methylation": matrices["methylation"]},
    )
    adata.uns["matrix_sources"] = {
        "X": MATRIX_DIRECTORIES["residuals"],
        "methylation": MATRIX_DIRECTORIES["methylation"],
    }
    adata.uns["spatial"] = {
        library_id: {
            "images": {"hires": image},
            "scalefactors": {"tissue_hires_scalef": IMAGE_SCALE_FACTOR},
            "metadata": {
                "source_pixel_size_um": SOURCE_PIXEL_SIZE_UM,
                "hires_pixel_size_um": HIRES_PIXEL_SIZE_UM,
            },
        }
    }
    return adata


def main() -> None:
    args = parse_args()
    try:
        input_dir = args.input_dir.expanduser().resolve()
        paths = resolve_inputs(input_dir)
        output = args.output.expanduser().resolve()
        library_id = args.library_id.strip()
        if not library_id:
            raise ValueError("Library identifier must not be empty")
        validate_output(output)

        adata = build_anndata(paths, library_id)
        write_anndata(adata, output)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    image = adata.uns["spatial"][library_id]["images"]["hires"]
    print(
        f"Wrote {output.expanduser().resolve()} "
        f"({adata.n_obs:,} spots x {adata.n_vars:,} features; "
        f"X residuals; layer methylation; "
        f"image {image.shape[1]:,} x {image.shape[0]:,})"
    )


if __name__ == "__main__":
    main()
