#!/usr/bin/env python3
"""Combine DBiT residual and methylation Matrix Market data in one AnnData."""

from __future__ import annotations

import argparse
import gzip
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse
from scipy.io import mmread


SOURCE_PIXEL_SIZE_UM = 0.294
HIRES_PIXEL_SIZE_UM = 5.88
IMAGE_SCALE_FACTOR = SOURCE_PIXEL_SIZE_UM / HIRES_PIXEL_SIZE_UM
MATRIX_DIRECTORIES = {
    "residuals": "mean_shrunken_residuals",
    "methylation": "methylation_fractions",
}
INPUT_FILENAMES = {
    "matrix": "matrix.mtx.gz",
    "features": "features.tsv.gz",
    "barcodes": "barcodes.tsv.gz",
    "positions": "tissue_positions.tsv.gz",
    "image": "tissue_raw_image.png",
}
POSITION_COLUMNS = (
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
)


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
            "methylation_fractions/."
        ),
    )
    return parser.parse_args()


def resolve_inputs(input_dir: Path) -> dict[str, dict[str, Path]]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    paths = {
        layer: {
            key: input_dir / directory / filename
            for key, filename in INPUT_FILENAMES.items()
        }
        for layer, directory in MATRIX_DIRECTORIES.items()
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


def read_nonempty_lines(path: Path) -> list[str]:
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        values = [line.rstrip("\r\n") for line in handle]
    if not values or any(not value for value in values):
        raise ValueError(f"Expected non-empty lines in {path}")
    return values


def read_features(path: Path) -> pd.DataFrame:
    features = pd.read_csv(
        path,
        sep="\t",
        header=None,
        dtype=str,
        compression="gzip",
        keep_default_na=False,
    )
    if features.empty:
        raise ValueError(f"No features found in {path}")
    if features.shape[1] != 3:
        raise ValueError(
            f"Expected exactly 3 feature columns in {path}, found {features.shape[1]}"
        )

    features.columns = ["feature_id", "feature_name", "feature_type"]
    if (features["feature_id"] == "").any():
        raise ValueError(f"Empty feature identifier in {path}")
    if features["feature_id"].duplicated().any():
        duplicate = features.loc[
            features["feature_id"].duplicated(), "feature_id"
        ].iloc[0]
        raise ValueError(f"Duplicate feature identifier in {path}: {duplicate}")

    features.index = pd.Index(features.pop("feature_id"), name="feature_id")
    return features


def read_positions(path: Path) -> pd.DataFrame:
    positions = pd.read_csv(path, sep="\t", compression="gzip", dtype=str)
    missing = [column for column in POSITION_COLUMNS if column not in positions.columns]
    if missing:
        raise ValueError(
            f"Missing position column(s) in {path}: {', '.join(missing)}"
        )

    integer_columns = POSITION_COLUMNS[1:]
    for column in integer_columns:
        try:
            positions[column] = pd.to_numeric(
                positions[column], errors="raise", downcast="integer"
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Position column {column!r} is not integer-valued") from error

    if positions["barcode"].duplicated().any():
        raise ValueError(f"Duplicate sequence barcode in {path}")
    if positions.duplicated(["array_row", "array_col"]).any():
        raise ValueError(f"Duplicate array_row/array_col pair in {path}")
    return positions


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

    coordinate_lookup = positions.set_index(["array_row", "array_col"], drop=False)
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


def read_image(path: Path) -> np.ndarray:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError(f"Expected an 8-bit grayscale image, found mode {image.mode}")
        target_size = tuple(
            max(1, round(length * IMAGE_SCALE_FACTOR)) for length in image.size
        )
        resized = image.resize(target_size, resample=Image.Resampling.LANCZOS)
        return np.asarray(resized)


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
    shared_paths = paths["residuals"]
    feature_frame = read_features(shared_paths["features"])
    matrix_barcodes = read_nonempty_lines(shared_paths["barcodes"])
    if len(matrix_barcodes) != len(set(matrix_barcodes)):
        raise ValueError(f"Duplicate matrix barcode in {shared_paths['barcodes']}")

    expected_shape = (len(feature_frame), len(matrix_barcodes))
    matrices = {
        layer: read_matrix(layer_paths["matrix"], expected_shape, layer)
        for layer, layer_paths in paths.items()
    }

    all_positions = read_positions(shared_paths["positions"])
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
    for column in (
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ):
        obs[column] = obs[column].astype(np.int32)

    print("Reading and downsampling grayscale image ...", file=sys.stderr, flush=True)
    image = read_image(shared_paths["image"])

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


def validate_output(output: Path) -> None:
    if output.exists():
        raise ValueError(f"Output file already exists: {output}.")


def write_anndata(adata: ad.AnnData, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        print(f"Writing {output} ...", file=sys.stderr, flush=True)
        adata.write_h5ad(temporary_path)
        temporary_path.replace(output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    args = parse_args()
    try:
        paths = resolve_inputs(args.input_dir)
        input_dir = args.input_dir.expanduser().resolve()
        output = (input_dir / f"{input_dir.name}.h5ad").resolve()
        library_id = input_dir.name
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
