#!/usr/bin/env python3
"""Read one DBiT transcriptome Matrix Market folder into AnnData."""

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
INPUT_FILENAMES = {
    "matrix": "matrix.mtx.gz",
    "features": "features.tsv.gz",
    "barcodes": "barcodes.tsv.gz",
    "positions": "tissue_positions.tsv.gz",
    "image": "tissue_raw_image.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one transcriptome sparse-matrix folder and write an h5ad "
            "containing the expression matrix, positions, and tissue image."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing matrix.mtx.gz, features.tsv.gz, "
            "barcodes.tsv.gz, tissue_positions.tsv.gz, and tissue_raw_image.png."
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
        default="mrna",
        help="Library identifier used as the key in adata.uns['spatial'].",
    )
    return parser.parse_args()


def resolve_inputs(input_dir: Path) -> dict[str, Path]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    paths = {key: input_dir / filename for key, filename in INPUT_FILENAMES.items()}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(
            "Missing required input file(s): " + ", ".join(str(path) for path in missing)
        )
    return paths


def match_positions(matrix_barcodes: list[str], positions: pd.DataFrame) -> pd.DataFrame:
    """Match matrix sequence barcodes directly to the position table."""
    sequence_lookup = positions.set_index("barcode", drop=False)
    missing = [
        barcode for barcode in matrix_barcodes if barcode not in sequence_lookup.index
    ]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(
            f"{len(missing)} matrix barcode(s) have no position: {preview}{suffix}"
        )

    matched = sequence_lookup.loc[matrix_barcodes].copy()
    matched.index = pd.Index(matrix_barcodes, name="matrix_barcode")
    return matched


def read_matrix(path: Path, expected_shape: tuple[int, int]) -> sparse.csr_matrix:
    print("Reading transcriptome sparse matrix ...", file=sys.stderr, flush=True)
    matrix = mmread(path, spmatrix=True)
    if not sparse.issparse(matrix):
        matrix = sparse.coo_matrix(matrix)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Transcriptome matrix shape {matrix.shape} does not match "
            f"{expected_shape[0]} features x {expected_shape[1]} barcodes"
        )
    matrix = matrix.transpose().tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    if not np.isfinite(matrix.data).all():
        raise ValueError("Transcriptome matrix contains non-finite values")
    return matrix


def build_anndata(paths: dict[str, Path], library_id: str) -> ad.AnnData:
    feature_frame = read_features(paths["features"])
    matrix_barcodes = read_nonempty_lines(paths["barcodes"])
    if len(matrix_barcodes) != len(set(matrix_barcodes)):
        raise ValueError(f"Duplicate matrix barcode in {paths['barcodes']}")

    matrix = read_matrix(
        paths["matrix"], (len(feature_frame), len(matrix_barcodes))
    )
    matched = match_positions(matrix_barcodes, read_positions(paths["positions"]))

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
    image = read_image(paths["image"], IMAGE_SCALE_FACTOR)

    adata = ad.AnnData(X=matrix, obs=obs, var=feature_frame)
    adata.uns["matrix_sources"] = {"X": "transcriptome"}
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
        f"Wrote {output} "
        f"({adata.n_obs:,} spots x {adata.n_vars:,} features; "
        f"X transcriptome; image {image.shape[1]:,} x {image.shape[0]:,})"
    )


if __name__ == "__main__":
    main()
