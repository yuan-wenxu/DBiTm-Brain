#!/usr/bin/env python3
"""Smooth mRNA expression over nearby spots for visualization after clustering."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from utils import validate_output, write_anndata


SMOOTHED_LAYER = "expression_smoothed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Clustered mRNA H5AD from 03-cluster-transcriptome.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output H5AD; defaults to "
            "<input-parent-parent>/imputed/<input-stem>.imputed.h5ad."
        ),
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        help="Nearest spots to average (default: stage-03 neighbor count).",
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        help="PCA components used to find neighbors (default: stage-03 count).",
    )
    args = parser.parse_args()
    if args.n_neighbors is not None and args.n_neighbors < 1:
        parser.error("--n-neighbors must be at least 1")
    if args.n_pcs is not None and args.n_pcs < 1:
        parser.error("--n-pcs must be at least 1")
    return args


def smoothing_parameters(
    adata: ad.AnnData, n_neighbors: int | None, n_pcs: int | None
) -> tuple[np.ndarray, int, int]:
    """Use the same PCA space and neighbor count as stage-03 clustering."""
    if "leiden" not in adata.obs or "X_pca" not in adata.obsm:
        raise ValueError("Input needs stage-03 clusters and obsm['X_pca']")
    params = adata.uns.get("neighbors", {}).get("params", {})
    n_neighbors = n_neighbors if n_neighbors is not None else params.get("n_neighbors")
    n_pcs = n_pcs if n_pcs is not None else params.get("n_pcs")
    if n_neighbors is None or n_pcs is None:
        raise ValueError("Missing stage-03 neighbor settings; set --n-neighbors and --n-pcs")
    n_neighbors = int(n_neighbors)
    n_pcs = int(n_pcs)
    scores = np.asarray(adata.obsm["X_pca"])
    if scores.ndim != 2 or scores.shape[0] != adata.n_obs:
        raise ValueError("obsm['X_pca'] has an invalid shape")
    if not np.isfinite(scores).all():
        raise ValueError("obsm['X_pca'] contains non-finite values")
    if not 1 <= n_neighbors < adata.n_obs:
        raise ValueError("Neighbor count must be between 1 and the spot count minus 1")
    if not 1 <= n_pcs <= scores.shape[1]:
        raise ValueError("PCA component count exceeds available components")
    return scores[:, :n_pcs], n_neighbors, n_pcs


def smooth_expression(
    expression: sparse.spmatrix | np.ndarray,
    scores: np.ndarray,
    n_neighbors: int,
) -> sparse.csr_matrix | np.ndarray:
    """Average each spot's expression over its k nearest neighbors, excluding itself."""
    values = expression.data if sparse.issparse(expression) else np.asarray(expression)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("mRNA expression must contain finite nonnegative values")
    if expression.shape[0] != scores.shape[0]:
        raise ValueError("Expression and PCA spot counts do not match")

    model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    model.fit(scores)
    indices = model.kneighbors(scores, return_distance=False)
    neighbors = np.empty((scores.shape[0], n_neighbors), dtype=int)
    for spot, candidates in enumerate(indices):
        other_spots = candidates[candidates != spot]
        if len(other_spots) < n_neighbors:
            raise ValueError(f"Could not find {n_neighbors} neighbors for spot {spot}")
        neighbors[spot] = other_spots[:n_neighbors]
    rows = np.repeat(np.arange(scores.shape[0]), n_neighbors)
    weights = sparse.csr_matrix(
        (
            np.full(rows.size, 1.0 / n_neighbors, dtype=np.float32),
            (rows, neighbors.ravel()),
        ),
        shape=(scores.shape[0], scores.shape[0]),
    )
    smoothed = weights @ expression
    if sparse.issparse(smoothed):
        return smoothed.tocsr().astype(np.float32)
    return np.asarray(smoothed, dtype=np.float32)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else input_path.parent.parent / "imputed" / f"{input_path.stem}.imputed.h5ad"
    )
    try:
        if not input_path.is_file():
            raise ValueError(f"Input H5AD does not exist: {input_path}")
        validate_output(output_path)
        adata = ad.read_h5ad(input_path)
        if SMOOTHED_LAYER in adata.layers:
            raise ValueError(f"Input already contains layer {SMOOTHED_LAYER!r}")
        scores, n_neighbors, n_pcs = smoothing_parameters(
            adata, args.n_neighbors, args.n_pcs
        )
        adata.layers[SMOOTHED_LAYER] = smooth_expression(
            adata.X, scores, n_neighbors
        )
        adata.uns["expression_smoothing"] = {
            "method": "mean_of_pca_nearest_neighbors",
            "source": "X_log1p_counts_per_10k",
            "n_neighbors": n_neighbors,
            "n_pcs": n_pcs,
            "self_included": False,
            "use": "visualization",
        }
        write_anndata(adata, output_path)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(
        f"Wrote {output_path} with layer {SMOOTHED_LAYER!r}; "
        f"{n_neighbors} neighbors, {n_pcs} PCs"
    )


if __name__ == "__main__":
    main()
