#!/usr/bin/env python3
"""Cluster QC-filtered transcriptome spots with Pearson residuals and SNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from utils import spatial_plot_data, write_anndata


CLUSTER_COLORS = (
    "#A73030",
    "#E64B35",
    "#2F5597",
    "#4DBBD5",
    "#CC79A7",
    "#7E57C2",
    "#C5A3E0",
    "#D6A500",
    "#FFE082",
    "#2CA02C",
    "#264653",
    "#E7298A",
    "#98DF8A",
    "#C49C94",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="QC-filtered transcriptome H5AD from 02-matrix-qc-transcriptome.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output clustered H5AD; defaults to "
            "<input-parent-parent>/clustered/<input-stem>-clustered.h5ad."
        ),
    )
    parser.add_argument(
        "--min-spots-per-gene",
        type=int,
        default=3,
        help="Minimum spots in which a gene must be detected (default: 3).",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=2000,
        help="Pearson-residual highly variable genes (default: 2000).",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=50,
        help="Maximum number of Pearson-residual PCs (default: 50).",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=100.0,
        help="Negative-binomial theta for Pearson residuals (default: 100).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=10,
        help="Nearest neighbors used to construct the SNN graph (default: 10).",
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=10,
        help="PCs used to construct the SNN graph (default: 10).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.2,
        help="Leiden clustering resolution (default: 0.2).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.3,
        help="UMAP minimum distance (default: 0.3).",
    )
    parser.add_argument(
        "--spot-size-um",
        type=float,
        default=50.0,
        help="Side length of spatial spot squares in µm (default: 50).",
    )
    parser.add_argument(
        "--crop-margin-um",
        type=float,
        default=500.0,
        help="Margin around the spatial spot bounding box in µm (default: 500).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for PCA, Leiden, and UMAP (default: 42).",
    )
    args = parser.parse_args()
    if args.min_spots_per_gene < 1:
        parser.error("--min-spots-per-gene must be at least 1")
    if args.n_top_genes < 2:
        parser.error("--n-top-genes must be at least 2")
    if args.n_components < 2:
        parser.error("--n-components must be at least 2")
    if args.theta <= 0:
        parser.error("--theta must be greater than 0")
    if args.n_neighbors < 2:
        parser.error("--n-neighbors must be at least 2")
    if args.n_pcs < 1:
        parser.error("--n-pcs must be at least 1")
    if args.resolution <= 0:
        parser.error("--resolution must be greater than 0")
    if not 0 <= args.umap_min_dist <= 1:
        parser.error("--umap-min-dist must be between 0 and 1")
    if args.spot_size_um <= 0:
        parser.error("--spot-size-um must be greater than 0")
    if args.crop_margin_um < 0:
        parser.error("--crop-margin-um must be at least 0")
    return args


def validated_counts(adata: ad.AnnData) -> sparse.spmatrix | np.ndarray:
    if "counts" not in adata.layers:
        raise ValueError(
            "The input H5AD is missing layers['counts']; run "
            "02-matrix-qc-transcriptome.py first"
        )
    counts = adata.layers["counts"]
    values = counts.data if sparse.issparse(counts) else np.asarray(counts)
    if not np.isfinite(values).all():
        raise ValueError("layers['counts'] contains non-finite values")
    if np.any(values < 0):
        raise ValueError("layers['counts'] contains negative values")
    if not np.allclose(values, np.rint(values)):
        raise ValueError("layers['counts'] must contain integer counts")
    return counts


def filter_genes(adata: ad.AnnData, min_spots: int) -> ad.AnnData:
    counts = validated_counts(adata)
    detected = np.asarray((counts > 0).sum(axis=0)).ravel()
    keep = detected >= min_spots
    if not keep.any():
        raise ValueError(
            f"No genes remain after requiring detection in {min_spots} spots"
        )
    removed = int((~keep).sum())
    if removed:
        print(f"Removed {removed:,} genes detected in fewer than {min_spots} spots")
    return adata[:, keep].copy()


def pearson_residual_pca(
    adata: ad.AnnData,
    n_top_genes: int,
    n_components: int,
    theta: float,
    random_seed: int,
) -> tuple[ad.AnnData, int]:
    counts = validated_counts(adata)
    working = ad.AnnData(
        X=counts.copy(),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    n_top_genes = min(n_top_genes, working.n_vars)

    while True:
        sc.experimental.pp.highly_variable_genes(
            working,
            flavor="pearson_residuals",
            n_top_genes=n_top_genes,
            theta=theta,
            inplace=True,
        )
        hvg_mask = working.var["highly_variable"].to_numpy(dtype=bool)
        hvg_totals = np.asarray(working[:, hvg_mask].X.sum(axis=1)).ravel()
        keep_spots = hvg_totals > 0
        removed_spots = int((~keep_spots).sum())
        if removed_spots == 0:
            break
        print(
            f"Removed {removed_spots:,} spots with zero counts across selected "
            "highly variable genes"
        )
        adata = adata[keep_spots].copy()
        working = working[keep_spots].copy()

    n_components = min(
        n_components,
        working.n_obs - 1,
        n_top_genes - 1,
    )
    if n_components < 2:
        raise ValueError("Too few spots or highly variable genes for PCA")

    sc.experimental.pp.recipe_pearson_residuals(
        working,
        theta=theta,
        n_top_genes=n_top_genes,
        n_comps=n_components,
        random_state=random_seed,
        inplace=True,
    )

    hvg_columns = (
        "highly_variable",
        "means",
        "variances",
        "residual_variances",
        "highly_variable_rank",
    )
    for column in hvg_columns:
        if column in working.var:
            adata.var[column] = working.var[column].to_numpy()
    adata.obsm["X_pca"] = working.obsm["X_pca"].astype(np.float32)
    adata.varm["PCs"] = working.varm["PCs"].astype(np.float32)
    adata.uns["pca"] = working.uns["pca"]
    normalization = working.uns["pearson_residuals_normalization"]
    clip = normalization["clip"]
    adata.uns["pearson_residuals_normalization"] = {
        "theta": float(normalization["theta"]),
        "clip": float(np.sqrt(working.n_obs) if clip is None else clip),
        "n_top_genes": int(n_top_genes),
    }
    return adata, n_components


def build_snn_graph(
    adata: ad.AnnData,
    n_neighbors: int,
    n_pcs: int,
) -> tuple[int, int]:
    if "X_pca" not in adata.obsm:
        raise ValueError("PCA coordinates are missing from obsm['X_pca']")
    if adata.n_obs < 3:
        raise ValueError("At least three spots are required for clustering")

    n_neighbors = min(n_neighbors, adata.n_obs - 1)
    n_pcs = min(n_pcs, adata.obsm["X_pca"].shape[1])
    scores = adata.obsm["X_pca"][:, :n_pcs]
    model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    model.fit(scores)
    distances, indices = model.kneighbors(scores)
    neighbor_indices = indices[:, 1:]
    neighbor_distances = distances[:, 1:]

    rows = np.repeat(np.arange(adata.n_obs), n_neighbors)
    columns = neighbor_indices.ravel()
    knn_graph = sparse.csr_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, columns)),
        shape=(adata.n_obs, adata.n_obs),
    )
    shared = knn_graph @ knn_graph.T
    shared.setdiag(0)
    shared.eliminate_zeros()
    shared = shared.tocoo()
    weights = shared.data / (2 * n_neighbors - shared.data)
    connectivities = sparse.csr_matrix(
        (weights, (shared.row, shared.col)),
        shape=(adata.n_obs, adata.n_obs),
    )
    connectivities = connectivities.maximum(connectivities.T)
    distances_graph = sparse.csr_matrix(
        (neighbor_distances.ravel(), (rows, columns)),
        shape=(adata.n_obs, adata.n_obs),
    )
    distances_graph = distances_graph.maximum(distances_graph.T)

    adata.obsp["distances"] = distances_graph
    adata.obsp["connectivities"] = connectivities
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {
            "n_neighbors": int(n_neighbors),
            "n_pcs": int(n_pcs),
            "method": "umap",
            "metric": "euclidean",
            "graph_type": "snn",
            "snn_weight": "jaccard",
        },
    }
    return n_neighbors, n_pcs


def cluster_colors(cluster_count: int) -> list[str]:
    colors = list(CLUSTER_COLORS[:cluster_count])
    if cluster_count > len(colors):
        palette = plt.get_cmap("turbo")
        extra_count = cluster_count - len(colors)
        colors.extend(
            matplotlib.colors.to_hex(palette(index / max(1, extra_count - 1)))
            for index in range(extra_count)
        )
    return colors


def add_cluster_squares(
    axis: plt.Axes,
    clusters: np.ndarray,
    colors: list[str],
    x: np.ndarray,
    y: np.ndarray,
    side_length: float,
) -> None:
    half_size = side_length / 2
    squares = [
        Rectangle(
            (x_value - half_size, y_value - half_size),
            side_length,
            side_length,
        )
        for x_value, y_value in zip(x, y, strict=True)
    ]
    axis.add_collection(
        PatchCollection(
            squares,
            facecolors=[colors[cluster] for cluster in clusters],
            edgecolors="none",
        )
    )


def add_scale_bar(
    axis: plt.Axes,
    view_width_pixels: float,
    view_height_pixels: float,
    pixel_size_um: float,
) -> None:
    target_length_um = view_width_pixels * pixel_size_um / 5
    magnitude = 10 ** np.floor(np.log10(target_length_um))
    scale_bar_um = max(
        multiplier * magnitude
        for multiplier in (1, 2, 5)
        if multiplier * magnitude <= target_length_um
    )
    scale_bar = AnchoredSizeBar(
        axis.transData,
        scale_bar_um / pixel_size_um,
        f"{scale_bar_um:g} µm",
        loc="lower left",
        pad=0.4,
        borderpad=0.5,
        sep=3,
        frameon=True,
        size_vertical=max(2, view_height_pixels * 0.003),
        color="white",
    )
    scale_bar.patch.set_facecolor("black")
    scale_bar.patch.set_alpha(0.6)
    scale_bar.patch.set_edgecolor("none")
    axis.add_artist(scale_bar)


def cluster_legend(colors: list[str]) -> list[Patch]:
    return [
        Patch(facecolor=color, edgecolor="none", label=str(index))
        for index, color in enumerate(colors)
    ]


def plot_umap(
    path: Path,
    embedding: np.ndarray,
    clusters: np.ndarray,
    colors: list[str],
) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[colors[cluster] for cluster in clusters],
        s=13,
        linewidths=0,
    )
    for cluster in range(len(colors)):
        points = embedding[clusters == cluster]
        axis.text(
            np.median(points[:, 0]),
            np.median(points[:, 1]),
            str(cluster),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    axis.set(title="Transcriptome Leiden clusters", xlabel="UMAP1", ylabel="UMAP2")
    axis.legend(
        handles=cluster_legend(colors),
        title="Cluster",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_spatial(
    path: Path,
    clusters: np.ndarray,
    colors: list[str],
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    pixel_size_um: float,
    spot_size_um: float,
    crop_margin_um: float | None,
) -> None:
    spot_size_pixels = spot_size_um / pixel_size_um
    half_size = spot_size_pixels / 2
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(image, cmap="gray", origin="upper")
    add_cluster_squares(axis, clusters, colors, x, y, spot_size_pixels)

    if crop_margin_um is None:
        left, right = 0.0, float(image.shape[1])
        top, bottom = 0.0, float(image.shape[0])
        title = "Transcriptome clusters on tissue image"
    else:
        margin_pixels = crop_margin_um / pixel_size_um
        left = max(0.0, float(x.min()) - half_size - margin_pixels)
        right = min(float(image.shape[1]), float(x.max()) + half_size + margin_pixels)
        top = max(0.0, float(y.min()) - half_size - margin_pixels)
        bottom = min(float(image.shape[0]), float(y.max()) + half_size + margin_pixels)
        title = "Transcriptome clusters on cropped tissue image"

    axis.set_xlim(left, right)
    axis.set_ylim(bottom, top)
    axis.set_aspect("equal")
    axis.set_title(title, fontsize=12)
    add_scale_bar(axis, right - left, bottom - top, pixel_size_um)
    axis.legend(
        handles=cluster_legend(colors),
        title="Cluster",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_array_positions(
    path: Path,
    clusters: np.ndarray,
    colors: list[str],
    rows: np.ndarray,
    columns: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    add_cluster_squares(axis, clusters, colors, columns, rows, 0.8)
    padding = 0.75
    axis.set_xlim(columns.max() + padding, columns.min() - padding)
    axis.set_ylim(rows.max() + padding, rows.min() - padding)
    axis.set_aspect("equal")
    axis.set_title("Transcriptome clusters by array position", fontsize=12)
    axis.legend(
        handles=cluster_legend(colors),
        title="Cluster",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_spot_clusters(
    path: Path,
    adata: ad.AnnData,
    clusters: np.ndarray,
    colors: list[str],
) -> None:
    columns = [
        column
        for column in (
            "array_row",
            "array_col",
            "total_counts",
            "n_genes_by_counts",
        )
        if column in adata.obs
    ]
    table = adata.obs[columns].copy()
    table.insert(0, "barcode", adata.obs_names.astype(str))
    table["leiden"] = clusters
    table["color"] = [colors[cluster] for cluster in clusters]
    table.to_csv(path, index=False)


def validate_paths(input_path: Path, output_path: Path, plot_dir: Path) -> None:
    if not input_path.is_file():
        raise ValueError(f"Input H5AD does not exist: {input_path}")
    if output_path.exists():
        raise ValueError(f"Output file already exists: {output_path}")
    if plot_dir.exists():
        raise ValueError(f"Output plot directory already exists: {plot_dir}")


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else input_path.parent.parent
        / "clustered"
        / f"{input_path.stem}-clustered.h5ad"
    )
    plot_dir = output_path.parent / "plots"

    try:
        validate_paths(input_path, output_path, plot_dir)
        adata = ad.read_h5ad(input_path)
        if adata.n_obs < 3 or adata.n_vars < 2:
            raise ValueError("The input needs at least three spots and two genes")
        adata = filter_genes(adata, args.min_spots_per_gene)
        adata, n_components = pearson_residual_pca(
            adata,
            args.n_top_genes,
            args.n_components,
            args.theta,
            args.random_seed,
        )
        n_neighbors, n_pcs = build_snn_graph(
            adata,
            args.n_neighbors,
            args.n_pcs,
        )
        sc.tl.leiden(
            adata,
            resolution=args.resolution,
            adjacency=adata.obsp["connectivities"],
            flavor="igraph",
            n_iterations=2,
            directed=False,
            random_state=args.random_seed,
            key_added="leiden",
        )
        cluster_ids = sorted(adata.obs["leiden"].astype(int).unique())
        categories = [str(cluster) for cluster in cluster_ids]
        adata.obs["leiden"] = pd.Categorical(
            adata.obs["leiden"].astype(str),
            categories=categories,
            ordered=True,
        )
        clusters = adata.obs["leiden"].astype(int).to_numpy()
        colors = cluster_colors(len(cluster_ids))
        adata.uns["leiden_colors"] = np.asarray(colors, dtype=str)
        sc.tl.umap(
            adata,
            min_dist=args.umap_min_dist,
            random_state=args.random_seed,
        )

        image, x, y, pixel_size_um = spatial_plot_data(adata)
        rows = adata.obs["array_row"].to_numpy(dtype=float)
        columns = adata.obs["array_col"].to_numpy(dtype=float)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir()
        plot_umap(plot_dir / "umap.png", adata.obsm["X_umap"], clusters, colors)
        plot_spatial(
            plot_dir / "spatial.png",
            clusters,
            colors,
            image,
            x,
            y,
            pixel_size_um,
            args.spot_size_um,
            crop_margin_um=None,
        )
        plot_spatial(
            plot_dir / "spatial-cropped.png",
            clusters,
            colors,
            image,
            x,
            y,
            pixel_size_um,
            args.spot_size_um,
            crop_margin_um=args.crop_margin_um,
        )
        plot_array_positions(
            plot_dir / "array-positions.png",
            clusters,
            colors,
            rows,
            columns,
        )
        write_spot_clusters(plot_dir / "spot-clusters.csv", adata, clusters, colors)
        write_anndata(adata, output_path)
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    sizes = adata.obs["leiden"].value_counts().sort_index()
    print(
        f"Wrote {output_path}: {adata.n_obs:,} spots x {adata.n_vars:,} genes; "
        f"{len(sizes)} clusters"
    )
    print(
        f"Pearson residuals: {int(adata.var['highly_variable'].sum()):,} HVGs, "
        f"{n_components} PCs; SNN: {n_neighbors} neighbors, {n_pcs} PCs"
    )
    print("Cluster sizes: " + ", ".join(f"{key}={value}" for key, value in sizes.items()))
    print(f"Plots and spot assignments: {plot_dir}")


if __name__ == "__main__":
    main()
