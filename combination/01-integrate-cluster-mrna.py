#!/usr/bin/env python3
"""Integrate QC-filtered mRNA samples with Pearson-residual PCA and Harmony."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import anndata as ad
import harmonypy as hm
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from utils import spatial_plot_data, write_anndata


BARCODE_PREFIXES = {"p7-rep1": "p7", "p35-rep1": "p35", "p60-rep1": "p60"}
SAMPLE_COLORS = ("#E64B35", "#4DBBD5", "#7E57C2")
UMAP_POINT_SIZE = 3.0
UMAP_POINT_ALPHA = 0.7
CLUSTER_COLORS = (
    "#A73030", "#E64B35", "#2F5597", "#4DBBD5", "#CC79A7", "#7E57C2",
    "#C5A3E0", "#D6A500", "#FFE082", "#2CA02C", "#264653", "#E7298A",
    "#98DF8A", "#C49C94",
)
OUTPUT_NAMES = (
    "mrna-combined.harmony.clustered.h5ad", "spot_clusters.csv",
    "sample_cluster_counts.csv", "highly_variable_genes.csv",
    "plots",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", nargs="+", type=Path, required=True,
        help="QC H5AD input paths.",
    )
    parser.add_argument(
        "--sample-names", nargs="+",
        help="Unique sample labels in input order; defaults to QC filename stems.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Result directory.",
    )
    parser.add_argument("--min-spots-per-gene", type=int, default=3)
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--pearson-theta", type=float, default=100.0)
    parser.add_argument(
        "--harmony-theta", type=float, default=2.0,
        help="Harmony diversity penalty; larger values favor more mixing (default: 2).",
    )
    parser.add_argument(
        "--harmony-lambda", type=float, default=1.0,
        help="Fixed ridge penalty; larger values damp correction (default: 1).",
    )
    parser.add_argument("--harmony-max-iter", type=int, default=50)
    parser.add_argument("--harmony-max-kmeans-iter", type=int, default=20)
    parser.add_argument("--harmony-tolerance", type=float, default=1e-4)
    parser.add_argument("--harmony-cluster-tolerance", type=float, default=1e-5)
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--n-pcs", type=int, default=20)
    parser.add_argument("--resolution", type=float, default=0.4)
    parser.add_argument("--umap-min-dist", type=float, default=0.3)
    parser.add_argument("--spot-size-um", type=float, default=50.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-threads", type=int, default=4)
    args = parser.parse_args()
    for name in (
        "min_spots_per_gene", "n_pcs", "harmony_max_iter", "n_threads",
        "harmony_max_kmeans_iter",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    for name in ("n_top_genes", "n_components", "n_neighbors"):
        if getattr(args, name) < 2:
            parser.error(f"--{name.replace('_', '-')} must be at least 2")
    for name in (
        "pearson_theta", "harmony_lambda", "resolution", "spot_size_um",
        "harmony_tolerance", "harmony_cluster_tolerance",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not np.isfinite(args.harmony_theta) or args.harmony_theta < 0:
        parser.error("--harmony-theta must be finite and nonnegative")
    if not 0 <= args.umap_min_dist <= 1:
        parser.error("--umap-min-dist must be between 0 and 1")
    if len(args.inputs) < 2:
        parser.error("At least two input samples are required")
    if args.random_seed < 0:
        parser.error("--random-seed must be nonnegative")
    return args


def validate_paths(args: argparse.Namespace) -> tuple[list[Path], list[str], Path]:
    paths = [path.expanduser().resolve() for path in args.inputs]
    if len(set(paths)) != len(paths):
        raise ValueError("Input paths must be distinct")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Input H5AD does not exist: {path}")
    names = args.sample_names or [
        path.name.removesuffix(".qc.h5ad").replace("-mrna-", "-")
        for path in paths
    ]
    if len(names) != len(paths) or len(set(names)) != len(names):
        raise ValueError("Provide one unique sample name per input")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in names):
        raise ValueError("Sample names must use letters, numbers, dots, hyphens or underscores")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output directory is a file: {output_dir}")
    for name in OUTPUT_NAMES:
        if (output_dir / name).exists():
            raise ValueError(f"Output already exists; choose a new directory: {output_dir / name}")
    return paths, names, output_dir


def validated_counts(adata: ad.AnnData, path: Path) -> sparse.csr_matrix:
    if "counts" not in adata.layers:
        raise ValueError(f"{path}: missing layers['counts']; use the QC output")
    counts = sparse.csr_matrix(adata.layers["counts"], dtype=np.float32)
    values = counts.data
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{path}: counts must be finite and nonnegative")
    if not np.allclose(values, np.rint(values), rtol=0, atol=1e-5):
        raise ValueError(f"{path}: layers['counts'] must contain integer counts")
    if np.any(np.asarray(counts.sum(axis=1)).ravel() <= 0):
        raise ValueError(f"{path}: zero-count spots are present")
    return counts


def combine_samples(
    paths: list[Path], names: list[str], min_spots: int,
) -> ad.AnnData:
    samples = []
    spatial_libraries = {}
    for path, name in zip(paths, names, strict=True):
        print(f"Reading {name}: {path}", flush=True)
        source = ad.read_h5ad(path)
        if not source.obs_names.is_unique or not source.var_names.is_unique:
            raise ValueError(f"{path}: duplicate spot or gene identifiers")
        if source.n_obs < 3 or source.n_vars < 2:
            raise ValueError(f"{path}: at least three spots and two genes are required")
        counts = validated_counts(source, path)
        # Retain gene annotations, but not single-sample gene QC statistics.
        columns = [
            column for column in ("feature_name", "feature_type", "mt", "hb")
            if column in source.var
        ]
        sample = ad.AnnData(X=counts, obs=source.obs.copy(), var=source.var[columns].copy())
        sample.obs["original_barcode"] = source.obs_names.astype(str)
        prefix = BARCODE_PREFIXES.get(name, name)
        sample.obs_names = prefix + "_" + source.obs_names.astype(str)
        sample.obs["barcode"] = sample.obs_names.astype(str)
        sample.obs["sample"] = name
        if "spatial" in source.obsm:
            sample.obsm["spatial"] = source.obsm["spatial"].copy()
        libraries = source.uns.get("spatial", {})
        if libraries:
            if len(libraries) != 1:
                raise ValueError(f"{path}: expected one tissue library per sample")
            spatial_libraries[name] = next(iter(libraries.values()))
        samples.append(sample)
        del source

    common_genes = samples[0].var_names
    for sample in samples[1:]:
        common_genes = common_genes.intersection(sample.var_names, sort=False)
    if len(common_genes) < 2:
        raise ValueError("Fewer than two shared gene IDs; align gene identifiers first")
    aligned = [sample[:, common_genes].copy() for sample in samples]
    for column in ("feature_name", "feature_type"):
        reference = aligned[0].var.get(column)
        if reference is not None:
            for sample in aligned[1:]:
                if column in sample.var and not reference.equals(sample.var[column]):
                    raise ValueError(f"Conflicting {column} annotations for shared genes")
    combined = ad.concat(
        aligned, join="inner", merge="same", label="sample", keys=names,
    )
    if not combined.obs_names.is_unique:
        raise ValueError("Combined spot IDs are not unique; rename sample labels")
    combined.obs["sample"] = pd.Categorical(combined.obs["sample"], categories=names)
    combined.uns["spatial"] = spatial_libraries
    detected = np.asarray((combined.X > 0).sum(axis=0)).ravel()
    keep = detected >= min_spots
    if keep.sum() < 2:
        raise ValueError("Fewer than two genes survive --min-spots-per-gene")
    combined = combined[:, keep].copy()
    if np.any(np.asarray(combined.X.sum(axis=1)).ravel() <= 0):
        raise ValueError("Some spots have no counts after shared-gene filtering")
    combined.layers["counts"] = combined.X.copy()
    combined.var["n_cells_by_counts"] = detected[keep]
    combined.var["total_counts"] = np.asarray(combined.X.sum(axis=0)).ravel()
    combined.var["mean_counts"] = combined.var["total_counts"] / combined.n_obs
    combined.var["pct_dropout_by_counts"] = 100 * (1 - detected[keep] / combined.n_obs)
    print(
        f"Combined {combined.n_obs:,} spots; {len(common_genes):,} shared genes, "
        f"{combined.n_vars:,} retained after filtering", flush=True,
    )
    return combined


def compute_pca(adata: ad.AnnData, args: argparse.Namespace) -> None:
    # Keep raw counts in .X; Pearson residuals are computed only for PCA.
    sc.experimental.pp.highly_variable_genes(
        adata, layer="counts", flavor="pearson_residuals", batch_key="sample",
        n_top_genes=min(args.n_top_genes, adata.n_vars), theta=args.pearson_theta,
        chunksize=1000,
    )
    hvg = adata.var["highly_variable"].to_numpy(dtype=bool)
    if np.any(np.asarray(adata.layers["counts"][:, hvg].sum(axis=1)).ravel() <= 0):
        raise ValueError("Some spots have no HVG counts; increase --n-top-genes")
    working = ad.AnnData(X=adata.layers["counts"][:, hvg].copy())
    n_components = min(args.n_components, working.n_obs - 1, working.n_vars - 1)
    if n_components < 2:
        raise ValueError("Too few spots or HVGs for PCA")
    sc.experimental.pp.normalize_pearson_residuals(working, theta=args.pearson_theta)
    adata.uns["pearson_residuals_normalization"] = working.uns["pearson_residuals_normalization"]
    sc.pp.pca(working, n_comps=n_components, random_state=args.random_seed)
    adata.obsm["X_pca"] = working.obsm["X_pca"].astype(np.float32)
    loadings = np.zeros((adata.n_vars, n_components), dtype=np.float32)
    loadings[hvg] = working.varm["PCs"]
    adata.varm["PCs"] = loadings
    adata.uns["pca"] = working.uns["pca"]
    adata.uns["pca"]["params"]["use_highly_variable"] = True
    adata.uns["pca"]["params"]["input_transform"] = "pearson_residuals"


def integrate_and_cluster(adata: ad.AnnData, args: argparse.Namespace) -> None:
    n_pcs = min(args.n_pcs, adata.obsm["X_pca"].shape[1])
    n_neighbors = min(args.n_neighbors, adata.n_obs - 1)
    # Cap the soft-cluster count safely for small synthetic/sample datasets.
    n_clusters = max(2, min(100, round(adata.n_obs / 30)))
    harmony = hm.run_harmony(
        adata.obsm["X_pca"][:, :n_pcs], adata.obs, ["sample"],
        theta=args.harmony_theta, lamb=args.harmony_lambda, nclust=n_clusters,
        max_iter_harmony=args.harmony_max_iter, random_state=args.random_seed,
        max_iter_kmeans=args.harmony_max_kmeans_iter,
        epsilon_harmony=args.harmony_tolerance,
        epsilon_cluster=args.harmony_cluster_tolerance,
        ncores=args.n_threads, verbose=False,
    )
    corrected = np.asarray(harmony.Z_corr)
    # harmonypy releases have used both cells x PCs and PCs x cells.
    if corrected.shape == (n_pcs, adata.n_obs):
        corrected = corrected.T
    if corrected.shape != (adata.n_obs, n_pcs) or not np.isfinite(corrected).all():
        raise ValueError("Harmony returned an invalid corrected embedding")
    adata.obsm["X_pca_harmony"] = corrected.astype(np.float32)
    sc.pp.neighbors(
        adata, use_rep="X_pca", n_pcs=n_pcs, n_neighbors=n_neighbors,
        random_state=args.random_seed, key_added="uncorrected_neighbors",
    )
    sc.tl.umap(
        adata, neighbors_key="uncorrected_neighbors", min_dist=args.umap_min_dist,
        random_state=args.random_seed,
    )
    adata.obsm["X_umap_uncorrected"] = adata.obsm["X_umap"].copy()
    sc.pp.neighbors(
        adata, use_rep="X_pca_harmony", n_pcs=n_pcs, n_neighbors=n_neighbors,
        random_state=args.random_seed,
    )
    sc.tl.leiden(
        adata, resolution=args.resolution, flavor="igraph", directed=False,
        n_iterations=2, random_state=args.random_seed, key_added="leiden",
    )
    categories = sorted(adata.obs["leiden"].unique(), key=int)
    adata.obs["leiden"] = pd.Categorical(
        adata.obs["leiden"], categories=categories, ordered=True,
    )
    sc.tl.umap(adata, min_dist=args.umap_min_dist, random_state=args.random_seed)


def cluster_colors(n_clusters: int) -> list[str]:
    colors = list(CLUSTER_COLORS[:n_clusters])
    extra = n_clusters - len(colors)
    if extra > 0:
        colors.extend(
            matplotlib.colors.to_hex(plt.get_cmap("turbo")(index / max(1, extra - 1)))
            for index in range(extra)
        )
    return colors


def embedding_panel(
    axis: plt.Axes, embedding: np.ndarray, labels: pd.Series,
    categories: list[str], colors: list[str], title: str,
    *, alpha: float = 1.0, show_legend: bool = True,
    legend_title: str | None = None,
) -> None:
    codes = pd.Categorical(labels, categories=categories).codes
    if (codes < 0).any():
        raise ValueError("UMAP labels contain missing or unknown categories")
    # Interleave samples so the last sample does not cover the others.
    order = np.random.default_rng(42).permutation(len(embedding))
    point_colors = np.asarray(colors)[codes]
    axis.scatter(
        embedding[order, 0], embedding[order, 1], s=UMAP_POINT_SIZE,
        c=point_colors[order], alpha=alpha, linewidths=0,
    )
    axis.set_title(title, fontsize=12)
    axis.set_xlabel("UMAP1", fontsize=10)
    axis.set_ylabel("UMAP2", fontsize=10)
    axis.tick_params(axis="both", labelsize=10)
    if show_legend:
        axis.legend(
            handles=[Line2D(
                [], [], marker="o", linestyle="none", markerfacecolor=color,
                markeredgecolor="none", markersize=5, label=label,
            ) for label, color in zip(categories, colors, strict=True)],
            title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left",
            frameon=False, fontsize=8,
        )


def sample_colors(n_samples: int) -> list[str]:
    palette = list(SAMPLE_COLORS)
    palette.extend(color for color in cluster_colors(n_samples) if color not in palette)
    return palette[:n_samples]


def plot_results(adata: ad.AnnData, output_dir: Path, args: argparse.Namespace) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir()
    names = list(adata.obs["sample"].cat.categories)
    clusters = list(adata.obs["leiden"].cat.categories)
    colors = cluster_colors(len(clusters))
    adata.uns["leiden_colors"] = np.asarray(colors, dtype=str)
    sample_palette = sample_colors(len(names))
    adata.uns["sample_colors"] = np.asarray(sample_palette, dtype=str)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    embedding_panel(
        axes[0], adata.obsm["X_umap_uncorrected"], adata.obs["sample"],
        names, sample_palette, "Before Harmony: sample", alpha=UMAP_POINT_ALPHA,
        show_legend=False,
    )
    embedding_panel(
        axes[1], adata.obsm["X_umap"], adata.obs["sample"],
        names, sample_palette, "After Harmony: sample", alpha=UMAP_POINT_ALPHA,
        legend_title="Sample",
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "umap-before-after.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    embedding_panel(
        axis, adata.obsm["X_umap"], adata.obs["leiden"],
        clusters, colors, "After Harmony: joint Leiden clusters",
        legend_title="Cluster",
    )
    figure.tight_layout()
    figure.savefig(plot_dir / "umap-clusters.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    for name in names:
        sample = adata[adata.obs["sample"] == name].copy()
        ids = sample.obs["leiden"].astype(int).to_numpy()
        facecolors = [colors[cluster] for cluster in ids]
        if {"array_row", "array_col"} <= set(sample.obs.columns):
            figure, axis = plt.subplots(figsize=(7, 6))
            rows = sample.obs["array_row"].to_numpy(dtype=float)
            columns = sample.obs["array_col"].to_numpy(dtype=float)
            axis.scatter(columns, rows, c=facecolors, marker="s", s=35, linewidths=0)
            axis.invert_xaxis()
            axis.invert_yaxis()
            axis.set_aspect("equal")
            axis.set_title(f"{name}: joint clusters", fontsize=12)
            axis.set_xlabel("Array column", fontsize=10)
            axis.set_ylabel("Array row", fontsize=10)
            axis.tick_params(axis="both", labelsize=10)
            add_cluster_legend(axis, clusters, colors)
            figure.tight_layout()
            figure.savefig(plot_dir / f"{name}.array-clusters.png", dpi=300, bbox_inches="tight")
            plt.close(figure)
        spatial_columns = {"pxl_row_in_fullres", "pxl_col_in_fullres"}
        if name in adata.uns["spatial"] and spatial_columns <= set(sample.obs.columns):
            sample.uns["spatial"] = {name: adata.uns["spatial"][name]}
            image, x, y, pixel_size_um = spatial_plot_data(sample)
            side = args.spot_size_um / pixel_size_um
            figure, axis = plt.subplots(figsize=(8, 7))
            axis.imshow(image, cmap="gray", origin="upper")
            axis.add_collection(PatchCollection(
                [Rectangle((xx - side / 2, yy - side / 2), side, side)
                 for xx, yy in zip(x, y, strict=True)],
                facecolors=facecolors, edgecolors="none",
            ))
            axis.set_xlim(0, image.shape[1])
            axis.set_ylim(image.shape[0], 0)
            axis.set_title(f"{name}: joint clusters on tissue", fontsize=12)
            axis.set_axis_off()
            add_cluster_legend(axis, clusters, colors)
            figure.tight_layout()
            figure.savefig(plot_dir / f"{name}.spatial-clusters.png", dpi=300, bbox_inches="tight")
            plt.close(figure)


def add_cluster_legend(axis: plt.Axes, clusters: list[str], colors: list[str]) -> None:
    axis.legend(
        handles=[Patch(facecolor=color, label=label)
                 for label, color in zip(clusters, colors, strict=True)],
        title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False,
    )


def main() -> None:
    args = parse_args()
    try:
        paths, names, output_dir = validate_paths(args)
        sc.settings.n_jobs = args.n_threads
        adata = combine_samples(paths, names, args.min_spots_per_gene)
        compute_pca(adata, args)
        integrate_and_cluster(adata, args)

        output_dir.mkdir(parents=True, exist_ok=True)
        plot_results(adata, output_dir, args)

        table = adata.obs.copy()
        table.index.name = "spot_id"
        table["color"] = [
            adata.uns["leiden_colors"][int(cluster)] for cluster in table["leiden"]
        ]
        table.to_csv(output_dir / "spot_clusters.csv")
        pd.crosstab(adata.obs["sample"], adata.obs["leiden"], dropna=False).to_csv(
            output_dir / "sample_cluster_counts.csv",
        )
        adata.var.loc[adata.var["highly_variable"]].to_csv(
            output_dir / "highly_variable_genes.csv", index_label="gene_id",
        )

        write_anndata(adata, output_dir / OUTPUT_NAMES[0])
        print(
            f"results in {output_dir}", flush=True,
        )
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
