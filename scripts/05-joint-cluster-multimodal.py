#!/usr/bin/env python3
"""Jointly cluster aligned mRNA and TAPS spots with a WNN graph."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import anndata as ad
import matplotlib
import mudata as md
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle


md.set_options(pull_on_update=False)

FIGURE_DPI = 300
REFERENCE_ID = "taps_beta"
JOINT_MODALITY_IDS = ("mrna", "taps")
OUTPUT_MODALITY_IDS = (*JOINT_MODALITY_IDS, REFERENCE_ID)
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
        help="Aligned H5MU from 04-apply-manual-registration-multimodal.py.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mrna-pcs",
        type=int,
        default=30,
        help="mRNA PCs used for WNN analysis (default: 30, as in the paper).",
    )
    parser.add_argument(
        "--taps-pcs",
        type=int,
        default=10,
        help="TAPS residual PCs used for WNN analysis (default: 10).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=20,
        help="Multimodal neighbors used for WNN, Leiden, and UMAP (default: 20).",
    )
    parser.add_argument(
        "--n-candidate-neighbors",
        type=int,
        default=200,
        help="Per-modality candidates considered by WNN (default: 200).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.4,
        help="Resolution for joint Leiden clustering (default: 0.4).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.3,
        help="Minimum distance for the joint UMAP (default: 0.3).",
    )
    parser.add_argument(
        "--spot-side-um",
        type=float,
        default=50.0,
        help="Spatial spot square side length in micrometers (default: 50).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for joint Leiden clustering and UMAP (default: 42).",
    )
    args = parser.parse_args()
    if args.mrna_pcs < 1:
        parser.error("--mrna-pcs must be at least 1")
    if args.taps_pcs < 1:
        parser.error("--taps-pcs must be at least 1")
    if args.n_neighbors < 2:
        parser.error("--n-neighbors must be at least 2")
    if args.n_candidate_neighbors < args.n_neighbors:
        parser.error("--n-candidate-neighbors must be at least --n-neighbors")
    if args.resolution <= 0:
        parser.error("--resolution must be greater than zero")
    if not 0 <= args.umap_min_dist <= 1:
        parser.error("--umap-min-dist must be between 0 and 1")
    if args.spot_side_um <= 0:
        parser.error("--spot-side-um must be greater than zero")
    return args


def read_mudata(path: Path) -> md.MuData:
    if not path.is_file():
        raise ValueError(f"Input H5MU does not exist: {path}")
    print(f"Reading {path} ...", flush=True)
    mdata = md.read_h5mu(path)
    missing = [
        modality_id
        for modality_id in OUTPUT_MODALITY_IDS
        if modality_id not in mdata.mod
    ]
    if missing:
        raise ValueError("Missing modalities: " + ", ".join(missing))
    expected = mdata.mod[REFERENCE_ID].obs_names
    if len(expected) < 3:
        raise ValueError("At least three aligned spots are required for WNN")
    for modality_id in OUTPUT_MODALITY_IDS:
        adata = mdata.mod[modality_id]
        if not adata.obs_names.equals(expected):
            raise ValueError(
                f"{modality_id} spots are not aligned to the TAPS-beta spot order; "
                "rerun stage 04"
            )
    print(
        f"Using {len(expected):,} aligned spots; WNN modalities: "
        + ", ".join(JOINT_MODALITY_IDS),
        flush=True,
    )
    return mdata


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


def l2_normalized_pca(
    adata: ad.AnnData,
    modality_id: str,
    n_pcs: int,
) -> np.ndarray:
    """Return the requested PCA block with Seurat-style row L2 normalization."""
    if "X_pca" not in adata.obsm:
        raise ValueError(
            f"{modality_id} is missing obsm['X_pca']; run stage 03 first"
        )
    scores = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != adata.n_obs:
        raise ValueError(f"{modality_id} obsm['X_pca'] has an invalid shape")
    if not np.isfinite(scores).all():
        raise ValueError(f"{modality_id} obsm['X_pca'] contains non-finite values")
    if scores.shape[1] < n_pcs:
        raise ValueError(
            f"{modality_id} has {scores.shape[1]} PCs, fewer than the "
            f"requested {n_pcs}"
        )

    scores = scores[:, :n_pcs]
    norms = np.linalg.norm(scores, axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError(f"{modality_id} PCA contains a zero-length spot embedding")
    return (scores / norms[:, None]).astype(np.float32)


def build_wnn_graph(
    mdata: md.MuData,
    args: argparse.Namespace,
) -> tuple[md.MuData, dict[str, int], int, int]:
    """Build a cell-weighted mRNA–TAPS nearest-neighbor graph."""
    component_counts = {
        "mrna": args.mrna_pcs,
        "taps": args.taps_pcs,
    }
    n_neighbors = min(args.n_neighbors, mdata.mod[REFERENCE_ID].n_obs - 1)
    n_candidates = min(
        args.n_candidate_neighbors,
        mdata.mod[REFERENCE_ID].n_obs - 1,
    )
    modality_adatas: dict[str, ad.AnnData] = {}
    for modality_id in JOINT_MODALITY_IDS:
        representation = l2_normalized_pca(
            mdata.mod[modality_id],
            modality_id,
            component_counts[modality_id],
        )
        modality = ad.AnnData(
            obs=pd.DataFrame(index=mdata.mod[modality_id].obs_names.copy())
        )
        modality.obsm["X_wnn_pca"] = representation
        sc.pp.neighbors(
            modality,
            n_neighbors=n_neighbors,
            use_rep="X_wnn_pca",
            metric="euclidean",
            random_state=args.random_seed,
            key_added="modality_neighbors",
        )
        modality_adatas[modality_id] = modality

    wnn = md.MuData(modality_adatas)
    mu.pp.neighbors(
        wnn,
        n_neighbors=n_neighbors,
        n_bandwidth_neighbors=n_neighbors,
        n_multineighbors=n_candidates,
        neighbor_keys={
            modality_id: "modality_neighbors"
            for modality_id in JOINT_MODALITY_IDS
        },
        metric="euclidean",
        key_added="wnn",
        weight_key="wnn_weight",
        add_weights_to_modalities=True,
        random_state=args.random_seed,
    )
    return wnn, component_counts, n_neighbors, n_candidates


def cluster_joint_spots(
    mdata: md.MuData,
    args: argparse.Namespace,
) -> ad.AnnData:
    wnn, component_counts, n_neighbors, n_candidates = build_wnn_graph(
        mdata,
        args,
    )
    joint = ad.AnnData(
        obs=pd.DataFrame(index=mdata.mod[REFERENCE_ID].obs_names.copy())
    )
    joint.obsm["X_wnn_input"] = np.column_stack(
        [
            wnn.mod[modality_id].obsm["X_wnn_pca"]
            for modality_id in JOINT_MODALITY_IDS
        ]
    )
    for modality_id in JOINT_MODALITY_IDS:
        joint.obs[f"{modality_id}_wnn_weight"] = wnn.mod[modality_id].obs[
            "wnn_weight"
        ].to_numpy()
    joint.obsp["wnn_distances"] = wnn.obsp["wnn_distances"].copy()
    joint.obsp["wnn_connectivities"] = wnn.obsp[
        "wnn_connectivities"
    ].copy()
    neighbor_params = dict(wnn.uns["wnn"]["params"])
    neighbor_params["use_rep"] = "X_wnn_input"
    neighbor_params["n_pcs"] = None
    joint.uns["wnn"] = {
        "connectivities_key": "wnn_connectivities",
        "distances_key": "wnn_distances",
        "params": neighbor_params,
    }

    sc.tl.leiden(
        joint,
        resolution=args.resolution,
        adjacency=joint.obsp["wnn_connectivities"],
        flavor="igraph",
        n_iterations=2,
        directed=False,
        random_state=args.random_seed,
        key_added="wnn_leiden",
    )
    cluster_ids = sorted(joint.obs["wnn_leiden"].astype(int).unique())
    cluster_labels = [f"W{cluster_id}" for cluster_id in cluster_ids]
    joint.obs["wnn_leiden"] = pd.Categorical(
        [f"W{value}" for value in joint.obs["wnn_leiden"].astype(str)],
        categories=cluster_labels,
        ordered=True,
    )
    joint.uns["wnn_leiden_colors"] = np.asarray(
        cluster_colors(len(cluster_ids)),
        dtype=str,
    )
    sc.tl.umap(
        joint,
        neighbors_key="wnn",
        min_dist=args.umap_min_dist,
        random_state=args.random_seed,
        init_pos="random",
    )
    weight_ranges = {
        modality_id: (
            float(joint.obs[f"{modality_id}_wnn_weight"].min()),
            float(joint.obs[f"{modality_id}_wnn_weight"].max()),
        )
        for modality_id in JOINT_MODALITY_IDS
    }
    print(
        f"WNN clustering: {len(cluster_ids)} clusters; {n_neighbors} neighbors; "
        f"candidate neighbors {n_candidates}; PCA components {component_counts}; "
        f"weight ranges {weight_ranges}",
        flush=True,
    )
    return joint


def add_wnn_results(
    mdata: md.MuData,
    joint: ad.AnnData,
) -> None:
    clusters = joint.obs["wnn_leiden"].copy()
    for adata in mdata.mod.values():
        adata.obs["wnn_leiden"] = clusters
    for modality_id in JOINT_MODALITY_IDS:
        mdata.mod[modality_id].obs["wnn_weight"] = joint.obs[
            f"{modality_id}_wnn_weight"
        ].to_numpy()

    mdata.obs["wnn_leiden"] = clusters
    for modality_id in JOINT_MODALITY_IDS:
        mdata.obs[f"{modality_id}_wnn_weight"] = joint.obs[
            f"{modality_id}_wnn_weight"
        ].to_numpy()
    mdata.obsm["X_wnn_umap"] = joint.obsm["X_umap"].copy()
    mdata.obsm["X_wnn_input"] = joint.obsm["X_wnn_input"].copy()
    mdata.obsp["wnn_distances"] = joint.obsp["wnn_distances"].copy()
    mdata.obsp["wnn_connectivities"] = joint.obsp[
        "wnn_connectivities"
    ].copy()
    mdata.uns["wnn_leiden_colors"] = joint.uns[
        "wnn_leiden_colors"
    ].copy()


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


def plot_joint_umap(path: Path, joint: ad.AnnData) -> None:
    embedding = joint.obsm["X_umap"]
    labels = joint.obs["wnn_leiden"].cat.categories.tolist()
    clusters = joint.obs["wnn_leiden"].cat.codes.to_numpy(dtype=int)
    colors = joint.uns["wnn_leiden_colors"].tolist()
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[colors[cluster] for cluster in clusters],
        s=12,
        linewidths=0,
        alpha=0.85,
    )
    for cluster in range(len(colors)):
        points = embedding[clusters == cluster]
        if len(points):
            axis.text(
                *np.median(points, axis=0),
                labels[cluster],
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
    axis.set(
        title="mRNA–TAPS WNN Leiden clusters",
        xlabel="UMAP1",
        ylabel="UMAP2",
    )
    axis.legend(
        handles=[
            Patch(facecolor=color, edgecolor="none", label=label)
            for label, color in zip(labels, colors, strict=True)
        ],
        title="Cluster",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=False,
    )
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def plot_joint_spatial(
    path: Path,
    joint: ad.AnnData,
    reference: ad.AnnData,
    spot_side_um: float,
) -> None:
    library = next(iter(reference.uns["spatial"].values()))
    image = np.asarray(library["images"]["hires"])
    scale = float(library["scalefactors"]["tissue_hires_scalef"])
    pixel_size_um = float(library["metadata"]["hires_pixel_size_um"])
    columns = reference.obs["pxl_col_in_fullres"].to_numpy(dtype=float) * scale
    rows = reference.obs["pxl_row_in_fullres"].to_numpy(dtype=float) * scale
    side_pixels = spot_side_um / pixel_size_um
    half_side = side_pixels / 2
    squares = [
        Rectangle(
            (column - half_side, row - half_side),
            side_pixels,
            side_pixels,
        )
        for column, row in zip(columns, rows, strict=True)
    ]
    labels = joint.obs["wnn_leiden"].cat.categories.tolist()
    clusters = joint.obs["wnn_leiden"].cat.codes.to_numpy(dtype=int)
    colors = joint.uns["wnn_leiden_colors"].tolist()

    figure, axis = plt.subplots(figsize=(5, 6))
    axis.imshow(image, cmap="gray", origin="upper")
    axis.add_collection(
        PatchCollection(
            squares,
            facecolors=[colors[cluster] for cluster in clusters],
            edgecolors="none",
            alpha=0.88,
        )
    )
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.set_aspect("equal")
    axis.set_title("mRNA–TAPS WNN clusters in TAPS-beta space", fontsize=12)
    axis.legend(
        handles=[
            Patch(facecolor=color, edgecolor="none", label=label)
            for label, color in zip(labels, colors, strict=True)
        ],
        title="Cluster",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=False,
    )
    axis.set_axis_off()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def write_shared_spot_table(
    path: Path,
    mdata: md.MuData,
    joint: ad.AnnData,
) -> None:
    reference = mdata.mod[REFERENCE_ID]
    table = pd.DataFrame(
        {
            "taps_beta_barcode": reference.obs_names.astype(str),
            "array_row": reference.obs["array_row"].to_numpy(),
            "array_col": reference.obs["array_col"].to_numpy(),
            "mrna_source_barcode": mdata.mod["mrna"].obs[
                "source_barcode"
            ].to_numpy(),
            "taps_source_barcode": mdata.mod["taps"].obs[
                "source_barcode"
            ].to_numpy(),
            "mrna_wnn_weight": joint.obs["mrna_wnn_weight"].to_numpy(),
            "taps_wnn_weight": joint.obs["taps_wnn_weight"].to_numpy(),
            "wnn_leiden": joint.obs["wnn_leiden"].astype(str).to_numpy(),
        }
    )
    table.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"Error: output directory already exists: {output_dir}")

    try:
        mdata = read_mudata(input_path)
        joint = cluster_joint_spots(mdata, args)
        add_wnn_results(mdata, joint)

        output_dir.mkdir(parents=True)
        write_mudata(mdata, output_dir / "wnn-clustered-multimodal.h5mu")
        write_shared_spot_table(output_dir / "wnn-spots.csv", mdata, joint)
        plot_joint_umap(output_dir / "wnn-umap.png", joint)
        plot_joint_spatial(
            output_dir / "wnn-clusters-on-taps-beta.png",
            joint,
            mdata.mod[REFERENCE_ID],
            args.spot_side_um,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(f"Wrote mRNA–TAPS WNN clustering results to {output_dir}")


if __name__ == "__main__":
    main()
