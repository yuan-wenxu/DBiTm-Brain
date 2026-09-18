#!/usr/bin/env python3
"""Align shared registered spots and jointly cluster three modalities."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import anndata as ad
import matplotlib
import mudata as md
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
MODALITY_IDS = ("mrna", "taps", REFERENCE_ID)
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
        help="Registered H5MU from 05-apply-manual-registration-multimodal.py.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--n-pcs-per-modality",
        type=int,
        default=10,
        help="Maximum PCs contributed by each modality (default: 10).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=20,
        help="Neighbors used for joint Leiden clustering and UMAP (default: 20).",
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
    if args.n_pcs_per_modality < 1:
        parser.error("--n-pcs-per-modality must be at least 1")
    if args.n_neighbors < 2:
        parser.error("--n-neighbors must be at least 2")
    if args.resolution <= 0:
        parser.error("--resolution must be greater than zero")
    if not 0 <= args.umap_min_dist <= 1:
        parser.error("--umap-min-dist must be between 0 and 1")
    if args.spot_side_um <= 0:
        parser.error("--spot-side-um must be greater than zero")
    return args


def read_modalities(path: Path) -> dict[str, ad.AnnData]:
    if not path.is_file():
        raise ValueError(f"Input H5MU does not exist: {path}")
    print(f"Reading {path} ...", flush=True)
    mdata = md.read_h5mu(path)
    missing = [modality_id for modality_id in MODALITY_IDS if modality_id not in mdata.mod]
    if missing:
        raise ValueError("Missing modalities: " + ", ".join(missing))
    return {modality_id: mdata.mod[modality_id].copy() for modality_id in MODALITY_IDS}


def rename_to_reference_spots(
    adata: ad.AnnData,
    reference_adata: ad.AnnData,
    modality_id: str,
) -> ad.AnnData:
    """Keep matched spots and adopt their TAPS-beta barcode and array indices."""
    match_key = "nearest_taps_beta_spot"
    if match_key not in adata.obs:
        raise ValueError(
            f"{modality_id} is missing obs[{match_key!r}]; run stage 05 first"
        )
    matched = adata[adata.obs[match_key].notna().to_numpy()].copy()
    if matched.n_obs == 0:
        raise ValueError(f"No {modality_id} spots overlap TAPS-beta spots")

    reference_barcodes = np.asarray(matched.obs[match_key].astype(object), dtype=str)
    if len(reference_barcodes) != len(set(reference_barcodes)):
        raise ValueError(
            f"Multiple {modality_id} spots map to the same TAPS-beta barcode"
        )

    matched.obs["source_barcode"] = matched.obs_names.astype(str)
    for column in ("array_row", "array_col"):
        if column not in matched.obs or column not in reference_adata.obs:
            raise ValueError(f"Required spot index {column!r} is missing")
        matched.obs[f"source_{column}"] = matched.obs[column].to_numpy()
        matched.obs[column] = reference_adata.obs.loc[
            reference_barcodes,
            column,
        ].to_numpy()
    matched.obs_names = pd.Index(
        reference_barcodes,
        name=reference_adata.obs_names.name or "barcode",
    )
    print(
        f"  {modality_id}: {matched.n_obs:,}/{adata.n_obs:,} spots map to "
        "unique TAPS-beta spots",
        flush=True,
    )
    return matched


def retain_shared_spots(
    adatas: dict[str, ad.AnnData],
) -> dict[str, ad.AnnData]:
    reference_names = adatas[REFERENCE_ID].obs_names.astype(str)
    common = set(reference_names)
    for modality_id in ("mrna", "taps"):
        common.intersection_update(adatas[modality_id].obs_names.astype(str))
    ordered_common = [barcode for barcode in reference_names if barcode in common]
    if len(ordered_common) < 3:
        raise ValueError(
            "Fewer than three spots contain information in all three modalities"
        )

    aligned = {
        modality_id: adata[ordered_common].copy()
        for modality_id, adata in adatas.items()
    }
    expected = aligned[REFERENCE_ID].obs_names
    for modality_id, adata in aligned.items():
        if not adata.obs_names.equals(expected):
            raise ValueError(f"Could not align {modality_id} to TAPS-beta spot order")
    print(
        f"Retained {len(ordered_common):,} spots with all three modalities",
        flush=True,
    )
    return aligned


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


def build_joint_representation(
    adatas: dict[str, ad.AnnData],
    n_pcs_per_modality: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Concatenate equal-weight, standardized PCA blocks from each modality."""
    blocks = []
    component_counts = {}
    for modality_id in MODALITY_IDS:
        adata = adatas[modality_id]
        if "X_pca" not in adata.obsm:
            raise ValueError(
                f"{modality_id} is missing obsm['X_pca']; run stage 03 first"
            )
        scores = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
        if scores.ndim != 2 or scores.shape[0] != adata.n_obs:
            raise ValueError(f"{modality_id} obsm['X_pca'] has an invalid shape")
        if not np.isfinite(scores).all():
            raise ValueError(f"{modality_id} obsm['X_pca'] contains non-finite values")

        scores = scores[:, : min(n_pcs_per_modality, scores.shape[1])]
        centered = scores - scores.mean(axis=0)
        standard_deviations = centered.std(axis=0)
        variable = standard_deviations > np.finfo(np.float64).eps
        if not variable.any():
            raise ValueError(f"{modality_id} has no variable PCA components")
        standardized = centered[:, variable] / standard_deviations[variable]
        standardized /= np.sqrt(standardized.shape[1])
        blocks.append(standardized.astype(np.float32))
        component_counts[modality_id] = int(standardized.shape[1])
    return np.column_stack(blocks).astype(np.float32), component_counts


def cluster_joint_spots(
    adatas: dict[str, ad.AnnData],
    args: argparse.Namespace,
) -> tuple[ad.AnnData, dict[str, int], int]:
    representation, component_counts = build_joint_representation(
        adatas,
        args.n_pcs_per_modality,
    )
    joint = ad.AnnData(
        obs=pd.DataFrame(index=adatas[REFERENCE_ID].obs_names.copy())
    )
    joint.obsm["X_joint"] = representation
    n_neighbors = min(args.n_neighbors, joint.n_obs - 1)
    sc.pp.neighbors(
        joint,
        n_neighbors=n_neighbors,
        use_rep="X_joint",
        metric="euclidean",
        random_state=args.random_seed,
    )
    sc.tl.leiden(
        joint,
        resolution=args.resolution,
        adjacency=joint.obsp["connectivities"],
        flavor="igraph",
        n_iterations=2,
        directed=False,
        random_state=args.random_seed,
        key_added="joint_leiden",
    )
    cluster_ids = sorted(joint.obs["joint_leiden"].astype(int).unique())
    joint.obs["joint_leiden"] = pd.Categorical(
        joint.obs["joint_leiden"].astype(str),
        categories=[str(cluster_id) for cluster_id in cluster_ids],
        ordered=True,
    )
    joint.uns["joint_leiden_colors"] = np.asarray(
        cluster_colors(len(cluster_ids)),
        dtype=str,
    )
    sc.tl.umap(
        joint,
        min_dist=args.umap_min_dist,
        random_state=args.random_seed,
        init_pos="random",
    )
    print(
        f"Joint clustering: {len(cluster_ids)} clusters; {n_neighbors} neighbors; "
        f"PCA components {component_counts}",
        flush=True,
    )
    return joint, component_counts, n_neighbors


def build_output_mudata(
    adatas: dict[str, ad.AnnData],
    joint: ad.AnnData,
    component_counts: dict[str, int],
    n_neighbors: int,
    args: argparse.Namespace,
) -> md.MuData:
    clusters = joint.obs["joint_leiden"].copy()
    for adata in adatas.values():
        adata.obs["joint_leiden"] = clusters
    mdata = md.MuData(adatas)
    mdata.obs["joint_leiden"] = clusters
    mdata.obsm["X_joint"] = joint.obsm["X_joint"].copy()
    mdata.obsm["X_umap"] = joint.obsm["X_umap"].copy()
    mdata.obsp["distances"] = joint.obsp["distances"].copy()
    mdata.obsp["connectivities"] = joint.obsp["connectivities"].copy()
    mdata.uns["joint_leiden_colors"] = joint.uns["joint_leiden_colors"].copy()
    mdata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": dict(joint.uns["neighbors"]["params"]),
    }
    mdata.uns["joint_clustering"] = {
        "modalities": np.asarray(MODALITY_IDS, dtype=str),
        "representation": "standardized equal-weight PCA concatenation",
        "pca_components": component_counts,
        "n_neighbors": n_neighbors,
        "resolution": args.resolution,
        "umap_min_dist": args.umap_min_dist,
        "random_seed": args.random_seed,
    }
    return mdata


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
    clusters = joint.obs["joint_leiden"].cat.codes.to_numpy(dtype=int)
    colors = joint.uns["joint_leiden_colors"].tolist()
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
                str(cluster),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
    axis.set(
        title="Joint multimodal Leiden clusters",
        xlabel="UMAP1",
        ylabel="UMAP2",
    )
    axis.legend(
        handles=[
            Patch(facecolor=color, edgecolor="none", label=str(cluster))
            for cluster, color in enumerate(colors)
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
    clusters = joint.obs["joint_leiden"].cat.codes.to_numpy(dtype=int)
    colors = joint.uns["joint_leiden_colors"].tolist()

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
    axis.set_title("Joint multimodal clusters in TAPS-beta space", fontsize=12)
    axis.legend(
        handles=[
            Patch(facecolor=color, edgecolor="none", label=str(cluster))
            for cluster, color in enumerate(colors)
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
    adatas: dict[str, ad.AnnData],
    joint: ad.AnnData,
) -> None:
    reference = adatas[REFERENCE_ID]
    table = pd.DataFrame(
        {
            "taps_beta_barcode": reference.obs_names.astype(str),
            "array_row": reference.obs["array_row"].to_numpy(),
            "array_col": reference.obs["array_col"].to_numpy(),
            "mrna_source_barcode": adatas["mrna"].obs[
                "source_barcode"
            ].to_numpy(),
            "taps_source_barcode": adatas["taps"].obs[
                "source_barcode"
            ].to_numpy(),
            "joint_leiden": joint.obs["joint_leiden"].astype(str).to_numpy(),
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
        adatas = read_modalities(input_path)
        for modality_id in ("mrna", "taps"):
            adatas[modality_id] = rename_to_reference_spots(
                adatas[modality_id],
                adatas[REFERENCE_ID],
                modality_id,
            )
        adatas = retain_shared_spots(adatas)
        joint, component_counts, n_neighbors = cluster_joint_spots(adatas, args)
        mdata = build_output_mudata(
            adatas,
            joint,
            component_counts,
            n_neighbors,
            args,
        )

        output_dir.mkdir(parents=True)
        write_mudata(mdata, output_dir / "joint-clustered-multimodal.h5mu")
        write_shared_spot_table(output_dir / "shared-spots.csv", adatas, joint)
        plot_joint_umap(output_dir / "joint-umap.png", joint)
        plot_joint_spatial(
            output_dir / "joint-clusters-on-taps-beta.png",
            joint,
            adatas[REFERENCE_ID],
            args.spot_side_um,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(f"Wrote joint multimodal clustering results to {output_dir}")


if __name__ == "__main__":
    main()
