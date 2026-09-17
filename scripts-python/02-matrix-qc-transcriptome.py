#!/usr/bin/env python3
"""Calculate transcriptome QC metrics, plot them, and filter spots."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import scanpy as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import in_tissue_mask, spatial_plot_data, write_anndata


HB_GENES_MOUSE = {
    "Hba-a1", "Hba-a2", "Hbb-bs", "Hbb-bt", "Hbb-b1", "Hbb-b2",
    "Hbb-y", "Hbb-bh1", "Hba-x", "Hbq1a", "Hbq1b",
}
S_GENES_MOUSE = {
    "Mcm5", "Pcna", "Tyms", "Fen1", "Mcm7", "Mcm4", "Rrm1", "Ung",
    "Gins2", "Mcm6", "Cdca7", "Dtl", "Prim1", "Uhrf1", "Cenpu", "Hells",
    "Rfc2", "Polr1b", "Nasp", "Rad51ap1", "Gmnn", "Wdr76", "Slbp", "Ccne2",
    "Ubr7", "Pold3", "Msh2", "Atad2", "Rad51", "Rrm2", "Cdc45", "Cdc6",
    "Exo1", "Tipin", "Dscc1", "Blm", "Casp8ap2", "Usp1", "Clspn", "Pola1",
    "Chaf1b", "Mrpl36", "E2f8",
}
G2M_GENES_MOUSE = {
    "Hmgb2", "Cdk1", "Nusap1", "Ube2c", "Birc5", "Tpx2", "Top2a", "Ndc80",
    "Cks2", "Nuf2", "Cks1b", "Mki67", "Tmpo", "Cenpf", "Tacc3", "Pimreg",
    "Smc4", "Ccnb2", "Ckap2l", "Ckap2", "Aurkb", "Bub1", "Kif11", "Anp32e",
    "Tubb4b", "Gtse1", "Kif20b", "Hjurp", "Cdca3", "Jpt1", "Cdc20", "Ttk",
    "Cdc25c", "Kif2c", "Rangap1", "Ncapd2", "Dlgap5", "Cdca2", "Cdca8",
    "Ect2", "Kif23", "Hmmr", "Aurka", "Psrc1", "Anln", "Lbr", "Ckap5",
    "Cenpe", "Ctcf", "Nek2", "G2e3", "Gas2l3", "Cbx5", "Cenpa",
}
QC_METRICS = (
    ("total_counts", "Total counts"),
    ("n_genes_by_counts", "Genes by counts"),
    ("pct_counts_mt", "Mito %"),
    ("pct_counts_hb", "HB %"),
    ("S_score", "S score"),
    ("G2M_score", "G2M score"),
)
QC_SPATIAL_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "qc_blue_white_red",
    ("#4DBBD5", "#FFFFFF", "#E64B35"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Transcriptome H5AD from 00-matrix-io-transcriptome.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output filtered H5AD; defaults to qc/<input-stem>-qc.h5ad.",
    )
    parser.add_argument(
        "--min-genes-per-spot",
        type=int,
        default=500,
        help="Minimum detected genes per in-tissue spot (default: 500).",
    )
    parser.add_argument(
        "--spot-size-um",
        type=float,
        default=50.0,
        help="Spatial spot-square side length in µm (default: 50).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for cell-cycle scoring (default: 42).",
    )
    args = parser.parse_args()
    if args.min_genes_per_spot < 1:
        parser.error("--min-genes-per-spot must be >= 1")
    if args.spot_size_um <= 0:
        parser.error("--spot-size-um must be > 0")
    return args


def gene_symbols(adata: ad.AnnData) -> np.ndarray:
    if "feature_name" in adata.var:
        return adata.var["feature_name"].astype(str).to_numpy()
    return adata.var_names.astype(str).to_numpy()


def add_qc_metrics(adata: ad.AnnData) -> None:
    """Calculate standard Scanpy QC metrics for spots and genes."""
    symbols = gene_symbols(adata)
    adata.var["mt"] = np.char.startswith(symbols.astype(str), "mt-")
    adata.var["hb"] = np.isin(symbols, list(HB_GENES_MOUSE))
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "hb"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )


def matching_feature_ids(adata: ad.AnnData, wanted_symbols: set[str]) -> list[str]:
    return [
        str(feature_id)
        for feature_id, symbol in zip(adata.var_names, gene_symbols(adata), strict=True)
        if symbol in wanted_symbols
    ]


def add_cell_cycle_scores(adata: ad.AnnData, random_seed: int) -> None:
    s_features = matching_feature_ids(adata, S_GENES_MOUSE)
    g2m_features = matching_feature_ids(adata, G2M_GENES_MOUSE)
    if s_features and g2m_features:
        sc.tl.score_genes_cell_cycle(
            adata,
            s_genes=s_features,
            g2m_genes=g2m_features,
            random_state=random_seed,
            use_raw=False,
        )
    else:
        print("Warning: cell-cycle genes unavailable; scores were set to zero")
        adata.obs["S_score"] = 0.0
        adata.obs["G2M_score"] = 0.0
        adata.obs["phase"] = "G1"


def spot_rectangles(x: np.ndarray, y: np.ndarray, side: float) -> list[Rectangle]:
    half = side / 2
    return [
        Rectangle((x_value - half, y_value - half), side, side)
        for x_value, y_value in zip(x, y, strict=True)
    ]


def plot_qc_violins(
    path: Path,
    adata: ad.AnnData,
    color: str = "#4DBBD5",
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(9, 6))
    for axis, (column, title) in zip(axes.flat, QC_METRICS, strict=True):
        values = adata.obs[column].to_numpy(dtype=float)
        violin = axis.violinplot(
            values[np.isfinite(values)], showmedians=True, widths=0.8
        )
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.9)
        for component in ("cbars", "cmins", "cmaxes", "cmedians"):
            violin[component].set_color("#2F5597")
        axis.set_title(title, fontsize=12)
        axis.set_xticks([])
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_qc_spatial(
    path: Path,
    adata: ad.AnnData,
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    spot_size_pixels: float,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    figure.subplots_adjust(
        left=0.03,
        right=0.94,
        bottom=0.05,
        top=0.94,
        wspace=0.40,
        hspace=0.28,
    )
    for axis, (column, title) in zip(axes.flat, QC_METRICS, strict=True):
        values = adata.obs[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        vmax = float(np.quantile(finite, 0.95)) if finite.size else 1.0
        vmin = float(finite.min()) if finite.size else 0.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        collection = PatchCollection(
            spot_rectangles(x, y, spot_size_pixels),
            cmap=QC_SPATIAL_CMAP,
            norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True),
            edgecolors="none",
        )
        collection.set_array(np.clip(values, vmin, vmax))
        axis.imshow(image, cmap="gray", origin="upper")
        axis.add_collection(collection)
        axis.set_xlim(0, image.shape[1])
        axis.set_ylim(image.shape[0], 0)
        axis.set_aspect("equal")
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_axis_off()
        divider = make_axes_locatable(axis)
        colorbar_axis = divider.append_axes("right", size="4%", pad="2%")
        figure.colorbar(collection, cax=colorbar_axis)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def filter_spots(adata: ad.AnnData, min_genes_per_spot: int) -> ad.AnnData:
    keep = adata.obs["n_genes_by_counts"].to_numpy() >= min_genes_per_spot
    if not keep.any():
        raise ValueError(
            "Spot filter removed every in-tissue spot at "
            f"n_genes_by_counts >= {min_genes_per_spot}"
        )
    return adata[keep].copy()


def validate_paths(input_path: Path, output_path: Path, artifact_dir: Path) -> None:
    if not input_path.is_file():
        raise ValueError(f"Input H5AD does not exist: {input_path}")
    if output_path.exists():
        raise ValueError(f"Output file already exists: {output_path}")
    if artifact_dir.exists():
        raise ValueError(f"Output artifact directory already exists: {artifact_dir}")


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else input_path.parent / "qc" / f"{input_path.stem}-qc.h5ad"
    )
    artifact_dir = output_path.parent / "plots"

    try:
        validate_paths(input_path, output_path, artifact_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir()

        adata = ad.read_h5ad(input_path)
        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError("The input H5AD matrix is empty")
        n_loaded = adata.n_obs
        tissue_mask = in_tissue_mask(adata)
        n_tissue = int(tissue_mask.sum())
        if n_tissue == 0:
            raise ValueError("The input H5AD contains no in-tissue spots")
        adata = adata[tissue_mask].copy()

        add_qc_metrics(adata)
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        add_cell_cycle_scores(adata, args.random_seed)

        image, x, y, pixel_size_um = spatial_plot_data(adata)
        spot_size_pixels = args.spot_size_um / pixel_size_um
        plot_qc_violins(artifact_dir / "qc-violin.png", adata)
        plot_qc_spatial(
            artifact_dir / "qc-spatial.png",
            adata,
            image,
            x,
            y,
            spot_size_pixels,
        )

        adata = filter_spots(adata, args.min_genes_per_spot)
        write_anndata(adata, output_path)
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(
        f"Wrote {output_path}: {n_loaded:,} loaded spots -> "
        f"{n_tissue:,} in-tissue spots -> {adata.n_obs:,} spots with "
        f"n_genes_by_counts >= {args.min_genes_per_spot:,}; retained "
        f"{adata.n_vars:,} genes"
    )
    print(f"QC plots: {artifact_dir}")


if __name__ == "__main__":
    main()
