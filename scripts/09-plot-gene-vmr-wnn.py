#!/usr/bin/env python3
"""Plot covered stage-08 VMRs and linked genes on UMAP and spatial maps."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import mudata as md
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt


md.set_options(pull_on_update=False)

METHYLATION_LAYER = "methylation_imputed"
EXPRESSION_LAYER = "expression_smoothed"
MIN_COVERAGE_FRACTION = 0.5
FIGURE_DPI = 300
RATE_COLORMAP = LinearSegmentedColormap.from_list(
    "expression_methylation", ("#4DBBD5", "#FFFFFF", "#E64B35")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Aligned multimodal H5MU.")
    parser.add_argument(
        "--markers", type=Path, required=True,
        help="heatmap-selected-markers-<cluster>.tsv written by stage 08.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory; defaults to spatial/ beside the marker table.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing plots and index."
    )
    return parser.parse_args()


def read_markers(path: Path) -> pd.DataFrame:
    markers = pd.read_csv(path, sep="\t")
    required = {
        "heatmap_row", "heatmap_label", "cluster", "modality", "marker_type",
        "feature_id", "gene_id", "gene_name",
    }
    missing = required.difference(markers.columns)
    if missing:
        raise ValueError("Stage-08 marker table is missing columns: " + ", ".join(sorted(missing)))
    selected = markers.loc[
        markers["marker_type"].eq("vmr")
        & markers["modality"].isin(("taps", "taps_beta"))
    ]
    if selected.empty:
        raise ValueError("Stage-08 marker table contains no TAPS or TAPS-beta VMRs")
    feature_ids = selected["feature_id"]
    if feature_ids.isna().any() or feature_ids.astype(str).str.strip().eq("").any():
        raise ValueError("Marker feature_id must be nonempty")
    return selected.reset_index(drop=True)


def dense_column(matrix: sparse.spmatrix | np.ndarray, index: int) -> np.ndarray:
    column = matrix[:, index]
    if sparse.issparse(column):
        column = column.toarray()
    return np.asarray(column, dtype=float).ravel()


def split_annotation(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def linked_genes(annotation: pd.Series) -> list[tuple[str, str]]:
    if "gene_id" not in annotation:
        return []
    gene_ids = split_annotation(annotation["gene_id"])
    gene_names = split_annotation(annotation.get("gene_name", pd.NA))
    return [
        (gene_id, gene_names[index] if index < len(gene_names) else gene_id)
        for index, gene_id in enumerate(gene_ids)
    ]


def array_squares(
    axis: plt.Axes, x: np.ndarray, y: np.ndarray, facecolors: np.ndarray | list[str]
) -> None:
    squares = [
        Rectangle((x_value - 0.44, y_value - 0.44), 0.88, 0.88)
        for x_value, y_value in zip(x, y, strict=True)
    ]
    axis.add_collection(PatchCollection(squares, facecolors=facecolors, edgecolors="none"))
    axis.set_xlim(float(x.min()) - 1, float(x.max()) + 1)
    axis.set_ylim(float(y.max()) + 1, float(y.min()) - 1)
    axis.set_aspect("equal")
    axis.set_axis_off()


def plot_marker_pair(
    path: Path,
    marker_id: str,
    modality: str,
    gene_name: str,
    methylation: np.ndarray,
    expression: np.ndarray,
    umap: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    figure = plt.figure(figsize=(5.0, 9.0), layout="constrained")
    grid = figure.add_gridspec(
        3, 2,
        height_ratios=(0.95, 1.6, 1.6),
        width_ratios=(1, 0.07),
        hspace=0.08,
        wspace=0.04,
    )
    umap_grid = grid[0, 0].subgridspec(1, 2, wspace=0.02)
    expression_umap_axis = figure.add_subplot(umap_grid[0, 0])
    methylation_umap_axis = figure.add_subplot(umap_grid[0, 1])
    expression_spatial_axis = figure.add_subplot(grid[1, 0])
    expression_color_axis = figure.add_subplot(grid[1, 1])
    methylation_spatial_axis = figure.add_subplot(grid[2, 0])
    methylation_color_axis = figure.add_subplot(grid[2, 1])
    methylation_norm = Normalize(0, 1)
    expression_norm = Normalize(0, max(1e-6, float(np.quantile(expression, 0.99))))
    methylation_cmap = RATE_COLORMAP.copy()
    methylation_cmap.set_under("#D9D9D9")
    methylation_for_plot = np.where(np.isfinite(methylation), methylation, -1)
    expression_umap_axis.scatter(
        umap[:, 0], umap[:, 1], c=expression, cmap=RATE_COLORMAP,
        norm=expression_norm, s=8, linewidths=0, rasterized=True,
    )
    methylation_umap_axis.scatter(
        umap[:, 0], umap[:, 1], c=methylation_for_plot, cmap=methylation_cmap,
        norm=methylation_norm, s=8, linewidths=0, rasterized=True,
    )
    for axis in (expression_umap_axis, methylation_umap_axis):
        axis.set_aspect("equal", adjustable="box")
        axis.set_axis_off()

    array_squares(
        expression_spatial_axis, x, y,
        RATE_COLORMAP(expression_norm(expression)),
    )
    array_squares(
        methylation_spatial_axis, x, y,
        methylation_cmap(methylation_norm(methylation_for_plot)),
    )
    figure.suptitle(f"{gene_name} · {modality.upper()} VMR\n{marker_id}", fontsize=12)
    expression_umap_axis.set_title("RNA", fontsize=10)
    methylation_umap_axis.set_title("DNAm", fontsize=10)
    expression_spatial_axis.set_title("RNA", fontsize=10)
    methylation_spatial_axis.set_title("DNAm", fontsize=10)
    for color_axis, norm, cmap, label in (
        (expression_color_axis, expression_norm, RATE_COLORMAP, "Smoothed log1p(CP10K)"),
        (methylation_color_axis, methylation_norm, methylation_cmap, "Methylation fraction"),
    ):
        figure.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=color_axis, orientation="vertical",
            label=label,
        )

    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    marker_path = args.markers.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else marker_path.parent / "spatial"
    )
    try:
        markers = read_markers(marker_path)
        print(f"Reading {input_path} ...", flush=True)
        mdata = md.read_h5mu(input_path)
        if "mrna" not in mdata.mod:
            raise ValueError("H5MU is missing the mrna modality")
        mrna = mdata.mod["mrna"]
        if EXPRESSION_LAYER not in mrna.layers:
            raise ValueError(f"mRNA is missing layer {EXPRESSION_LAYER!r}")
        if not mdata.obs_names.equals(mrna.obs_names):
            raise ValueError("MuData spots are not aligned with mRNA")
        if "X_wnn_umap" not in mdata.obsm:
            raise ValueError("H5MU is missing obsm['X_wnn_umap']")
        umap = np.asarray(mdata.obsm["X_wnn_umap"], dtype=float)
        if umap.shape != (mrna.n_obs, 2) or not np.isfinite(umap).all():
            raise ValueError("WNN UMAP must have two finite coordinates per spot")
        if not mrna.var_names.is_unique:
            raise ValueError("mRNA gene IDs are not unique")
        for coordinate in ("array_row", "array_col"):
            if coordinate not in mrna.obs:
                raise ValueError(f"mRNA is missing obs[{coordinate!r}]")
        x = -mrna.obs["array_col"].to_numpy(dtype=float)
        y = mrna.obs["array_row"].to_numpy(dtype=float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Array coordinates must be finite")

        for modality in markers["modality"].unique():
            if modality not in mdata.mod:
                raise ValueError(f"H5MU is missing the {modality} modality")
            adata = mdata.mod[modality]
            if not adata.obs_names.equals(mrna.obs_names):
                raise ValueError(f"{modality} spots are not aligned with mRNA")
            if not adata.var_names.is_unique:
                raise ValueError(f"{modality} VMR IDs are not unique")
            if METHYLATION_LAYER not in adata.layers:
                raise ValueError(f"{modality} is missing layer {METHYLATION_LAYER!r}")
            selected_ids = markers.loc[markers["modality"].eq(modality), "feature_id"].astype(str)
            absent = selected_ids[adata.var_names.get_indexer(selected_ids) < 0]
            if not absent.empty:
                raise ValueError(f"{modality} VMR is absent from H5MU: {absent.iloc[0]}")

        index_path = output_dir / "spatial-marker-index.tsv"
        expression_cache: dict[str, np.ndarray] = {}
        records: list[dict[str, object]] = []
        plot_jobs: list[tuple[Path, str, str, str, np.ndarray, np.ndarray]] = []
        for row_number, marker in markers.iterrows():
            modality = str(marker["modality"])
            marker_id = str(marker["feature_id"])
            cluster = str(marker["cluster"])
            adata = mdata.mod[modality]
            vmr_index = adata.var_names.get_loc(marker_id)
            methylation = dense_column(adata.layers[METHYLATION_LAYER], vmr_index)
            finite = np.isfinite(methylation)
            if np.isinf(methylation).any() or np.any(
                (methylation[finite] < 0) | (methylation[finite] > 1)
            ):
                raise ValueError(f"Invalid imputed methylation fractions for {modality} {marker_id}")
            covered_spots = int(finite.sum())
            coverage_fraction = covered_spots / adata.n_obs
            base_record = marker.to_dict()
            base_record.update(
                covered_spots=covered_spots,
                total_spots=adata.n_obs,
                coverage_fraction=coverage_fraction,
                coverage_source=METHYLATION_LAYER,
                linked_gene_id="",
                linked_gene_name="",
                image="",
            )
            if coverage_fraction < MIN_COVERAGE_FRACTION:
                records.append({**base_record, "status": "low_coverage"})
                continue

            genes = linked_genes(marker)
            if not genes:
                records.append({**base_record, "status": "no_linked_gene"})
                continue
            for gene_number, (gene_id, gene_name) in enumerate(genes, start=1):
                record = {
                    **base_record,
                    "linked_gene_id": gene_id,
                    "linked_gene_name": gene_name,
                }
                gene_index = mrna.var_names.get_indexer([gene_id])[0]
                if gene_index < 0:
                    records.append({**record, "status": "gene_not_in_mrna"})
                    continue
                if gene_id not in expression_cache:
                    values = dense_column(mrna.layers[EXPRESSION_LAYER], int(gene_index))
                    if not np.isfinite(values).all() or np.any(values < 0):
                        raise ValueError(f"Invalid smoothed expression for {gene_id}")
                    expression_cache[gene_id] = values
                safe_marker_id = re.sub(r"[^A-Za-z0-9._-]+", "_", marker_id)
                safe_gene_id = re.sub(r"[^A-Za-z0-9._-]+", "_", gene_id)
                plot_name = (
                    f"{row_number + 1:04d}-{modality}-{cluster}"
                    f"-row{int(marker['heatmap_row']):02d}-{safe_marker_id}"
                    f"-g{gene_number:02d}-{safe_gene_id}.png"
                )
                relative_path = Path(modality) / plot_name
                output_path = output_dir / relative_path
                plot_jobs.append(
                    (output_path, marker_id, modality, gene_name, methylation,
                     expression_cache[gene_id])
                )
                records.append({**record, "status": "plotted", "image": relative_path.as_posix()})

        existing = [path for path in (index_path, *(job[0] for job in plot_jobs)) if path.exists()]
        if existing and not args.overwrite:
            raise ValueError(f"Output already exists: {existing[0]}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for modality in markers["modality"].unique():
            (output_dir / modality).mkdir(parents=True, exist_ok=True)
        for output_path, marker_id, modality, gene_name, methylation, expression in plot_jobs:
            plot_marker_pair(
                output_path, marker_id, modality, gene_name,
                methylation, expression, umap, x, y,
            )
            print(f"Wrote {output_path}", flush=True)
        pd.DataFrame(records).to_csv(index_path, sep="\t", index=False)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(f"Plot index: {index_path}")


if __name__ == "__main__":
    main()
