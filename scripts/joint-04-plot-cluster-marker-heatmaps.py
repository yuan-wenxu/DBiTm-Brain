#!/usr/bin/env python3
"""Plot imputed marker expression and methylation for individual WNN spots."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import mudata as md
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt


md.set_options(pull_on_update=False)

FIGURE_DPI = 300
CLUSTER_KEY = "wnn_leiden"
SMOOTHED_EXPRESSION_LAYER = "expression_smoothed"
IMPUTED_METHYLATION_LAYER = "methylation_imputed"
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "marker_zscore",
    ("#4DBBD5", "#FFFFFF", "#E64B35"),
)
HEATMAP_CMAP.set_bad("#D9D9D9")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="WNN-clustered H5MU containing imputed mRNA, TAPS, and TAPS-beta.",
    )
    parser.add_argument(
        "--markers",
        type=Path,
        help="Marker TSV; defaults to <input-directory>/wnn-markers-by-modality.tsv.",
    )
    parser.add_argument(
        "--cluster",
        required=True,
        help="WNN cluster to plot, for example W0.",
    )
    parser.add_argument(
        "--top-markers",
        type=int,
        default=10,
        help="Up to this many mRNA genes and hypomethylated VMRs per modality (default: 10).",
    )
    parser.add_argument(
        "--zscore-limit",
        type=float,
        default=2.5,
        help="Symmetric heatmap color limit for row Z-scores (default: 2.5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to the input H5MU directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing heatmap files.",
    )
    args = parser.parse_args()
    if args.top_markers < 1:
        parser.error("--top-markers must be at least 1")
    if args.zscore_limit <= 0:
        parser.error("--zscore-limit must be greater than zero")
    return args


def read_top_markers(
    path: Path,
    cluster: str,
    top_markers: int,
) -> dict[str, pd.DataFrame]:
    markers = pd.read_csv(path, sep="\t")
    required = {
        "cluster",
        "modality",
        "marker_type",
        "rank",
        "feature_id",
        "feature_name",
        "direction",
    }
    missing = required.difference(markers.columns)
    if missing:
        raise ValueError("Marker table is missing columns: " + ", ".join(sorted(missing)))
    selected = markers.loc[markers["cluster"].astype(str).eq(cluster)]
    result = {}
    for modality, marker_type in (
        ("mrna", "gene"),
        ("taps", "vmr"),
        ("taps_beta", "vmr"),
    ):
        subset = selected.loc[
            selected["modality"].eq(modality)
            & selected["marker_type"].eq(marker_type)
        ]
        if marker_type == "vmr":
            subset = subset.loc[subset["direction"].eq("hypomethylated")]
        subset = subset.sort_values("rank").head(top_markers)
        if subset.empty:
            raise ValueError(
                f"No {cluster} {modality} "
                f"{'hypomethylated VMR' if marker_type == 'vmr' else 'gene'} markers"
            )
        if len(subset) < top_markers:
            print(
                f"{cluster} {modality}: using all {len(subset)} available "
                f"markers (requested {top_markers})",
                flush=True,
            )
        result[modality] = subset
    return result


def row_zscores(values: np.ndarray) -> np.ndarray:
    """Standardize each feature across spots; keep absent features gray."""
    finite = np.isfinite(values)
    counts = finite.sum(axis=1, keepdims=True)
    means = np.divide(
        np.where(finite, values, 0).sum(axis=1, keepdims=True),
        counts,
        out=np.zeros((values.shape[0], 1), dtype=float),
        where=counts > 0,
    )
    deviations = np.where(finite, values - means, 0)
    variance = np.divide(
        (deviations * deviations).sum(axis=1, keepdims=True),
        counts,
        out=np.zeros_like(means),
        where=counts > 0,
    )
    standard_deviations = np.sqrt(variance)
    zscores = np.full(values.shape, np.nan, dtype=float)
    variable = finite & (standard_deviations > np.finfo(float).eps)
    np.divide(values - means, standard_deviations, out=zscores, where=variable)
    zscores[finite & ~variable] = 0.0
    return zscores


def selected_values(
    matrix: sparse.spmatrix | np.ndarray,
    feature_indices: np.ndarray,
    spot_order: np.ndarray,
) -> np.ndarray:
    selected = matrix[spot_order][:, feature_indices]
    if sparse.issparse(selected):
        selected = selected.toarray()
    return np.asarray(selected, dtype=float).T


def expression_matrix(adata) -> sparse.spmatrix | np.ndarray:
    if SMOOTHED_EXPRESSION_LAYER not in adata.layers:
        raise ValueError(f"mRNA is missing layer {SMOOTHED_EXPRESSION_LAYER!r}")
    return adata.layers[SMOOTHED_EXPRESSION_LAYER]


def imputed_methylation_matrix(adata, modality: str) -> np.ndarray:
    if IMPUTED_METHYLATION_LAYER not in adata.layers:
        raise ValueError(
            f"{modality} is missing layer {IMPUTED_METHYLATION_LAYER!r}"
        )
    return adata.layers[IMPUTED_METHYLATION_LAYER]


def gene_heatmap_data(
    adata,
    markers: pd.DataFrame,
    spot_order: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    feature_ids = markers["feature_id"].astype(str).tolist()
    feature_indices = adata.var_names.get_indexer(feature_ids)
    if np.any(feature_indices < 0):
        raise ValueError("A selected marker gene is absent from mRNA var_names")
    values = selected_values(expression_matrix(adata), feature_indices, spot_order)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Smoothed mRNA expression must be finite and nonnegative")
    labels = markers["feature_name"].astype(str).tolist()
    duplicated = pd.Index(labels).duplicated(keep=False)
    labels = [
        f"{label} [{feature_id}]" if duplicate else label
        for label, feature_id, duplicate in zip(
            labels,
            feature_ids,
            duplicated,
            strict=True,
        )
    ]
    return row_zscores(values), labels


def vmr_heatmap_data(
    adata,
    markers: pd.DataFrame,
    spot_order: np.ndarray,
    modality: str,
) -> tuple[np.ndarray, list[str]]:
    feature_ids = markers["feature_id"].astype(str).tolist()
    feature_indices = adata.var_names.get_indexer(feature_ids)
    if np.any(feature_indices < 0):
        raise ValueError(f"A selected marker VMR is absent from {modality} var_names")
    methylation = selected_values(
        imputed_methylation_matrix(adata, modality), feature_indices, spot_order
    )
    if not np.isfinite(methylation).all() or np.any((methylation < 0) | (methylation > 1)):
        raise ValueError(f"Imputed {modality} methylation rates must be finite fractions")
    return row_zscores(methylation), feature_ids


def cluster_assignments(adata) -> tuple[np.ndarray, list[str]]:
    if CLUSTER_KEY not in adata.obs:
        raise ValueError(f"mRNA is missing obs[{CLUSTER_KEY!r}]")
    clusters = adata.obs[CLUSTER_KEY]
    if clusters.isna().any():
        raise ValueError(f"mRNA obs[{CLUSTER_KEY!r}] contains missing values")
    labels = clusters.astype(str)
    observed = set(labels)
    if isinstance(clusters.dtype, pd.CategoricalDtype):
        cluster_names = [
            str(value) for value in clusters.cat.categories if str(value) in observed
        ]
    else:
        cluster_names = sorted(observed)
    category = pd.Categorical(labels, categories=cluster_names, ordered=True)
    cluster_codes = category.codes
    if np.any(cluster_codes < 0):
        raise ValueError("Failed to assign an order to all WNN clusters")
    return cluster_codes, cluster_names


def cluster_colors_from_uns(mdata: md.MuData, cluster_names: list[str]) -> list[str]:
    color_key = f"{CLUSTER_KEY}_colors"
    if color_key not in mdata.uns:
        raise ValueError(f"MuData is missing uns[{color_key!r}]")
    if CLUSTER_KEY not in mdata.obs:
        raise ValueError(f"MuData is missing obs[{CLUSTER_KEY!r}]")
    clusters = mdata.obs[CLUSTER_KEY]
    if not isinstance(clusters.dtype, pd.CategoricalDtype):
        raise ValueError(f"MuData obs[{CLUSTER_KEY!r}] must be categorical")
    categories = [str(value) for value in clusters.cat.categories]
    colors = np.asarray(mdata.uns[color_key], dtype=str).tolist()
    if len(colors) != len(categories):
        raise ValueError(f"uns[{color_key!r}] must have one color per WNN cluster")
    color_by_cluster = dict(zip(categories, colors, strict=True))
    missing = set(cluster_names).difference(color_by_cluster)
    if missing:
        raise ValueError("Missing WNN cluster colors: " + ", ".join(sorted(missing)))
    return [color_by_cluster[name] for name in cluster_names]


def plot_heatmap(
    path: Path,
    values: np.ndarray,
    row_labels: list[str],
    cluster_names: list[str],
    cluster_colors: list[str],
    cluster_sizes: np.ndarray,
    title: str,
    zscore_limit: float,
) -> None:
    cluster_count = len(cluster_names)
    if len(cluster_colors) != cluster_count:
        raise ValueError("Cluster colors do not match the plotted clusters")
    if len(cluster_sizes) != cluster_count or np.any(cluster_sizes <= 0):
        raise ValueError("Cluster sizes do not match the plotted clusters")
    if values.shape != (len(row_labels), int(cluster_sizes.sum())):
        raise ValueError("Heatmap must contain one column per ordered spot")
    boundaries = np.concatenate(([0], np.cumsum(cluster_sizes)))
    centers = (boundaries[:-1] + boundaries[1:]) / 2
    spot_boundaries = np.arange(values.shape[1] + 1)
    spot_cluster_codes = np.repeat(np.arange(cluster_count), cluster_sizes)
    row_boundaries = np.arange(len(row_labels) + 1)

    figure_side = max(8, min(16, len(row_labels) * 0.55 + 2))
    figure = plt.figure(figsize=(figure_side, figure_side), constrained_layout=True)
    figure.suptitle(title, fontsize=12)
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(0.035, 1),
        width_ratios=(1, 0.035),
        hspace=0.02,
    )
    cluster_axis = figure.add_subplot(grid[0, 0])
    heatmap_axis = figure.add_subplot(grid[1, 0])
    color_axis = figure.add_subplot(grid[1, 1])

    cluster_axis.pcolormesh(
        spot_boundaries,
        (0, 1),
        spot_cluster_codes[None, :],
        cmap=ListedColormap(cluster_colors),
        vmin=-0.5,
        vmax=cluster_count - 0.5,
        shading="flat",
    )
    cluster_axis.set_xlim(boundaries[0], boundaries[-1])
    cluster_axis.set_yticks([])
    cluster_axis.set_xticks(centers, labels=cluster_names, fontsize=10)
    cluster_axis.xaxis.tick_top()
    cluster_axis.tick_params(
        axis="x",
        top=False,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        length=0,
        pad=1,
        labelsize=7,
    )
    plt.setp(
        cluster_axis.get_xticklabels(),
        rotation=60,
        ha="left",
        va="bottom",
        rotation_mode="anchor",
    )
    for spine in cluster_axis.spines.values():
        spine.set_visible(False)

    image = heatmap_axis.pcolormesh(
        spot_boundaries,
        row_boundaries,
        np.ma.masked_invalid(values),
        cmap=HEATMAP_CMAP,
        vmin=-zscore_limit,
        vmax=zscore_limit,
        shading="flat",
        linewidth=0,
        rasterized=True,
    )
    heatmap_axis.set_xlim(boundaries[0], boundaries[-1])
    heatmap_axis.set_ylim(len(row_labels), 0)
    heatmap_axis.set_yticks(
        np.arange(len(row_labels)) + 0.5,
        labels=row_labels,
        fontsize=8,
    )
    heatmap_axis.set_xticks([])
    heatmap_axis.set_xlabel(f"Individual spots (n={values.shape[1]:,})")
    heatmap_axis.tick_params(axis="both", length=0, pad=4)
    for boundary in boundaries[1:-1]:
        heatmap_axis.axvline(boundary, color="#333333", linewidth=0.7)
    for spine in heatmap_axis.spines.values():
        spine.set_visible(False)

    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("Row Z-score", fontsize=10)
    colorbar.set_ticks((-2, 0, 2))
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    marker_path = (
        args.markers.expanduser().resolve()
        if args.markers is not None
        else input_path.parent / "wnn-markers-by-modality.tsv"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_path.parent
    )
    cluster_file_id = re.sub(r"[^A-Za-z0-9._-]+", "_", args.cluster)
    suffix = f"-{cluster_file_id}"
    names = {
        "mrna": "mrna",
        "taps": "taps",
        "taps_beta": "taps-beta",
    }
    paths = {key: output_dir / f"{name}{suffix}.png" for key, name in names.items()}
    spot_order_path = output_dir / f"heatmap-spot-order{suffix}.tsv"
    selected_marker_path = output_dir / f"heatmap-selected-markers{suffix}.tsv"
    existing = [
        path
        for path in (*paths.values(), spot_order_path, selected_marker_path)
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise SystemExit(
            "Error: heatmap output already exists: "
            + ", ".join(str(path) for path in existing)
        )

    try:
        markers = read_top_markers(
            marker_path,
            args.cluster,
            args.top_markers,
        )
        print(f"Reading {input_path} ...", flush=True)
        mdata = md.read_h5mu(input_path)
        missing_modalities = {"mrna", "taps", "taps_beta"}.difference(mdata.mod)
        if missing_modalities:
            raise ValueError(
                "Missing modalities: " + ", ".join(sorted(missing_modalities))
            )
        mrna = mdata.mod["mrna"]
        taps = mdata.mod["taps"]
        taps_beta = mdata.mod["taps_beta"]
        if not mrna.obs_names.equals(taps.obs_names) or not mrna.obs_names.equals(
            taps_beta.obs_names
        ):
            raise ValueError("mRNA, TAPS, and TAPS-beta spot orders are not aligned")
        for modality, adata in (("TAPS", taps), ("TAPS-beta", taps_beta)):
            if CLUSTER_KEY not in adata.obs or not adata.obs[CLUSTER_KEY].astype(str).equals(
                mrna.obs[CLUSTER_KEY].astype(str)
            ):
                raise ValueError(f"{modality} WNN clusters do not match mRNA")
        expression_matrix(mrna)
        imputed_methylation_matrix(taps, "TAPS")
        imputed_methylation_matrix(taps_beta, "TAPS-beta")
        cluster_codes, cluster_names = cluster_assignments(mrna)
        cluster_colors = cluster_colors_from_uns(mdata, cluster_names)
        spot_order = np.argsort(cluster_codes, kind="stable")
        cluster_sizes = np.bincount(cluster_codes, minlength=len(cluster_names))
        gene_zscores, gene_labels = gene_heatmap_data(
            mrna, markers["mrna"], spot_order
        )
        taps_zscores, taps_labels = vmr_heatmap_data(
            taps, markers["taps"], spot_order, "TAPS"
        )
        beta_zscores, beta_labels = vmr_heatmap_data(
            taps_beta, markers["taps_beta"], spot_order, "TAPS-beta"
        )

        heatmaps = (
            ("mrna", gene_zscores, gene_labels, "Smoothed mRNA marker genes"),
            ("taps", taps_zscores, taps_labels, "Imputed TAPS hypomethylated marker VMRs"),
            ("taps_beta", beta_zscores, beta_labels, "Imputed TAPS-beta hypomethylated marker VMRs"),
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        selected_marker_rows = []
        for key, values, labels, title in heatmaps:
            plot_heatmap(
                paths[key], values, labels, cluster_names, cluster_colors, cluster_sizes,
                title, args.zscore_limit,
            )
            selected = markers[key].copy()
            selected.insert(0, "heatmap_row", np.arange(1, len(labels) + 1))
            selected.insert(1, "heatmap_label", labels)
            selected_marker_rows.append(selected)
        pd.concat(selected_marker_rows, ignore_index=True).to_csv(
            selected_marker_path, sep="\t", index=False
        )
        pd.DataFrame(
            {
                "column": np.arange(1, len(spot_order) + 1),
                "spot": mrna.obs_names.to_numpy()[spot_order],
                "wnn_cluster": np.asarray(cluster_names)[cluster_codes[spot_order]],
            }
        ).to_csv(spot_order_path, sep="\t", index=False)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    for path in paths.values():
        print(f"Heatmap: {path}")
    print(f"Selected markers: {selected_marker_path}")
    print(f"Spot order: {spot_order_path}")


if __name__ == "__main__":
    main()
