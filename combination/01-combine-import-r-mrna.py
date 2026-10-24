#!/usr/bin/env python3
"""Concatenate QC mRNA H5ADs and import R UMAP, clusters and labels."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from utils import spatial_plot_data, write_anndata


BARCODE_PREFIXES = {"p7-rep1": "P7", "p35-rep1": "P35", "p60-rep1": "P60"}
CLUSTER_COLORS = (
    "#A73030", "#E64B35", "#2F5597", "#4DBBD5", "#CC79A7", "#7E57C2",
    "#C5A3E0", "#D6A500", "#FFE082", "#2CA02C", "#264653", "#E7298A",
    "#98DF8A", "#C49C94", "#00A087", "#7E8A97", "#B86B3E",
)
OUTPUT_NAMES = (
    "mrna-combined.r-annotated.h5ad", "spot_clusters.csv",
    "cluster_annotation.csv", "sample_cluster_counts.csv", "plots",
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
        "--spot-prefixes", nargs="+",
        help="R spot ID prefixes in input order; inferred from sample names.",
    )
    parser.add_argument(
        "--r-results-dir", type=Path, required=True,
        help="Folder containing the three R-exported CSV files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spot-size-um", type=float, default=50.0)
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error("At least two input samples are required")
    args.sample_names = args.sample_names or [
        path.name.removesuffix(".qc.h5ad").replace("-mrna-", "-")
        for path in args.inputs
    ]
    args.spot_prefixes = args.spot_prefixes or [
        BARCODE_PREFIXES.get(name, name) for name in args.sample_names
    ]
    if not np.isfinite(args.spot_size_um) or args.spot_size_um <= 0:
        parser.error("--spot-size-um must be finite and positive")
    for name in ("sample_names", "spot_prefixes"):
        values = getattr(args, name)
        if len(values) != len(args.inputs) or len(set(values)) != len(values):
            parser.error(
                f"--{name.replace('_', '-')} requires one unique value per input"
            )
        if any(
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)
            for value in values
        ):
            parser.error(
                f"--{name.replace('_', '-')} must be unique; use letters, "
                "numbers, dots, hyphens or underscores"
            )
    return args


def read_csv(path: Path, required: set[str], key: str) -> pd.DataFrame:
    table = pd.read_csv(
        path, dtype=str, keep_default_na=False, encoding="utf-8-sig",
    )
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if table.empty:
        raise ValueError(f"{path}: empty table")
    for column in required:
        table[column] = table[column].str.strip()
        if table[column].eq("").any():
            raise ValueError(f"{path}: empty values in {column}")
    if table[key].duplicated().any():
        raise ValueError(f"{path}: duplicate {key} values")
    return table.set_index(key)


def require_same_spots(
    expected: pd.Index, actual: pd.Index, description: str,
) -> None:
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(
            f"{description}: spot IDs differ; missing={len(missing)} "
            f"{missing[:3].tolist()}, extra={len(extra)} {extra[:3].tolist()}"
        )


def read_r_results(
    directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = read_csv(
        directory / "spot-metadata.csv",
        {"spot_id", "sample", "seurat_clusters"}, "spot_id",
    )
    umap = read_csv(
        directory / "umap-coordinates.csv",
        {"spot_id", "UMAP_1", "UMAP_2"}, "spot_id",
    )
    labels = read_csv(
        directory / "cluster-labels.csv", {"cluster", "label"}, "cluster",
    )
    require_same_spots(metadata.index, umap.index, "R metadata vs UMAP")
    for values in (metadata["seurat_clusters"], labels.index.to_series()):
        if not values.str.fullmatch(r"0|[1-9][0-9]*").all():
            raise ValueError("R cluster IDs must be nonnegative integers")
    missing = set(metadata["seurat_clusters"]).difference(labels.index)
    if missing:
        raise ValueError(f"Missing annotations for clusters: {sorted(missing)}")
    clusters = sorted(metadata["seurat_clusters"].unique(), key=int)
    labels = labels.loc[clusters, ["label"]].rename(
        columns={"label": "annotation"},
    )
    umap = umap.loc[metadata.index, ["UMAP_1", "UMAP_2"]].astype(float)
    if not np.isfinite(umap.to_numpy()).all():
        raise ValueError("R UMAP coordinates must be finite")
    return metadata, umap, labels


def combine_samples(args: argparse.Namespace) -> ad.AnnData:
    paths = [path.expanduser().resolve() for path in args.inputs]
    if len(set(paths)) != len(paths):
        raise ValueError("Input H5AD paths must be distinct")
    samples = []
    libraries = {}
    for path, name, prefix in zip(
        paths, args.sample_names, args.spot_prefixes, strict=True,
    ):
        print(f"Reading {name}: {path}", flush=True)
        source = ad.read_h5ad(path)
        if not source.obs_names.is_unique or not source.var_names.is_unique:
            raise ValueError(f"{path}: spot and gene IDs must be unique")
        if source.n_obs == 0 or source.n_vars == 0:
            raise ValueError(f"{path}: empty expression matrix")
        if "counts" not in source.layers:
            raise ValueError(f"{path}: missing layers['counts']")
        counts = sparse.csr_matrix(source.layers["counts"], copy=True)
        counts.sum_duplicates()
        if (not np.isfinite(counts.data).all()
                or (counts.data < 0).any()
                or not np.allclose(counts.data, np.rint(counts.data),
                                   rtol=0, atol=1e-5)):
            raise ValueError(f"{path}: counts must be nonnegative integers")
        # Keep expression and QC metadata; discard single-sample embeddings.
        var_columns = [
            column for column in ("feature_name", "feature_type", "mt", "hb")
            if column in source.var
        ]
        sample = ad.AnnData(
            X=counts, obs=source.obs.copy(), var=source.var[var_columns].copy(),
        )
        sample.obs["original_barcode"] = source.obs_names.astype(str)
        sample.obs_names = prefix + "_" + source.obs_names.astype(str)
        sample.obs_names.name = "spot_id"
        sample.obs["sample"] = name
        if "spatial" in source.obsm:
            sample.obsm["spatial"] = source.obsm["spatial"].copy()
        spatial_libraries = source.uns.get("spatial", {})
        if spatial_libraries:
            if len(spatial_libraries) != 1:
                raise ValueError(f"{path}: expected one tissue library")
            libraries[name] = next(iter(spatial_libraries.values()))
        samples.append(sample)
        del source
    combined = ad.concat(samples, join="outer", merge="first", fill_value=0)
    if not combined.obs_names.is_unique:
        raise ValueError("Combined spot IDs must be unique")
    combined.obs["sample"] = pd.Categorical(
        combined.obs["sample"], categories=args.sample_names,
    )
    combined.layers["counts"] = combined.X.copy()
    combined.uns["spatial"] = libraries

    print(f"Combined {combined.n_obs:,} spots, {combined.n_vars:,} genes",
          flush=True)
    return combined


def attach_r_results(
    data: ad.AnnData, metadata: pd.DataFrame, umap: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    require_same_spots(data.obs_names, metadata.index, "H5AD vs R metadata")
    metadata = metadata.loc[data.obs_names]
    if not np.array_equal(
        data.obs["sample"].astype(str).to_numpy(), metadata["sample"].to_numpy(),
    ):
        raise ValueError("R sample assignments disagree with the input H5ADs")
    data.obs["age"] = metadata["age"]
    data.obs["seurat_clusters"] = pd.Categorical(
        metadata["seurat_clusters"], categories=labels.index, ordered=True,
    )
    annotations = metadata["seurat_clusters"].map(labels["annotation"])
    data.obs["annotation"] = pd.Categorical(
        annotations, categories=pd.unique(labels["annotation"]),
    )
    data.obsm["X_umap"] = umap.loc[data.obs_names].to_numpy(dtype=float)


def color_maps(labels: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    cluster_colors = {}
    annotation_colors = {}
    for cluster, row in labels.iterrows():
        index = int(cluster)
        if index < len(CLUSTER_COLORS):
            color = CLUSTER_COLORS[index]
        else:
            # Fixed extension so a cluster's color does not depend on subsets.
            color = matplotlib.colors.to_hex(
                plt.get_cmap("tab20")((index - len(CLUSTER_COLORS)) % 20),
            )
        cluster_colors[cluster] = color
        annotation_colors.setdefault(row["annotation"], color)
    return cluster_colors, annotation_colors


def finish_plot(
    figure: plt.Figure, axis: plt.Axes, title: str,
    palette: dict[str, str], path: Path,
) -> None:
    axis.set_title(title, pad=3)
    axis.set_axis_off()
    axis.legend(
        handles=[Patch(facecolor=color, label=label)
                 for label, color in palette.items()],
        bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False,
        borderaxespad=0, labelspacing=0.25,
        ncol=1,
    )
    # Preserve the 3 x 2 inch canvas, including the reserved legend area.
    figure.savefig(path, dpi=300, bbox_inches=figure.bbox_inches, pad_inches=0)
    plt.close(figure)


def create_plot() -> tuple[plt.Figure, plt.Axes]:
    figure, axis = plt.subplots(figsize=(3, 2))
    figure.subplots_adjust(left=0.02, right=0.64, bottom=0.04, top=0.90)
    return figure, axis


def plot_results(
    data: ad.AnnData, labels: pd.DataFrame, output: Path, spot_size_um: float,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans",
        ],
        "axes.titlesize": 7, "axes.labelsize": 6,
        "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 5,
    })
    plot_dir = output / "plots"
    plot_dir.mkdir()
    cluster_colors, annotation_colors = color_maps(labels)
    data.uns["seurat_clusters_colors"] = list(cluster_colors.values())
    data.uns["annotation_colors"] = list(annotation_colors.values())
    colors = data.obs["annotation"].astype(str).map(annotation_colors).to_numpy()
    embedding = data.obsm["X_umap"]
    order = np.random.default_rng(42).permutation(data.n_obs)
    figure, axis = create_plot()
    axis.set_box_aspect(1)
    axis.scatter(
        embedding[order, 0], embedding[order, 1], c=colors[order],
        s=1, linewidths=0, alpha=0.8,
    )
    finish_plot(figure, axis, "UMAP: annotations", annotation_colors,
                plot_dir / "umap-annotations.png")
    for name in data.obs["sample"].cat.categories:
        selected = data.obs["sample"].eq(name).to_numpy()
        sample = ad.AnnData(obs=data.obs.loc[selected].copy())
        if name in data.uns["spatial"]:
            sample.uns["spatial"] = {name: data.uns["spatial"][name]}
            image, x, y, pixel_size = spatial_plot_data(sample)
            side = spot_size_um / pixel_size
            figure, axis = create_plot()
            axis.imshow(image, cmap="gray", origin="upper")
            axis.set_xlim(0, image.shape[1])
            axis.set_ylim(image.shape[0], 0)
        elif {"array_row", "array_col"} <= set(sample.obs):
            x = sample.obs["array_col"].to_numpy(dtype=float)
            y = sample.obs["array_row"].to_numpy(dtype=float)
            side = 0.7 * min(
                (float(np.diff(np.unique(values)).min())
                 for values in (x, y) if len(np.unique(values)) > 1),
                default=1.0,
            )
            figure, axis = create_plot()
            axis.invert_xaxis()
            axis.invert_yaxis()
        else:
            print(f"Skipping spatial plot: no coordinates for {name}")
            continue
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"{name}: spatial coordinates must be finite")
        axis.add_collection(PatchCollection(
            [Rectangle((xx - side / 2, yy - side / 2), side, side)
             for xx, yy in zip(x, y, strict=True)],
            facecolors=colors[selected], edgecolors="none",
        ))
        axis.autoscale_view()
        axis.set_aspect("equal")
        finish_plot(figure, axis, f"{name}: annotations", annotation_colors,
                    plot_dir / f"{name}.spatial-annotations.png")


def main() -> None:
    args = parse_args()
    try:
        output = args.output_dir.expanduser().resolve()
        for name in OUTPUT_NAMES:
            if (output / name).exists():
                raise ValueError(f"Output already exists: {output / name}")
        metadata, umap, labels = read_r_results(args.r_results_dir.expanduser())
        data = combine_samples(args)
        attach_r_results(data, metadata, umap, labels)
        output.mkdir(parents=True, exist_ok=True)
        plot_results(data, labels, output, args.spot_size_um)
        write_anndata(data, output / OUTPUT_NAMES[0])
        table = data.obs.copy()
        table[["UMAP_1", "UMAP_2"]] = data.obsm["X_umap"]
        table.to_csv(output / "spot_clusters.csv", index_label="spot_id")
        labels.to_csv(output / "cluster_annotation.csv", index_label="cluster")
        pd.crosstab(data.obs["sample"], data.obs["seurat_clusters"]).to_csv(
            output / "sample_cluster_counts.csv",
        )
        print(f"Saved R clusters, annotations, UMAP and plots to {output}")
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
