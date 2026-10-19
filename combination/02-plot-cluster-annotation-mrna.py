#!/usr/bin/env python3
"""Plot mRNA annotations on UMAP, tissues, and a reference marker dot plot."""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from utils import spatial_plot_data


CLUSTER_COLORS = (
    "#A73030", "#E64B35", "#2F5597", "#4DBBD5", "#CC79A7", "#7E57C2",
    "#C5A3E0", "#D6A500", "#FFE082", "#2CA02C", "#264653", "#E7298A",
    "#98DF8A", "#C49C94",
)
HIGHLIGHT_COLOR = "#E64B35"
BACKGROUND_COLOR = "#D0D0D0"
CLUSTER_KEY = "leiden"
SAMPLE_ORDER = ("p7-rep1", "p35-rep1", "p60-rep1")
PANEL_SIZE = 4
SPOT_SIZE_UM = 50.0
DOTPLOT_NAME = "marker-dotplot-reference.png"
DOTPLOT_MAX_AREA = 10.0
# Common reference markers; POST uses candidates from this dataset.
# These panels support inspection, rather than defining the annotations.
# Sources: https://doi.org/10.1038/s41586-023-06812-z
#          https://doi.org/10.1038/s41467-018-08079-9
REFERENCE_MARKERS = {
    "Astrocytes": ("Slc1a3", "Aqp4"),
    "Vascular and Leptomeningeal Cells": ("Col1a1", "Col1a2"),
    "Cajal Retzius cells": ("Reln", "Trp73"),
    "Oligodendrocytes": ("Mog", "Opalin"),
    "POST": ("Tshz3", "Dsc3"),
    "CA2/3": ("Amigo2", "Grik4"),
    "CA1": ("Wfs1", "Spink8"),
    "L2/3/4": ("Cux2", "Rorb"),
    "DG": ("Prox1", "Dsp"),
    "L5": ("Fezf2", "Bcl11b"),
    "L6": ("Tle4", "Foxp2"),
    "SUB": ("Nts", "Rxfp1"),
    "TH": ("Tcf7l2", "Slc17a6"),
    "Ependymal": ("Foxj1", "Pifo"),
    "Striatal spiny projection neurons": ("Drd1", "Drd2"),
}


class TissuePanel(NamedTuple):
    sample: str
    image: np.ndarray
    x: np.ndarray
    y: np.ndarray
    side: float
    indices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Clustered H5AD from 01-integrate-cluster-mrna.py.",
    )
    parser.add_argument(
        "--annotations", type=Path, required=True,
        help="Filled cluster_annotation.csv with cluster and annotation columns.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for annotation figures and a reference marker dot plot.",
    )
    return parser.parse_args()


def read_plot_data(path: Path) -> ad.AnnData:
    source = ad.read_h5ad(path, backed="r")
    try:
        required = {
            CLUSTER_KEY, "sample", "pxl_row_in_fullres", "pxl_col_in_fullres",
        }
        missing = required.difference(source.obs.columns)
        if missing:
            raise ValueError("Missing spot columns: " + ", ".join(sorted(missing)))
        if not source.obs_names.is_unique:
            raise ValueError("Input spot IDs must be unique")
        if source.obs[[CLUSTER_KEY, "sample"]].isna().any().any():
            raise ValueError("Cluster and sample labels must not be missing")
        if "X_umap" not in source.obsm:
            raise ValueError("Input is missing obsm['X_umap']")
        embedding = np.asarray(source.obsm["X_umap"], dtype=float)
        if embedding.shape != (source.n_obs, 2) or not np.isfinite(embedding).all():
            raise ValueError("UMAP must contain two finite coordinates per spot")
        # Keep only plotting metadata; never copy or modify expression values.
        data = ad.AnnData(obs=source.obs[list(required)].copy())
        data.obsm["X_umap"] = embedding.copy()
        data.uns["spatial"] = source.uns.get("spatial", {})
    finally:
        source.file.close()
    data.obs[CLUSTER_KEY] = data.obs[CLUSTER_KEY].astype(str)
    if not data.obs[CLUSTER_KEY].str.fullmatch(r"0|[1-9][0-9]*").all():
        raise ValueError("Cluster IDs must be nonnegative integers")
    return data


def read_annotations(path: Path, clusters: pd.Series) -> pd.DataFrame:
    table = pd.read_csv(
        path, dtype=str, keep_default_na=False, encoding="utf-8-sig",
    )
    if not {"cluster", "annotation"} <= set(table.columns):
        raise ValueError("Annotation CSV must contain cluster and annotation")
    table = table[["cluster", "annotation"]].copy()
    table["cluster"] = table["cluster"].str.strip()
    table["annotation"] = table["annotation"].str.strip()
    if table["cluster"].duplicated().any():
        raise ValueError("Annotation CSV has duplicate cluster IDs")
    if table["annotation"].eq("").any():
        empty = table.loc[table["annotation"].eq(""), "cluster"]
        raise ValueError("Fill annotations for clusters: " + ", ".join(empty))
    observed = set(clusters)
    missing = observed.difference(table["cluster"])
    unknown = set(table["cluster"]).difference(observed)
    if missing or unknown:
        raise ValueError(
            f"Annotation IDs do not match the H5AD; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    order = sorted(observed, key=int)
    return table.set_index("cluster").loc[order].reset_index()


def annotation_colors(table: pd.DataFrame) -> dict[str, str]:
    maximum = max(map(int, table["cluster"]))
    extra = max(0, maximum + 1 - len(CLUSTER_COLORS))
    result = {}
    for row in table.itertuples(index=False):
        index = int(row.cluster)
        if index < len(CLUSTER_COLORS):
            color = CLUSTER_COLORS[index]
        else:
            position = (index - len(CLUSTER_COLORS)) / max(1, extra - 1)
            color = matplotlib.colors.to_hex(plt.get_cmap("turbo")(position))
        # Repeated annotation names share their lowest-cluster-ID color.
        result.setdefault(row.annotation, color)
    return result


def prepare_tissue_panels(data: ad.AnnData) -> list[TissuePanel]:
    panels = []
    for name in SAMPLE_ORDER:
        indices = np.flatnonzero(data.obs["sample"].astype(str).eq(name))
        sample = ad.AnnData(obs=data.obs.iloc[indices].copy())
        if name not in data.uns["spatial"]:
            raise ValueError(f"Missing tissue library for {name}")
        sample.uns["spatial"] = {name: data.uns["spatial"][name]}
        image, x, y, pixel_size = spatial_plot_data(sample)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Nonfinite spatial coordinates in {name}")
        if image.ndim not in (2, 3) or min(image.shape[:2]) < 1:
            raise ValueError(f"Invalid tissue image for {name}")
        panels.append(TissuePanel(
            name, image, x, y, SPOT_SIZE_UM / pixel_size, indices,
        ))
    return panels


def read_annotation_expression(
    path: Path, annotations: pd.Series,
) -> ad.AnnData:
    """Normalize raw counts, summing features with identical gene symbols."""
    source = ad.read_h5ad(path, backed="r")
    try:
        if not source.obs_names.equals(annotations.index):
            raise ValueError("Expression and annotation spot IDs do not match")
        if "counts" not in source.layers:
            raise ValueError("Marker dot plots require layers['counts']")
        counts = sparse.csr_matrix(
            source.layers["counts"], dtype=np.float32, copy=True,
        )
        counts.sum_duplicates()
        counts.eliminate_zeros()
        if (not np.isfinite(counts.data).all()
                or (counts.data < 0).any()
                or not np.allclose(counts.data, np.rint(counts.data),
                                   rtol=0, atol=1e-5)):
            raise ValueError("Marker dot plots require nonnegative raw counts")
        if np.any(np.asarray(counts.sum(axis=1)).ravel() <= 0):
            raise ValueError("Marker dot plots cannot use zero-count spots")
        symbols = source.var.get(
            "feature_name", pd.Series(source.var_names, index=source.var_names),
        ).astype(str).str.strip()
        symbols = symbols.where(
            symbols.ne("") & symbols.ne("nan"), source.var_names,
        )
        codes, unique_symbols = pd.factorize(symbols, sort=False)
        gene_mapping = sparse.csr_matrix(
            (np.ones(source.n_vars, dtype=np.float32),
             (np.arange(source.n_vars), codes)),
            shape=(source.n_vars, len(unique_symbols)),
        )
        expression = ad.AnnData(
            X=(counts @ gene_mapping).tocsr(),
            obs=pd.DataFrame({"annotation": pd.Categorical(
                annotations, categories=pd.unique(annotations),
            )}, index=annotations.index.copy()),
            var=pd.DataFrame(index=pd.Index(unique_symbols, name="gene")),
        )
    finally:
        source.file.close()
    sc.pp.normalize_total(expression, target_sum=10000)
    sc.pp.log1p(expression)
    return expression


def marker_dotplot_statistics(
    expression: ad.AnnData, annotations: list[str], genes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return spot detection fractions and gene-wise z-scored group means."""
    values = expression[:, genes].X
    fractions = np.zeros((len(annotations), len(genes)))
    means = np.zeros_like(fractions)
    for index, annotation in enumerate(annotations):
        selected = expression.obs["annotation"].eq(annotation).to_numpy()
        if not selected.any():
            raise ValueError(f"No spots for annotation: {annotation}")
        group = values[selected]
        fractions[index] = np.asarray((group > 0).mean(axis=0)).ravel()
        means[index] = np.asarray(group.mean(axis=0)).ravel()
    deviation = means.std(axis=0)
    scaled = np.divide(means - means.mean(axis=0), deviation,
                       out=np.zeros_like(means), where=deviation > 0)
    return fractions, scaled


def plot_marker_dotplot(
    expression: ad.AnnData, annotations: list[str],
    marker_groups: dict[str, list[str] | tuple[str, ...]],
    title: str, output: Path,
) -> None:
    genes = list(dict.fromkeys(
        gene for annotation in annotations
        for gene in marker_groups.get(annotation, ())
        if gene in expression.var_names
    ))
    missing = sorted({gene for values in marker_groups.values()
                      for gene in values if gene not in expression.var_names})
    if missing:
        print("Skipping unavailable marker genes: " + ", ".join(missing),
              flush=True)
    if not genes:
        raise ValueError(f"No available marker genes for {title}")
    fractions, scaled = marker_dotplot_statistics(
        expression, annotations, genes,
    )
    color_map = LinearSegmentedColormap.from_list(
        "marker_expression", ["#4DBBD5", "white", "#E64B35"],
    )
    figure = plt.figure(figsize=(4, 2))
    try:
        axis = figure.add_axes((0.33, 0.26, 0.48, 0.66))
        x, y = np.meshgrid(np.arange(len(genes)), np.arange(len(annotations)))
        points = axis.scatter(
            x.ravel(), y.ravel(), s=DOTPLOT_MAX_AREA * fractions.ravel(),
            c=scaled.ravel(), cmap=color_map, norm=Normalize(-2, 2),
            edgecolors="none",
        )
        axis.set_xticks(range(len(genes)), genes, rotation=90, fontsize=5)
        axis.set_yticks(range(len(annotations)), annotations, fontsize=5)
        axis.set_xlim(-0.6, len(genes) - 0.4)
        axis.set_ylim(-0.6, len(annotations) - 0.4)
        axis.tick_params(length=0, pad=2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_linewidth(0.5)
        axis.set_title(title, fontsize=7, pad=3)
        color_axis = figure.add_axes((0.86, 0.64, 0.018, 0.2))
        colorbar = figure.colorbar(points, cax=color_axis, ticks=[-2, 0, 2])
        colorbar.ax.tick_params(labelsize=5, width=0.5, length=2, pad=2)
        colorbar.outline.set_linewidth(0.5)
        color_axis.set_title("Mean expr.\n(gene z-score)", fontsize=5, pad=3)
        size_axis = figure.add_axes((0.86, 0.27, 0.11, 0.22))
        for index, fraction in enumerate((0.25, 0.5, 0.75, 1.0)):
            size_axis.scatter(0.15, index, s=DOTPLOT_MAX_AREA * fraction,
                              c="black", edgecolors="none")
            size_axis.text(0.4, index, f"{fraction:.0%}", fontsize=5,
                           va="center")
        size_axis.set(xlim=(0, 1), ylim=(-0.6, 3.6))
        size_axis.invert_yaxis()
        size_axis.set_axis_off()
        size_axis.set_title("Expressing\nspots", fontsize=5, pad=3)
        figure.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.02)
    finally:
        plt.close(figure)
    print(f"Saved {title}: {len(genes)} genes, {len(annotations)} annotations",
          flush=True)


def plot_umap(
    axis: plt.Axes, embedding: np.ndarray, colors: np.ndarray,
    selected: np.ndarray | None,
) -> None:
    if selected is None:
        order = np.random.default_rng(42).permutation(len(embedding))
        axis.scatter(
            embedding[order, 0], embedding[order, 1], c=colors[order],
            s=2, alpha=0.7, linewidths=0,
        )
    else:
        axis.scatter(
            embedding[:, 0], embedding[:, 1], c=BACKGROUND_COLOR,
            s=2, linewidths=0,
        )
        axis.scatter(
            embedding[selected, 0], embedding[selected, 1],
            c=HIGHLIGHT_COLOR, s=2, linewidths=0,
        )
    count = len(embedding) if selected is None else int(selected.sum())
    axis.set_title(
        f"UMAP (n={count:,})", fontsize=7, pad=3,
    )
    axis.set_axis_off()


def plot_tissue(
    axis: plt.Axes, panel: TissuePanel, colors: np.ndarray,
    selected: np.ndarray | None,
) -> None:
    axis.imshow(panel.image, cmap="gray", origin="upper")
    mask = (
        np.ones(len(panel.indices), dtype=bool) if selected is None
        else selected[panel.indices]
    )
    if mask.any():
        facecolors = (
            colors[panel.indices[mask]] if selected is None else HIGHLIGHT_COLOR
        )
        axis.add_collection(PatchCollection(
            [Rectangle((x - panel.side / 2, y - panel.side / 2),
                       panel.side, panel.side)
             for x, y in zip(panel.x[mask], panel.y[mask], strict=True)],
            facecolors=facecolors, edgecolors="none",
        ))
    axis.set_xlim(0, panel.image.shape[1])
    axis.set_ylim(panel.image.shape[0], 0)
    axis.set_aspect("equal")
    axis.set_title(
        f"{panel.sample} (n={int(mask.sum()):,})",
        fontsize=7, pad=3,
    )
    axis.set_axis_off()


def plot_four_panels(
    data: ad.AnnData, panels: list[TissuePanel], colors: np.ndarray,
    palette: dict[str, str], selected: np.ndarray | None,
    title: str, output: Path,
) -> None:
    figure, grid = plt.subplots(
        2, 2, figsize=(2 * PANEL_SIZE, 2 * PANEL_SIZE),
    )
    axes = grid.ravel()
    try:
        tissue_axes = axes[:3] if selected is None else axes[1:]
        if selected is not None:
            plot_umap(axes[0], data.obsm["X_umap"], colors, selected)
        for axis, panel in zip(tissue_axes, panels, strict=True):
            plot_tissue(axis, panel, colors, selected)
        figure.suptitle(title, fontsize=7, y=0.98)
        figure.subplots_adjust(
            left=0.01, right=0.99, bottom=0.02,
            top=0.94,
            wspace=0.07, hspace=0.12,
        )
        if selected is None:
            axes[3].set_axis_off()
            axes[3].legend(
                handles=[Patch(facecolor=color, label=textwrap.fill(name, 28))
                         for name, color in palette.items()],
                loc="center",
                frameon=False, fontsize=5, borderaxespad=0,
            )
        figure.savefig(
            output, dpi=300, bbox_inches="tight", pad_inches=0.02,
        )
    finally:
        plt.close(figure)


def run(args: argparse.Namespace) -> None:
    input_path = args.input.expanduser().resolve()
    annotation_path = args.annotations.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (input_path, annotation_path):
        if not path.is_file():
            raise ValueError(f"Input does not exist: {path}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output directory is a file: {output_dir}")
    data = read_plot_data(input_path)
    table = read_annotations(annotation_path, data.obs[CLUSTER_KEY])
    if set(data.obs["sample"].astype(str)) != set(SAMPLE_ORDER):
        raise ValueError("Input must contain p7-rep1, p35-rep1 and p60-rep1")
    panels = prepare_tissue_panels(data)
    palette = annotation_colors(table)
    mapping = table.set_index("cluster")["annotation"]
    data.obs["annotation"] = data.obs[CLUSTER_KEY].map(mapping)
    colors = data.obs["annotation"].map(palette).to_numpy()
    annotations = table["annotation"].drop_duplicates().tolist()
    filenames = {}
    used_names = {"annotation-overview.png"}
    for annotation in annotations:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", annotation)
        slug = slug.strip("-._")[:80] or "annotation"
        filename = f"annotation-{slug}.png"
        suffix = 2
        while filename in used_names:
            filename = f"annotation-{slug}-{suffix}.png"
            suffix += 1
        filenames[annotation] = filename
        used_names.add(filename)
    outputs = ["annotation-overview.png"]
    outputs.extend(filenames.values())
    outputs.append(DOTPLOT_NAME)
    existing = [name for name in outputs if (output_dir / name).exists()]
    if existing:
        raise ValueError("Choose a new output directory; files exist: "
                         + ", ".join(existing))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans",
        ],
    })
    expression = read_annotation_expression(input_path, data.obs["annotation"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_marker_dotplot(
        expression, annotations, REFERENCE_MARKERS, "Reference markers",
        output_dir / DOTPLOT_NAME,
    )
    del expression
    plot_four_panels(
        data, panels, colors, palette, None, "Annotations",
        output_dir / "annotation-overview.png",
    )
    for annotation in annotations:
        selected = data.obs["annotation"].eq(annotation).to_numpy()
        counts = {
            panel.sample: int(selected[panel.indices].sum()) for panel in panels
        }
        plot_four_panels(
            data, panels, colors, palette, selected,
            annotation, output_dir / filenames[annotation],
        )
        print(
            f"{annotation}; "
            f"n={int(selected.sum()):,}; {counts}", flush=True,
        )
    print(f"Saved overview, {len(annotations)} annotation plots, "
          f"and a reference marker dot plot in {output_dir}")


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
