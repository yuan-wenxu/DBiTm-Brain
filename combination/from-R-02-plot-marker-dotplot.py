#!/usr/bin/env python3
"""Plot marker expression by Seurat cluster in an R-annotated mRNA H5AD."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Column order in marker-dotplot-r-annotations.png, which supplements
# qs/01-integrate/canonical-markers.csv with markers for the final annotations.
DEFAULT_MARKERS = (
    "Mog", "Opalin", "Slc1a3", "Apoe", "Shox2", "Rgs16", "Cux2", "Satb2",
    "Cdh18", "Nxph3", "Drd1", "Drd2", "Fezf2", "Etv1", "Hap1", "Gad2",
    "Rorb", "Kcnh5", "Col1a1", "Slc6a13", "Dlx1", "Tshz2", "Dsc3", "Pvalb",
    "Six3", "Nts", "Nr4a2", "Cplx3", "Ccn2", "Foxj1", "Tmem212", "Lct",
    "Prox1", "Fibcd1", "Wfs1", "Neurod6", "Cpne4", "Col23a1", "Rps6ka2",
)
TITLE = "Marker expression by Seurat cluster"
MAX_DOT_AREA = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True,
        help="R-annotated H5AD from from-R-01-combine.py.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output PNG.",
    )
    parser.add_argument(
        "--genes", nargs="+", default=DEFAULT_MARKERS,
        help="Gene symbols in plotting order. Defaults to the 39-marker panel.",
    )
    return parser.parse_args()


def cluster_labels(obs: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    missing = {"seurat_clusters", "annotation"}.difference(obs.columns)
    if missing:
        raise ValueError("H5AD is missing obs columns: " + ", ".join(sorted(missing)))
    if obs.empty:
        raise ValueError("H5AD contains no spots")

    clusters = obs["seurat_clusters"].astype("string").str.strip()
    annotations = obs["annotation"].astype("string").str.strip()
    if not clusters.str.fullmatch(r"0|[1-9][0-9]*").fillna(False).all():
        raise ValueError("Seurat cluster IDs must be nonnegative integers")
    if annotations.fillna("").eq("").any():
        raise ValueError("Every spot must have an annotation")
    labels = clusters.str.cat(annotations, sep="  ")
    rows = pd.DataFrame({"cluster": clusters, "label": labels})
    rows = rows.drop_duplicates()
    if rows["cluster"].duplicated().any():
        raise ValueError("Each Seurat cluster must have exactly one annotation")
    ordered = rows.sort_values("cluster", key=lambda values: values.map(int))
    return labels, ordered["label"].tolist()


def annotation_expression(
    source: ad.AnnData, annotations: pd.Series,
) -> ad.AnnData:
    """Normalize raw counts, summing features with identical gene symbols."""
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
        symbols.notna() & ~symbols.isin(("", "nan")), source.var_names,
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


def plot_dotplot(
    genes: list[str], labels: list[str], fractions: np.ndarray,
    scaled: np.ndarray, output: Path,
) -> None:
    color_map = LinearSegmentedColormap.from_list(
        "marker_expression", ["#4DBBD5", "white", "#E64B35"],
    )
    figure = plt.figure(figsize=(4, 2))
    try:
        axis = figure.add_axes((0.16, 0.22, 0.67, 0.68))
        x, y = np.meshgrid(np.arange(len(genes)), np.arange(len(labels)))
        points = axis.scatter(
            x.ravel(), y.ravel(), s=MAX_DOT_AREA * fractions.ravel(),
            c=scaled.ravel(), cmap=color_map, norm=Normalize(-2, 2),
            edgecolors="none",
        )
        axis.set_xticks(range(len(genes)), genes, rotation=90, fontsize=5)
        axis.set_yticks(range(len(labels)), labels, fontsize=5)
        axis.set(xlim=(-0.65, len(genes) - 0.35),
                 ylim=(len(labels) - 0.5, -0.5))
        axis.tick_params(length=0, pad=2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_linewidth(0.5)
        axis.set_title(TITLE, fontsize=7, pad=3)

        color_axis = figure.add_axes((0.865, 0.60, 0.018, 0.21))
        colorbar = figure.colorbar(points, cax=color_axis, ticks=[-2, 0, 2])
        colorbar.ax.tick_params(labelsize=5, width=0.5, length=2, pad=2)
        colorbar.outline.set_linewidth(0.5)
        color_axis.set_title(
            "Mean expression\n(gene z-score)", loc="left", fontsize=5, pad=3,
        )

        size_axis = figure.add_axes((0.865, 0.27, 0.13, 0.22))
        for index, fraction in enumerate((0.25, 0.5, 0.75, 1.0)):
            size_axis.scatter(
                0.15, index, s=MAX_DOT_AREA * fraction, c="#303030",
                edgecolors="none",
            )
            size_axis.text(
                0.31, index, f"{fraction:.0%}", fontsize=5, va="center",
            )
        size_axis.set(xlim=(0, 1), ylim=(3.6, -0.6))
        size_axis.set_axis_off()
        size_axis.set_title("Expressing spots", loc="left", fontsize=5, pad=3)
        figure.savefig(
            output, dpi=300, bbox_inches="tight", pad_inches=0.02)
    finally:
        plt.close(figure)


def main() -> None:
    args = parse_args()
    output = args.output.expanduser()
    try:
        if output.suffix.lower() != ".png":
            raise ValueError("--output must end in .png")
        if output.exists():
            raise ValueError(f"Output file already exists: {output}.")
        source = ad.read_h5ad(args.input.expanduser(), backed="r")
        try:
            spot_labels, labels = cluster_labels(source.obs)
            expression = annotation_expression(source, spot_labels)
        finally:
            source.file.close()
        requested = list(dict.fromkeys(args.genes))
        available = set(expression.var_names)
        genes = [gene for gene in requested if gene in available]
        if not genes:
            raise ValueError("None of the requested markers are in the H5AD")
        missing = [gene for gene in requested if gene not in available]
        if missing:
            print("Skipping unavailable marker genes: " + ", ".join(missing))
        fractions, scaled = marker_dotplot_statistics(expression, labels, genes)
        output.parent.mkdir(parents=True, exist_ok=True)
        with plt.rc_context({
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans",
            ],
        }):
            plot_dotplot(genes, labels, fractions, scaled, output)
        print(f"Saved {len(genes)} genes × {len(labels)} clusters to {output}")
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
