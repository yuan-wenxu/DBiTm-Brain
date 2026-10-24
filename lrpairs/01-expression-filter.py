"""Filter cortical LR pairs per sample and plot their overlap."""

import argparse
from pathlib import Path
import re

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


SAMPLE_COLORS = ("#E64B35", "#2F5597", "#4DBBD5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-pct", type=float, default=10.0,
        help="Each gene must exceed this percentage of pooled cortical spots.",
    )
    parser.add_argument("--labels", nargs="+", default=["L2/3/4", "L5", "L6"])
    return parser.parse_args()


def map_gene(gene_id: str, symbol: str, var: pd.DataFrame) -> int:
    if pd.notna(gene_id) and gene_id in var.index:
        return int(var.index.get_loc(gene_id))
    matches = np.flatnonzero(var["feature_name"].to_numpy() == symbol)
    return int(matches[0]) if len(matches) == 1 else -1


def filter_sample(
    counts: sparse.csr_matrix, pairs: pd.DataFrame, indices: np.ndarray,
    min_pct: float,
) -> pd.DataFrame:
    # Pool all selected cortical spots within this sample.
    pct = np.zeros(counts.shape[1])
    if counts.shape[0]:
        pct = np.asarray((counts > 0).mean(axis=0)).ravel() * 100
    coverage = np.zeros(indices.shape)
    mapped = indices >= 0
    coverage[mapped] = pct[indices[mapped]]
    result = pairs.copy()
    result["ligand_cortex_pct"] = coverage[:, 0]
    result["receptor_cortex_pct"] = coverage[:, 1]
    keep = mapped.all(axis=1) & (coverage > min_pct).all(axis=1)
    return (
        result.loc[keep].drop_duplicates("LR Pair")
        .sort_values("LR Pair").reset_index(drop=True)
    )


def plot_overlap(pair_sets: dict[str, set[str]], path: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans",
        ],
        "font.size": 5, "axes.titlesize": 7,
    })
    samples = list(pair_sets)
    if len(samples) == 3:
        centers = [(-0.45, 0.25), (0.45, 0.25), (0, -0.45)]
        labels = [(-0.8, 1.03), (0.8, 1.03), (0, -1.3)]
        regions = {
            1: (-0.75, 0.4), 2: (0.75, 0.4), 3: (0, 0.55),
            4: (0, -0.85), 5: (-0.42, -0.3), 6: (0.42, -0.3),
            7: (0, -0.05),
        }
    else:
        centers = [(0, 0)] if len(samples) == 1 else [(-0.45, 0), (0.45, 0)]
        labels = [(x, -1.0) for x, _ in centers]
        regions = {1: (0, 0)} if len(samples) == 1 else {
            1: (-0.75, 0), 2: (0.75, 0), 3: (0, 0),
        }
    region_counts = dict.fromkeys(regions, 0)
    for pair in set().union(*pair_sets.values()):
        membership = sum(
            1 << index for index, sample in enumerate(samples)
            if pair in pair_sets[sample]
        )
        region_counts[membership] += 1
    figure, axis = plt.subplots(figsize=(4, 2))
    for index, sample in enumerate(samples):
        axis.add_patch(Circle(
            centers[index], 0.72, facecolor=SAMPLE_COLORS[index],
            edgecolor=SAMPLE_COLORS[index], alpha=0.35, linewidth=0.6,
        ))
        axis.text(
            *labels[index], f"{sample} (n={len(pair_sets[sample])})",
            ha="center", va="center",
        )
    for membership, position in regions.items():
        axis.text(*position, str(region_counts[membership]),
                  ha="center", va="center")
    axis.set(xlim=(-1.6, 1.6), ylim=(-1.45, 1.2), aspect="equal")
    axis.set_title("Cortical LR pair overlap", pad=3)
    axis.text(0.5, -0.04, "Unique LR pairs; areas not to scale",
              transform=axis.transAxes, ha="center", va="top")
    axis.axis("off")
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0 < args.min_pct <= 100:
        raise ValueError("--min-pct must be in (0, 100]")
    database = pd.read_csv(args.database)
    tokens = database["LR Pair"].str.split()
    if database.empty or not tokens.str.len().eq(2).all():
        raise ValueError("Every LR Pair must contain two gene symbols")
    pairs = pd.DataFrame({
        "LR Pair": tokens.str.join(" "),
        "ligand": tokens.str[0], "receptor": tokens.str[1],
    })
    data = ad.read_h5ad(args.h5ad)
    if "counts" not in data.layers:
        raise ValueError(f"{args.h5ad}: missing layers['counts']")
    if not data.var_names.is_unique:
        raise ValueError("Feature IDs must be unique")
    cortex = data.obs["annotation"].isin(args.labels).to_numpy()
    obs = data.obs.loc[cortex].copy()
    if obs.empty or data.obs["sample"].isna().any():
        raise ValueError("Expected cortical spots with nonmissing samples")
    samples = data.obs["sample"].astype(str).unique().tolist()
    if not 1 <= len(samples) <= 3:
        raise ValueError("The Venn diagram supports one to three samples")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sample)
           for sample in samples):
        raise ValueError("Sample names must be safe filenames")
    indices = np.column_stack([
        [map_gene(gene_id, symbol, data.var)
         for gene_id, symbol in zip(
             database[f"{role.title()} ENSEMBL ID"], pairs[role],
         )]
        for role in ("ligand", "receptor")
    ])
    for column, role in enumerate(("ligand", "receptor")):
        pairs[f"{role}_feature_id"] = [
            data.var_names[index] if index >= 0 else ""
            for index in indices[:, column]
        ]
    counts = sparse.csr_matrix(data.layers["counts"], dtype=float)[cortex]
    del data
    counts.sum_duplicates()
    counts.eliminate_zeros()
    if (not np.isfinite(counts.data).all() or (counts.data < 0).any()
            or (counts.data != np.floor(counts.data)).any()):
        raise ValueError("Counts must be finite nonnegative integers")
    if (np.asarray(counts.sum(axis=1)).ravel() == 0).any():
        raise ValueError("Cortical observations contain empty libraries")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pair_sets = {}
    for sample in samples:
        mask = obs["sample"].astype(str).eq(sample).to_numpy()
        result = filter_sample(
            counts[mask], pairs, indices, args.min_pct,
        )
        result.to_csv(args.output_dir / f"{sample}.lr-pairs.csv",
                      index=False, encoding="utf-8-sig")
        pair_sets[sample] = set(result["LR Pair"])
        print(f"{sample}: {len(result)} LR pairs")
    plot_overlap(pair_sets, args.output_dir / "lr-pairs-venn.png")


if __name__ == "__main__":
    main()
