"""Plot per-sample histograms of median nearest distances across LR pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


DIRECTIONS = ("ligand_to_receptor", "receptor_to_ligand")
COLORS = ("#E64B35", "#4DBBD5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument(
        "--pairs-dir", type=Path, required=True,
        help="Directory containing SAMPLE.lr-pairs.csv from script 01.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", nargs="+")
    parser.add_argument("--labels", nargs="+", default=["L2/3/4", "L5", "L6"])
    parser.add_argument("--min-counts", type=int, default=3)
    parser.add_argument(
        "--expression-quantile", type=float, default=0.75,
        help="Quantile of each gene's nonzero cortical CP10K (0 to 1).",
    )
    parser.add_argument(
        "--pixel-size-um", type=float,
        help="Override source_pixel_size_um for all selected samples.",
    )
    parser.add_argument("--bins", type=int, default=100)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a new directory")
    if args.min_counts < 1 or args.bins < 1:
        parser.error("--min-counts and --bins must be positive")
    if not 0 <= args.expression_quantile <= 1:
        parser.error("--expression-quantile must be in [0, 1]")
    if args.pixel_size_um is not None and (
        not np.isfinite(args.pixel_size_um) or args.pixel_size_um <= 0
    ):
        parser.error("--pixel-size-um must be finite and positive")
    return args


def gene_index(row: pd.Series, role: str, var: pd.DataFrame) -> int:
    feature_id = row.get(f"{role}_feature_id")
    if pd.notna(feature_id) and str(feature_id):
        if feature_id not in var.index:
            raise ValueError(f"Feature ID absent from H5AD: {feature_id}")
        return int(var.index.get_loc(feature_id))
    matches = np.flatnonzero(var["feature_name"].to_numpy() == row[role])
    if len(matches) != 1:
        raise ValueError(
            f"Cannot uniquely map {role} {row[role]}; rerun script 01 "
            "to retain feature IDs"
        )
    return int(matches[0])


def high_expression(
    counts: sparse.csc_matrix, column: int, totals: np.ndarray,
    min_counts: int, quantile: float,
) -> np.ndarray:
    raw = counts[:, column].toarray().ravel()
    cp10k = raw / totals * 1e4
    positive = cp10k[raw > 0]
    threshold = (
        float(np.quantile(positive, quantile)) if positive.size else np.nan
    )
    spots = np.flatnonzero((raw >= min_counts) & (cp10k >= threshold))
    return spots


def nearest_distances(
    coordinates: np.ndarray, source: np.ndarray, target: np.ndarray,
) -> np.ndarray:
    """Find nearest target spots, always excluding the source spot itself."""
    if not source.size or not target.size:
        return np.full(source.size, np.nan)
    k = min(2, target.size)
    distances, neighbors = cKDTree(coordinates[target]).query(
        coordinates[source], k=k,
    )
    distances = distances.reshape(source.size, k)
    neighbors = neighbors.reshape(source.size, k)
    distances[target[neighbors] == source[:, None]] = np.inf
    nearest = distances.min(axis=1)
    nearest[~np.isfinite(nearest)] = np.nan
    return nearest


def sample_distances(
    sample: str, obs: pd.DataFrame, counts: sparse.csc_matrix,
    var: pd.DataFrame, pairs: pd.DataFrame, pixel_size: float,
    args: argparse.Namespace,
) -> list[dict]:
    coordinates = obs[[
        "pxl_col_in_fullres", "pxl_row_in_fullres",
    ]].to_numpy(dtype=float) * pixel_size
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{sample}: spatial coordinates must be finite")
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError(f"{sample}: duplicate spatial spot centers")
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if (totals <= 0).any():
        raise ValueError(f"{sample}: empty cortical count libraries")
    cache = {}
    results = []
    for _, row in pairs.iterrows():
        columns = [gene_index(row, role, var)
                   for role in ("ligand", "receptor")]
        for column in columns:
            if column not in cache:
                cache[column] = high_expression(
                    counts, column, totals, args.min_counts,
                    args.expression_quantile,
                )
        ligand, receptor = [cache[column] for column in columns]
        result = {
            "sample": sample, "LR Pair": row["LR Pair"],
            "ligand": row["ligand"], "receptor": row["receptor"],
            "ligand_feature_id": var.index[columns[0]],
            "receptor_feature_id": var.index[columns[1]],
            "n_ligand_high": ligand.size, "n_receptor_high": receptor.size,
        }
        for direction, source, target in (
            (DIRECTIONS[0], ligand, receptor),
            (DIRECTIONS[1], receptor, ligand),
        ):
            distances = nearest_distances(coordinates, source, target)
            distances = distances[np.isfinite(distances)]
            result[f"{direction}_median_um"] = (
                float(np.median(distances)) if distances.size else np.nan
            )
        results.append(result)
    return results


def plot_histograms(
    summary: pd.DataFrame, output: Path, bins: int,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans",
        ],
        "axes.titlesize": 7, "axes.labelsize": 6,
        "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 5,
    })
    columns = [f"{direction}_median_um" for direction in DIRECTIONS]
    for sample, group in summary.groupby("sample", sort=False):
        values = group[columns].to_numpy().ravel()
        values = values[np.isfinite(values)]
        maximum = (
            max(float(np.quantile(values, 0.95)), 1.0)
            if values.size else 1.0
        )
        edges = np.linspace(0, maximum, bins + 1)
        peak = max(
            int(np.histogram(group[column].dropna(), bins=edges)[0].max())
            for column in columns
        )
        figure, axis = plt.subplots(figsize=(4, 2))
        distances = [group[column].dropna().to_numpy() for column in columns]
        labels = [
            f"{label} (shown={(values <= maximum).sum()}/{len(values)})"
            for label, values in zip(
                ("Ligand to receptor", "Receptor to ligand"), distances,
            )
        ]
        axis.hist(
            distances, bins=edges, histtype="bar", color=COLORS,
            label=labels, rwidth=0.95, linewidth=0,
        )
        if not group[columns].notna().any().any():
            axis.text(0.5, 0.5, "No valid distances",
                      transform=axis.transAxes, ha="center", fontsize=5)
        axis.set_xlim(0, maximum)
        axis.set_ylim(0, max(peak, 1) * 1.12)
        axis.set_xlabel("Median nearest-neighbor distance per LR pair (μm)")
        axis.set_ylabel("LR pairs", labelpad=5)
        axis.set_title(f"{sample}: LR distances (P95 cutoff)", pad=3)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=8))
        axis.legend(frameon=False, loc="upper right")
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout(pad=0.3)
        figure.savefig(output / f"{sample}.lr-distances.histogram.png",
                       dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(figure)


def main() -> None:
    args = parse_args()
    files = sorted(args.pairs_dir.glob("*.lr-pairs.csv"))
    if not files:
        raise ValueError("No SAMPLE.lr-pairs.csv files in --pairs-dir")
    tables = {}
    for path in files:
        sample = path.name.removesuffix(".lr-pairs.csv")
        if args.samples and sample not in args.samples:
            continue
        table = pd.read_csv(path)
        required = ["LR Pair", "ligand", "receptor"]
        if (not set(required) <= set(table)
                or table[required].isna().any().any()):
            raise ValueError(f"{path}: missing LR pair identifiers")
        table["LR Pair"] = table["LR Pair"].str.split().str.join(" ")
        if table["LR Pair"].duplicated().any():
            raise ValueError(f"{path}: duplicate LR pairs")
        tables[sample] = table
    if args.samples and set(args.samples) - set(tables):
        raise ValueError("Missing LR CSV for a requested sample")
    if not any(len(table) for table in tables.values()):
        raise ValueError("No LR pairs selected from the sample CSVs")
    data = ad.read_h5ad(args.h5ad)
    if "counts" not in data.layers:
        raise ValueError("H5AD is missing layers['counts']")
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError("Spot and feature IDs must be unique")
    obs, var = data.obs, data.var
    metadata = {
        sample: library.get("metadata", {})
        for sample, library in data.uns.get("spatial", {}).items()
    }
    counts = sparse.csr_matrix(data.layers["counts"], dtype=float, copy=True)
    del data
    counts.sum_duplicates()
    counts.eliminate_zeros()
    if (not np.isfinite(counts.data).all() or (counts.data < 0).any()
            or (counts.data != np.floor(counts.data)).any()):
        raise ValueError("Counts must be finite nonnegative integers")
    results = []
    for sample, pairs in tables.items():
        if pairs.empty:
            print(f"{sample}: no selected LR pairs; skipped")
            continue
        mask = (obs["sample"].eq(sample)
                & obs["annotation"].isin(args.labels)).to_numpy()
        if not mask.any():
            raise ValueError(f"{sample}: no selected cortical spots")
        pixel_size = args.pixel_size_um
        if pixel_size is None:
            pixel_size = metadata.get(sample, {}).get("source_pixel_size_um")
        if (pixel_size is None or not np.isfinite(pixel_size)
                or pixel_size <= 0):
            raise ValueError(f"{sample}: set --pixel-size-um; no valid scale")
        results.extend(sample_distances(
            sample, obs.loc[mask], counts[mask].tocsc(), var, pairs,
            float(pixel_size), args,
        ))
        print(f"{sample}: computed {len(pairs)} LR pairs")
    summary = pd.DataFrame(results)
    args.output_dir.mkdir(parents=True)
    summary.to_csv(args.output_dir / "lr-distances.csv", index=False,
                   encoding="utf-8-sig")
    plots = args.output_dir / "plots"
    plots.mkdir()
    plot_histograms(summary, plots, args.bins)
    print(f"Saved distance summaries and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
