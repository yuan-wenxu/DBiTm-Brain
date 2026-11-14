#!/usr/bin/env python3
"""Compare gene-bin methylation for high, low and unexpressed RNA genes."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_colors import cluster_color


REGIONS = ("Cortex", "Other")
GROUPS = ("no_expression", "low_expression", "high_expression")
GROUP_LABELS = ("No expression", "Low expression", "High expression")
# Fixed IDs keep the shared palette independent of missing groups.
GROUP_COLORS = tuple(cluster_color(index) for index in (2, 6, 1))
ASSAYS = ("TAPS", "TAPS-beta", "5hmC")
FRACTION_ASSAY = "5hmC_fraction"
PLOT_ASSAYS = (*ASSAYS, FRACTION_ASSAY)
PARQUET_COLUMNS = ("spot", "gene_id", "bin_index", "bin", "methylation_rate")


@dataclass
class ExpressionData:
    genes: pd.DataFrame
    spots: pd.DataFrame
    means: np.ndarray
    groups: np.ndarray
    threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--mrna-h5ad", type=Path, required=True,
                        help="One sample, with raw counts and annotations.")
    parser.add_argument("--taps-parquet", type=Path, required=True,
                        help="TAPS gene-bin Parquet (total modification).")
    parser.add_argument("--beta-parquet", type=Path, required=True,
                        help="TAPS-beta gene-bin Parquet (5mC).")
    parser.add_argument("--context", choices=("CG", "CA"), default="CG",
                        help="Cytosine context for plot and table labels.")
    parser.add_argument("--mrna-pairs", type=Path, required=True,
                        help="TSV columns: mrna-cell, beta-cell.")
    parser.add_argument("--taps-pairs", type=Path, required=True,
                        help="TSV columns: taps-cell, beta-cell.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New or empty directory; existing files refused.")
    parser.add_argument("--counts-layer", default="counts",
                        help="Raw count layer; use X explicitly for raw X.")
    parser.add_argument("--annotation-key", default="annotation")
    parser.add_argument("--annotations", nargs="+",
                        default=["L2/3/4", "L5", "L6"],
                        help="Exact labels to combine into Cortex.")
    parser.add_argument("--upstream-bins", type=int, default=20)
    parser.add_argument("--body-bins", type=int, default=100)
    parser.add_argument("--downstream-bins", type=int, default=20)
    parser.add_argument("--upstream-bp", type=int, default=2000)
    parser.add_argument("--downstream-bp", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=250_000,
                        help="Parquet rows decoded at once.")
    args = parser.parse_args()
    for name in ("upstream_bins", "body_bins", "downstream_bins",
                 "upstream_bp", "downstream_bp", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    return args


def normalized_counts(source: ad.AnnData, layer: str) -> sparse.csr_matrix:
    """Return sparse log1p(CP10K); implicit zeros stay zero."""
    if layer == "X":
        matrix = source.X
    else:
        if layer not in source.layers:
            raise ValueError(f"Missing raw-count layer: {layer}")
        matrix = source.layers[layer]
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    counts = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    if (not np.isfinite(counts.data).all()
            or (counts.data < 0).any()
            or not np.allclose(counts.data, np.rint(counts.data),
                               rtol=0, atol=1e-6)):
        raise ValueError("RNA requires finite nonnegative raw counts")
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError("RNA contains zero-count spots")
    counts = counts.multiply((10_000 / totals)[:, None]).tocsr()
    np.log1p(counts.data, out=counts.data)
    return counts


def read_expression(args: argparse.Namespace) -> ExpressionData:
    source = ad.read_h5ad(args.mrna_h5ad, backed="r")
    try:
        if not source.obs_names.is_unique or not source.var_names.is_unique:
            raise ValueError("RNA spot and gene IDs must be unique")
        required = {args.annotation_key, "array_row", "array_col"}
        if not required.issubset(source.obs):
            missing = required - set(source.obs)
            raise ValueError(f"Missing RNA obs fields: {missing}")
        if "sample" in source.obs:
            samples = source.obs["sample"]
            if samples.isna().any() or samples.nunique() != 1:
                raise ValueError("RNA matrix must contain exactly one sample")
        obs = source.obs
        annotations = obs[args.annotation_key]
        if (annotations.isna().any()
                or annotations.astype(str).str.strip().eq("").any()):
            raise ValueError("RNA annotations must not be missing or empty")
        region_ids = np.where(
            annotations.astype(str).isin(args.annotations), 0, 1
        )
        coordinates = obs[["array_row", "array_col"]].to_numpy(dtype=float)
        if (not np.isfinite(coordinates).all() or (coordinates < 0).any()
                or not np.equal(coordinates, np.floor(coordinates)).all()):
            raise ValueError("Array coordinates must be nonnegative integers")
        spot_ids = [f"{int(row):02d}_{int(col):02d}"
                    for row, col in coordinates]
        if len(set(spot_ids)) != len(spot_ids):
            raise ValueError("RNA has duplicate array coordinate pairs")
        spots = pd.DataFrame({
            "mrna_id": source.obs_names.astype(str), "mrna-cell": spot_ids,
            "annotation": annotations.astype(str).to_numpy(),
            "region": np.asarray(REGIONS)[region_ids],
            "region_index": region_ids,
        })
        genes = pd.DataFrame({
            "gene_id": source.var_names.astype(str),
            "gene_name": source.var.get(
                "feature_name", pd.Series(source.var_names,
                                          index=source.var_names)
            ).astype(str).to_numpy(),
        })
        if genes["gene_id"].eq("").any():
            raise ValueError("RNA gene IDs must not be empty")
        matrix = normalized_counts(source, args.counts_layer)
        threshold = float(np.median(matrix.data))
        means = np.full((len(REGIONS), source.n_vars), np.nan)
        groups = np.full(means.shape, -1, dtype=np.int8)
        for region_index, region in enumerate(REGIONS):
            selected = region_ids == region_index
            if not selected.any():
                print(f"No RNA spots in {region}; panel will be empty.",
                      flush=True)
                continue
            means[region_index] = np.asarray(
                matrix[selected].mean(axis=0)
            ).ravel()
            values = means[region_index]
            groups[region_index] = np.where(
                values == 0, 0, np.where(values <= threshold, 1, 2)
            )
        return ExpressionData(genes, spots, means, groups, threshold)
    finally:
        source.file.close()


def read_pairs(path: Path, moving: str) -> pd.DataFrame:
    columns = [f"{moving}-cell", "beta-cell"]
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if not set(columns).issubset(table):
        raise ValueError(f"{path}: required columns are {columns}")
    table = table[columns].copy()
    for column in columns:
        if not table[column].str.fullmatch(r"\d+_\d+").all():
            raise ValueError(f"{path}: invalid row_column IDs in {column}")
        table[column] = table[column].map(
            lambda value: "_".join(f"{int(part):02d}"
                                   for part in value.split("_"))
        )
        if table[column].duplicated().any():
            raise ValueError(f"{path}: {column} must map one to one")
    return table


def match_spots(args: argparse.Namespace, spots: pd.DataFrame) -> pd.DataFrame:
    """Use the same RNA/TAPS/beta triplets for both assays."""
    matched = spots.merge(read_pairs(args.mrna_pairs, "mrna"),
                          on="mrna-cell", how="inner", validate="one_to_one")
    matched = matched.merge(read_pairs(args.taps_pairs, "taps"),
                            on="beta-cell", how="inner", validate="one_to_one")
    if matched.empty:
        raise ValueError("No shared RNA/TAPS/beta spot triplets")
    return matched


def bin_labels(args: argparse.Namespace) -> list[str]:
    return ([f"up_{i + 1}" for i in range(args.upstream_bins)]
            + [f"body_{i + 1}" for i in range(args.body_bins)]
            + [f"down_{i + 1}" for i in range(args.downstream_bins)])


def aggregate_parquet(
    path: Path, spot_column: str, matches: pd.DataFrame,
    genes: pd.DataFrame, args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream rates into region/gene/bin sums and observation counts."""
    parquet = pq.ParquetFile(path)
    if not set(PARQUET_COLUMNS).issubset(parquet.schema_arrow.names):
        raise ValueError(f"{path}: required fields are {PARQUET_COLUMNS}")
    labels = bin_labels(args)
    shape = (len(REGIONS), len(genes), len(labels))
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.uint32)
    spot_index = pd.Index(matches[spot_column])
    gene_index = pd.Index(genes["gene_id"])
    regions = matches["region_index"].to_numpy()
    arrow_spots = pa.array(spot_index)
    arrow_genes = pa.array(gene_index)
    selected_rows = 0
    for batch_index, batch in enumerate(parquet.iter_batches(
        batch_size=args.batch_size, columns=list(PARQUET_COLUMNS),
    )):
        # Filter in Arrow before decoding string columns into Python objects.
        table = pa.Table.from_batches([batch])
        selected = pc.and_(
            pc.is_in(table["spot"], value_set=arrow_spots),
            pc.is_in(table["gene_id"], value_set=arrow_genes),
        )
        table = table.filter(selected)
        if not len(table):
            continue
        if any(table[name].null_count for name in PARQUET_COLUMNS):
            raise ValueError(f"{path}: missing values in selected rows")
        frame = table.to_pandas()
        if frame.duplicated(["spot", "gene_id", "bin_index"]).any():
            raise ValueError(f"{path}: duplicate spot/gene/bin rows in batch")
        bin_values = frame["bin_index"].to_numpy(dtype=float)
        rates = frame["methylation_rate"].to_numpy(dtype=float)
        if (not np.isfinite(bin_values).all()
                or not np.equal(bin_values, np.floor(bin_values)).all()
                or (bin_values < 0).any()
                or (bin_values >= len(labels)).any()):
            raise ValueError(f"{path}: incompatible bin indices")
        bin_indices = bin_values.astype(np.intp)
        if not np.array_equal(frame["bin"].to_numpy(),
                              np.asarray(labels)[bin_indices]):
            raise ValueError(f"{path}: bin labels do not match the layout")
        if not np.isfinite(rates).all() or ((rates < 0) | (rates > 1)).any():
            raise ValueError(f"{path}: rates must be finite in [0, 1]")
        positions = spot_index.get_indexer(frame["spot"])
        gene_indices = gene_index.get_indexer(frame["gene_id"])
        indices = (regions[positions], gene_indices, bin_indices)
        np.add.at(sums, indices, rates)
        np.add.at(counts, indices, 1)
        selected_rows += len(frame)
        if (batch_index + 1) % 100 == 0:
            print(f"{path.name}: {batch_index + 1} batches read", flush=True)
    if not selected_rows:
        raise ValueError(f"{path}: no matching spot/gene rows")
    # A gene/bin can have at most one row per matched spot in its region.
    region_spots = np.bincount(regions, minlength=len(REGIONS))
    if (counts > region_spots[:, None, None]).any():
        raise ValueError(f"{path}: repeated spot/gene/bin rows detected")
    np.divide(sums, counts, out=sums, where=counts > 0)
    sums[counts == 0] = np.nan
    return sums, counts


def summarize_profiles(
    expression: ExpressionData, taps: np.ndarray, beta: np.ndarray,
    taps_counts: np.ndarray, beta_counts: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Average the same supported genes in each assay and their difference."""
    rows = []
    for region_index, region in enumerate(REGIONS):
        for group_index, group in enumerate(GROUPS):
            selected = expression.groups[region_index] == group_index
            for bin_index, label in enumerate(bin_labels(args)):
                supported = (
                    selected
                    & (taps_counts[region_index, :, bin_index] > 0)
                    & (beta_counts[region_index, :, bin_index] > 0)
                )
                total = taps[region_index, supported, bin_index]
                methyl = beta[region_index, supported, bin_index]
                assay_values = (total, methyl, total - methyl)
                for assay, values in zip(ASSAYS, assay_values):
                    rows.append({
                        "context": args.context,
                        "region": region, "expression_group": group,
                        "assay": assay, "bin_index": bin_index, "bin": label,
                        "methylation_rate": float(values.mean())
                        if len(values) else np.nan,
                        "supported_genes": int(supported.sum()),
                        "expression_group_genes": int(selected.sum()),
                        "taps_spot_gene_bins": int(taps_counts[
                            region_index, supported, bin_index].sum()),
                        "beta_spot_gene_bins": int(beta_counts[
                            region_index, supported, bin_index].sum()),
                    })
    return add_hmc_fraction(pd.DataFrame(rows))


def add_hmc_fraction(profiles: pd.DataFrame) -> pd.DataFrame:
    """Add the ratio of group-mean 5hmC to group-mean total modification."""
    keys = ["context", "region", "expression_group", "bin_index"]
    total = profiles.loc[profiles["assay"].eq("TAPS")].set_index(keys)
    hydroxyl = profiles.loc[profiles["assay"].eq("5hmC")].set_index(keys)
    fraction = total.copy()
    fraction["assay"] = FRACTION_ASSAY
    fraction["methylation_rate"] = (
        hydroxyl["methylation_rate"].reindex(total.index)
        / total["methylation_rate"].where(total["methylation_rate"] > 0)
    )
    return pd.concat([
        profiles.loc[profiles["assay"].ne(FRACTION_ASSAY)],
        fraction.reset_index(),
    ], ignore_index=True)


def plot_profiles(profiles: pd.DataFrame, args: argparse.Namespace) -> None:
    profiles = add_hmc_fraction(profiles)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans",
        ],
        "font.size": 5, "axes.titlesize": 7, "axes.labelsize": 6,
        "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 5,
    })
    bin_count = len(bin_labels(args))
    start = args.upstream_bins - 0.5
    end = args.upstream_bins + args.body_bins - 0.5
    axis_limits = {}
    for assay in PLOT_ASSAYS:
        values = profiles.loc[
            profiles["assay"].eq(assay), "methylation_rate"
        ].dropna().to_numpy() * 100
        if not len(values):
            axis_limits[assay] = (0.0, 1.0)
            continue
        lower, upper = float(values.min()), float(values.max())
        padding = ((upper - lower) * 0.05 if upper > lower
                   else max(abs(lower) * 0.05, 0.01))
        axis_limits[assay] = (lower - padding, upper + padding)
    assay_labels = (
        f"5m{args.context} + 5hm{args.context}",
        f"5m{args.context}", f"5hm{args.context}", "5hmC / (5mC + 5hmC)",
    )
    fig, axes = plt.subplots(len(REGIONS), len(PLOT_ASSAYS), figsize=(5, 2),
                             sharex="col", squeeze=False)
    for row_index, region in enumerate(REGIONS):
        for axis, assay, title in zip(
            axes[row_index], PLOT_ASSAYS, assay_labels
        ):
            frame = profiles.loc[
                profiles["region"].eq(region) & profiles["assay"].eq(assay)
            ]
            for group, label, color in zip(GROUPS, GROUP_LABELS, GROUP_COLORS):
                values = frame.loc[frame["expression_group"].eq(group)]
                axis.plot(values["bin_index"],
                          values["methylation_rate"] * 100,
                          color=color, label=label, linewidth=0.8)
            for boundary in (start, end):
                axis.axvline(boundary, color="0.65", linewidth=0.4,
                             linestyle="--", zorder=0)
            axis.set_ylim(*axis_limits[assay])
            if (assay in ("5hmC", FRACTION_ASSAY)
                    and axis_limits[assay][0] <= 0 <= axis_limits[assay][1]):
                axis.axhline(0, color="0.65", linewidth=0.4, zorder=0)
            axis.set_xlim(-0.5, bin_count - 0.5)
            axis.set_xticks(
                [-0.5, start, end, bin_count - 0.5],
                [f"-{args.upstream_bp / 1000:g}kb", "TSS", "TES",
                 f"+{args.downstream_bp / 1000:g}kb"],
            )
            tick_labels = axis.get_xticklabels()
            for label, alignment in zip(
                tick_labels, ("center", "left", "right", "center")
            ):
                label.set_ha(alignment)
            if row_index == 0:
                axis.set_title(title, pad=3)
            axis.tick_params(length=2, pad=1)
            axis.spines[["top", "right"]].set_visible(False)
            if not frame["methylation_rate"].notna().any():
                axis.text(0.5, 0.5, "No shared coverage", ha="center",
                          va="center", transform=axis.transAxes, fontsize=5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.supylabel("Modification (%)", x=0.015, fontsize=6)
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.01),
               handlelength=1.4, columnspacing=1)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.88, bottom=0.2,
                        wspace=0.45, hspace=0.28)
    for row_index, region in enumerate(REGIONS):
        bounds = axes[row_index, 0].get_position()
        fig.text(0.065, (bounds.y0 + bounds.y1) / 2, region,
                 rotation=90, ha="center", va="center", fontsize=6)
    fig.savefig(args.output_dir / f"{args.context.lower()}-methylation.png",
                dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_outputs(
    expression: ExpressionData, profiles: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    gene_groups = []
    for region_index, region in enumerate(REGIONS):
        frame = expression.genes.copy()
        frame["region"] = region
        frame["mean_log1p_cp10k"] = expression.means[region_index]
        frame["expression_group"] = [
            GROUPS[index] if index >= 0 else "no_rna_spots"
            for index in expression.groups[region_index]
        ]
        gene_groups.append(frame)
    pd.concat(gene_groups, ignore_index=True).to_csv(
        args.output_dir / "expression-groups.tsv.gz", sep="\t", index=False
    )
    profiles.to_csv(args.output_dir / "methylation-profiles.tsv",
                    sep="\t", index=False, na_rep="NA")
    plot_profiles(profiles, args)


def main() -> None:
    args = parse_args()
    try:
        for name in ("mrna_h5ad", "taps_parquet", "beta_parquet",
                     "mrna_pairs", "taps_pairs"):
            path = getattr(args, name)
            if path is not None and not path.is_file():
                raise ValueError(f"Input does not exist: {path}")
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("Output directory must be new or empty")
        expression = read_expression(args)
        matches = match_spots(args, expression.spots)
        print(f"RNA spots={len(expression.spots):,}; "
              f"matched triplets={len(matches):,}; "
              f"expression threshold={expression.threshold:.10g}", flush=True)
        taps, taps_counts = aggregate_parquet(
            args.taps_parquet, "taps-cell", matches, expression.genes, args
        )
        beta, beta_counts = aggregate_parquet(
            args.beta_parquet, "beta-cell", matches, expression.genes, args
        )
        profiles = summarize_profiles(expression, taps, beta,
                                      taps_counts, beta_counts, args)
        if not profiles["methylation_rate"].notna().any():
            raise ValueError("TAPS and beta have no shared regional gene-bins")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_outputs(expression, profiles, args)
    except (OSError, ValueError, KeyError, pa.ArrowException) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"Saved profiles, expression groups and figures: {args.output_dir}")


if __name__ == "__main__":
    main()
