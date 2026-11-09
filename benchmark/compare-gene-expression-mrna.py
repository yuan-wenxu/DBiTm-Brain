#!/usr/bin/env python3
"""Compare mean log1p(CP10K) gene expression between two QC mRNA H5ADs.

Normalize each spot using its total counts across all measured genes, apply
log1p, then average each gene across all spots, including zeros. Pair genes
by var_names; do not pair spots or use integrated/imputed expression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


TARGET_SUM = 10_000.0
EXPRESSION_COLUMN = "mean_log1p_cp10k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input1", type=Path, required=True,
        help="First QC mRNA H5AD with layers['counts']; x-axis sample.",
    )
    parser.add_argument(
        "--input2", type=Path, required=True,
        help="Second QC mRNA H5AD with layers['counts']; y-axis sample.",
    )
    parser.add_argument(
        "--x-label", default="Input 1: mean log1p(CP10K)",
        help="Full x-axis title for input1.",
    )
    parser.add_argument(
        "--y-label", default="Input 2: mean log1p(CP10K)",
        help="Full y-axis title for input2.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-spots-per-gene", type=int, default=1,
        help=(
            "Retain shared genes detected in at least this many spots in "
            "either sample (default: 1; excludes genes zero in both)."
        ),
    )
    args = parser.parse_args()
    if args.min_spots_per_gene < 1:
        parser.error("--min-spots-per-gene must be positive")
    args.input1 = args.input1.expanduser().resolve()
    args.input2 = args.input2.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.input1 == args.input2:
        parser.error("The two inputs must be distinct H5AD files")
    return args


def aggregate_expression(path: Path) -> tuple[pd.DataFrame, dict]:
    """Recompute spot normalization from raw counts before gene filtering."""
    data = ad.read_h5ad(path, backed="r")
    try:
        if data.n_obs == 0 or data.n_vars == 0:
            raise ValueError(f"Empty expression matrix: {path}")
        if not data.var_names.is_unique or not data.obs_names.is_unique:
            raise ValueError(f"Gene and spot IDs must be unique: {path}")
        if "counts" not in data.layers:
            raise ValueError(f"Missing layers['counts']: {path}")
        counts = sparse.csr_matrix(
            data.layers["counts"], dtype=np.float64, copy=True,
        )
        counts.sum_duplicates()
        counts.eliminate_zeros()
        if (
            not np.isfinite(counts.data).all()
            or (counts.data < 0).any()
            or not np.allclose(
                counts.data, np.rint(counts.data), rtol=0, atol=1e-5,
            )
        ):
            raise ValueError(
                f"Counts must be finite nonnegative integers: {path}"
            )
        totals = np.asarray(counts.sum(axis=1)).ravel()
        if (totals <= 0).any():
            raise ValueError(f"Some spots have zero total counts: {path}")
        table = pd.DataFrame(index=data.var_names.copy())
        table.index.name = "gene_id"
        if "feature_name" in data.var:
            table["gene_name"] = data.var["feature_name"].to_numpy()
        table["total_counts"] = np.asarray(counts.sum(axis=0)).ravel()
        table["detected_spots"] = np.asarray((counts > 0).sum(axis=0)).ravel()
        normalized = counts.multiply((TARGET_SUM / totals)[:, None]).tocsr()
        table["mean_cp10k"] = np.asarray(normalized.mean(axis=0)).ravel()
        normalized.data = np.log1p(normalized.data)
        table[EXPRESSION_COLUMN] = np.asarray(normalized.mean(axis=0)).ravel()
        summary = {
            "path": str(path), "spots": data.n_obs, "genes": data.n_vars,
            "total_counts": float(totals.sum()),
        }
    finally:
        data.file.close()
    print(
        f"{path.name}: {summary['spots']:,} spots, "
        f"{summary['genes']:,} genes", flush=True,
    )
    return table, summary


def compare_genes(
    input1: pd.DataFrame, input2: pd.DataFrame, min_spots: int,
) -> tuple[pd.DataFrame, dict]:
    pairs = input1.add_suffix("_input1").join(
        input2.add_suffix("_input2"), how="inner",
    ).sort_index()
    pairs["used_for_comparison"] = (
        pairs["detected_spots_input1"].ge(min_spots)
        | pairs["detected_spots_input2"].ge(min_spots)
    )
    selected = pairs.loc[pairs["used_for_comparison"]]
    if selected.empty:
        raise ValueError("No shared genes pass the detection threshold")
    x = selected[f"{EXPRESSION_COLUMN}_input1"]
    y = selected[f"{EXPRESSION_COLUMN}_input2"]
    variable = len(selected) > 1 and x.nunique() > 1 and y.nunique() > 1
    statistics = {
        "shared_genes": len(pairs),
        "input1_only_genes": len(input1.index.difference(input2.index)),
        "input2_only_genes": len(input2.index.difference(input1.index)),
        "compared_genes": len(selected),
        "excluded_shared_genes": len(pairs) - len(selected),
        "min_spots_per_gene_in_either_sample": min_spots,
        "pearson_r": float(x.corr(y)) if variable else None,
        "spearman_r": float(x.corr(y, method="spearman"))
        if variable else None,
        "rmse_mean_log1p_cp10k": float(np.sqrt(np.mean((y - x) ** 2))),
    }
    return pairs, statistics


def draw_scatter(
    pairs: pd.DataFrame, statistics: dict, path: Path,
    x_label: str, y_label: str,
) -> None:
    values = pairs.loc[pairs["used_for_comparison"]]
    x = values[f"{EXPRESSION_COLUMN}_input1"]
    y = values[f"{EXPRESSION_COLUMN}_input2"]
    upper = max(float(x.max()), float(y.max())) * 1.05
    limits = (0, max(upper, 0.1))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans",
        ],
        "font.size": 5, "axes.titlesize": 7, "axes.labelsize": 6,
        "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 5,
    })
    figure, axis = plt.subplots(figsize=(4, 2))
    axis.scatter(x, y, s=2, c="#4DBBD5", alpha=0.45, linewidths=0)
    axis.plot(limits, limits, "--", color="0.4", linewidth=0.6)
    axis.set(xlim=limits, ylim=limits)
    axis.set_aspect("equal", adjustable="box")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label, labelpad=5)
    axis.set_title("mRNA reproducibility", pad=3)
    pearson = statistics["pearson_r"]
    correlation = "NA" if pearson is None else f"{pearson:.3f}"
    axis.text(
        0.03, 0.97,
        f"n = {len(values):,} genes\nPearson r = {correlation}",
        transform=axis.transAxes, va="top", fontsize=5,
    )
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(pad=0.4)
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    try:
        if args.output_dir.exists():
            raise ValueError(
                f"Output directory already exists: {args.output_dir}"
            )
        input1, summary1 = aggregate_expression(args.input1)
        input2, summary2 = aggregate_expression(args.input2)
        pairs, statistics = compare_genes(
            input1, input2, args.min_spots_per_gene,
        )
        statistics.update({
            "input1": summary1, "input2": summary2,
            "x_label": args.x_label, "y_label": args.y_label,
            "normalization": {
                "source": "layers['counts']",
                "target_sum_per_spot": TARGET_SUM,
                "formula": "mean_spots(log1p(10000 * count / spot_total))",
                "spot_total": "Sum across all genes before gene selection",
                "zero_expression_spots": "Included in each gene mean",
                "gene_pairing": "Intersection of var_names (gene IDs)",
            },
        })
        args.output_dir.mkdir(parents=True, exist_ok=False)
        pairs.to_csv(
            args.output_dir / "per_gene_comparison.tsv", sep="\t",
            index_label="gene_id", na_rep="NA",
        )
        draw_scatter(
            pairs, statistics, args.output_dir / "mRNA_input1_vs_input2.png",
            args.x_label, args.y_label,
        )
        result = json.dumps(statistics, indent=2, allow_nan=False)
        (args.output_dir / "statistics.json").write_text(
            result + "\n", encoding="utf-8",
        )
        print(result, flush=True)
        print(f"Saved comparison to {args.output_dir}", flush=True)
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
