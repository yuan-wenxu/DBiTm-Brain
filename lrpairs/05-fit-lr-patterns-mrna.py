"""Fit a selected NMF rank to saved SpaLRFactor scores and export factors."""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from spalrfactor import fit_nmf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--scores", type=Path, required=True,
        help="lr-scores.h5ad exported by script 04.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-patterns", type=int, required=True)
    parser.add_argument(
        "--direction", choices=["incoming", "outgoing", "both"], default="both",
    )
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a new directory")
    if min(args.n_patterns, args.n_repeats, args.max_iter) < 1:
        parser.error(
            "Pattern count, repeats and max iterations must be positive"
        )
    if not np.isfinite(args.tol) or args.tol < 0:
        parser.error("--tol must be finite and nonnegative")
    if args.random_state < 0:
        parser.error("--random-state must be nonnegative")
    return args


def export_factors(
    scores: ad.AnnData, weights: pd.DataFrame, loadings: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Export metadata and L1-scaled factors while preserving W @ H."""
    scales = loadings.sum(axis=1)
    weights = weights.mul(scales, axis=1)
    loadings = loadings.div(scales.where(scales > 0, 1), axis=0)
    patterns = weights.columns.tolist()

    usage = scores.obs.copy()
    usage["nmf_included"] = usage.index.isin(weights.index)
    usage[patterns] = weights.reindex(usage.index).to_numpy()
    usage.to_csv(output_dir / "nmf_spot_usage.csv", index_label="spot_id",
                 encoding="utf-8-sig")

    pairs = scores.var.copy()
    pairs["lr_pair"] = scores.var_names
    pairs["nmf_included"] = pairs.index.isin(loadings.columns)
    pairs[patterns] = loadings.T.reindex(pairs.index).to_numpy()
    pairs.to_csv(output_dir / "nmf_pair_loadings.csv", index=False,
                 encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    scores = ad.read_h5ad(args.scores)
    if "score_available" not in scores.obs:
        raise ValueError("Expected obs['score_available'] from script 04")
    if not scores.obs_names.is_unique or not scores.var_names.is_unique:
        raise ValueError("Spot IDs and LR pair IDs must be unique")
    available = scores.obs["score_available"].to_numpy(dtype=bool)
    fitting_scores = scores[available].copy()
    if not fitting_scores.n_obs or not fitting_scores.n_vars:
        raise ValueError("No available score matrix for NMF")
    if args.n_patterns > min(fitting_scores.shape):
        raise ValueError(
            f"--n-patterns must not exceed min{fitting_scores.shape} "
            "after excluding unavailable spots"
        )
    directions = (
        ["incoming", "outgoing"] if args.direction == "both"
        else [args.direction]
    )
    if "outgoing" in directions and "outgoing" not in scores.layers:
        raise ValueError("Scores are missing layers['outgoing']")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for direction in directions:
        print(f"{direction}: fitting {args.n_patterns} patterns on "
              f"{fitting_scores.n_obs} spots x {fitting_scores.n_vars} "
              f"LR pairs ({args.n_repeats} initializations)", flush=True)
        weights, loadings = fit_nmf(
            fitting_scores, args.n_patterns,
            layer=None if direction == "incoming" else "outgoing",
            n_repeats=args.n_repeats, max_iter=args.max_iter,
            tol=args.tol, random_state=args.random_state,
        )
        output_dir = args.output_dir / direction
        output_dir.mkdir()
        export_factors(scores, weights, loadings, output_dir)
        print(f"Saved {direction} factors to {output_dir}", flush=True)
    print(f"Spots included in NMF: {int(available.sum())} / {scores.n_obs}")


if __name__ == "__main__":
    main()
