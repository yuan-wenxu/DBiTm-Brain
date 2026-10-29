"""Score spatial LR pairs and evaluate NMF pattern counts with SpaLRFactor."""

import argparse
from pathlib import Path
import tempfile

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from spalrfactor import evaluate_nmf, score_lr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


DIRECTION_COLORS = {"incoming": "#4DBBD5", "outgoing": "#E64B35"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--h5ad", type=Path, required=True,
                        help="Cortical H5AD exported by script 03.")
    parser.add_argument("--pairs", type=Path, required=True,
                        help="shared-lr-pairs.csv exported by script 01.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--n-patterns", nargs="+", type=int, default=[6, 8, 10, 12, 14, 16, 18, 20],
        help="Candidate pattern counts for evaluate_nmf.",
    )
    parser.add_argument(
        "--direction", choices=["incoming", "outgoing", "both"], default="both",
    )
    parser.add_argument("--radius", type=float, default=250.0,
                        help="Neighborhood radius in micrometers.")
    parser.add_argument("--sigma", type=float, default=125.0,
                        help="Gaussian distance scale in micrometers.")
    parser.add_argument("--include-self", action="store_true")
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a new directory")
    if (min(args.n_patterns) < 1
            or len(set(args.n_patterns)) != len(args.n_patterns)):
        parser.error("--n-patterns must contain unique positive ranks")
    if args.n_repeats < 2 or args.max_iter < 1:
        parser.error("--n-repeats must be >= 2 and --max-iter must be positive")
    if args.random_state < 0:
        parser.error("--random-state must be nonnegative")
    if not np.isfinite(args.tol) or args.tol < 0:
        parser.error("--tol must be finite and nonnegative")
    for name in ("radius", "sigma", "target_sum"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            option = name.replace("_", "-")
            parser.error(f"--{option} must be finite and positive")
    return args


def save_scores(scores: ad.AnnData, output: Path) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        scores.write_h5ad(temporary_path, compression="gzip")
        temporary_path.replace(output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def plot_evaluation(summaries: dict[str, pd.DataFrame], output: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans",
        ],
        "font.size": 5, "axes.titlesize": 7, "axes.labelsize": 6,
        "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 5,
    })
    figure, axes = plt.subplots(1, 3, figsize=(7, 2), layout="constrained")
    for direction, summary in summaries.items():
        summary = summary.sort_values("n_patterns")
        ranks = summary["n_patterns"].to_numpy()
        values = [
            summary["stability"], summary["relative_error_mean"],
            summary["converged_repeats"] / summary["n_repeats"],
        ]
        for axis, metric in zip(axes, values):
            axis.plot(
                ranks, metric, color=DIRECTION_COLORS[direction],
                marker="o" if direction == "incoming" else "s",
                markersize=2.5, linewidth=0.8, label=direction.capitalize(),
            )
    all_ranks = sorted({
        rank for summary in summaries.values() for rank in summary["n_patterns"]
    })
    tick_indices = np.linspace(0, len(all_ranks) - 1,
                               min(8, len(all_ranks)), dtype=int)
    titles = ["Stability", "Reconstruction", "Convergence"]
    labels = ["Cosine similarity", "Mean relative error", "Converged runs"]
    for axis, title, label in zip(axes, titles, labels):
        axis.set_title(title, pad=3)
        axis.set_xlabel("Patterns")
        axis.set_ylabel(label, labelpad=5)
        axis.set_xticks([all_ranks[index] for index in tick_indices])
        axis.margins(y=0.08)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.tick_params(length=2, pad=1)
        axis.spines[["top", "right"]].set_visible(False)
    axes[2].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    figure.legend(
        *axes[0].get_legend_handles_labels(), loc="outside lower center",
        ncol=len(summaries), frameon=False,
    )
    figure.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    data = ad.read_h5ad(args.h5ad)
    pairs = pd.read_csv(args.pairs, encoding="utf-8-sig")
    print(f"Scoring {data.n_obs} spots and {len(pairs)} LR pairs", flush=True)
    scores = score_lr(
        data, pairs, radius=args.radius, sigma=args.sigma,
        include_self=args.include_self, target_sum=args.target_sum,
    )
    del data
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_scores(scores, args.output_dir / "lr-scores.h5ad")
    print(f"Saved scores to {args.output_dir / 'lr-scores.h5ad'}", flush=True)
    available = scores.obs["score_available"].to_numpy(dtype=bool)
    evaluation_scores = scores[available].copy()
    if not evaluation_scores.n_obs:
        raise ValueError("No spots with spatial neighbors available for NMF")
    directions = (
        ["incoming", "outgoing"] if args.direction == "both"
        else [args.direction]
    )
    summaries = {}
    for direction in directions:
        output_dir = args.output_dir / direction
        output_dir.mkdir()
        print(f"{direction}: evaluating ranks {args.n_patterns} on "
              f"{evaluation_scores.n_obs} spots x {evaluation_scores.n_vars} "
              "LR pairs", flush=True)
        summary, diagnostics = evaluate_nmf(
            evaluation_scores, args.n_patterns,
            layer=None if direction == "incoming" else "outgoing",
            n_repeats=args.n_repeats, max_iter=args.max_iter,
            tol=args.tol, random_state=args.random_state,
        )
        summary.to_csv(output_dir / "nmf_evaluation.csv", index=False)
        diagnostics.to_csv(output_dir / "nmf_run_diagnostics.csv", index=False)
        summaries[direction] = summary
    plot_evaluation(summaries, args.output_dir / "nmf_evaluation.png")
    print(f"Spots included in evaluation: {int(available.sum())} / "
          f"{scores.n_obs}")
    print(f"Saved NMF evaluation results to {args.output_dir}")


if __name__ == "__main__":
    main()
