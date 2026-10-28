#!/usr/bin/env python3
"""Step 3: discover mRNA programs from counts, then fit a selected K."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cnmf import cNMF
from cnmf.cnmf import load_df_from_npz, save_df_to_npz


COMPONENTS = (6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30)
N_REPEATS = 100
MAX_NMF_ITER = 1000
N_TOP_GENES = 3000
TOP_GENES_PER_PROGRAM = 50
RANDOM_SEED = 42
N_THREADS = 4
DENSITY_THRESHOLD = 0.5
COUNTS_LAYER = "counts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        help="Integrated H5AD with raw counts; required for a new run.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Exact run directory.",
    )
    parser.add_argument(
        "--selected-k", type=int,
        help="Fit consensus for a tested K using an existing run.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue factorization using saved counts and parameters.",
    )
    args = parser.parse_args()
    if args.selected_k is not None and args.selected_k < 2:
        parser.error("--selected-k must be at least 2")
    if sum((args.input is not None, args.resume,
            args.selected_k is not None)) != 1:
        parser.error("Choose one: --input, --resume, or --selected-k")
    return args


def configure_plots() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans",
        ],
        "axes.titlesize": 7, "axes.labelsize": 6,
        "xtick.labelsize": 5, "ytick.labelsize": 5,
        "legend.fontsize": 5, "axes.titlepad": 3,
    })


def read_counts(path: Path) -> ad.AnnData:
    if not path.is_file():
        raise ValueError(f"Input H5AD does not exist: {path}")
    source = ad.read_h5ad(path, backed="r")
    try:
        if not source.obs_names.is_unique or not source.var_names.is_unique:
            raise ValueError("Spot and gene IDs must be unique")
        if COUNTS_LAYER not in source.layers:
            raise ValueError(f"Missing layers[{COUNTS_LAYER!r}]")
        counts = sparse.csr_matrix(
            source.layers[COUNTS_LAYER], dtype=np.float64, copy=True,
        )
        counts.sum_duplicates()
        if not np.isfinite(counts.data).all() or (counts.data < 0).any():
            raise ValueError("Counts must be finite and nonnegative")
        if not np.allclose(counts.data, counts.data.round(), atol=1e-5,
                           rtol=0):
            raise ValueError("Use integer raw counts, not transformed data")
        if (np.asarray(counts.sum(axis=1)).ravel() <= 0).any():
            raise ValueError("Input includes zero-count spots")
        keep = np.asarray(counts.sum(axis=0)).ravel() > 0
        data = ad.AnnData(
            X=counts[:, keep], obs=source.obs.copy(),
            var=source.var.loc[keep].copy(),
        )
    finally:
        source.file.close()
    if data.n_obs < 3 or data.n_vars < 2:
        raise ValueError("At least three spots and two expressed genes needed")
    if max(COMPONENTS) > min(data.n_obs, data.n_vars, N_TOP_GENES):
        raise ValueError("Candidate K exceeds spot or factorization gene count")
    data.uns["cnmf_input"] = {
        "source": str(path), "layer": COUNTS_LAYER,
        "batch_corrected": False,
    }
    print(f"Using {data.n_obs:,} spots and {data.n_vars:,} expressed genes",
          flush=True)
    print("Using raw counts; PCA/Harmony/UMAP are not factorized", flush=True)
    return data


def require_preparation(run: Path) -> None:
    name = run.name
    required = [run / f"{name}.counts-input.h5ad",
                run / f"{name}.overdispersed_genes.txt"]
    required.extend(
        run / "cnmf_tmp" / f"{name}.{suffix}" for suffix in (
            "norm_counts.h5ad", "nmf_params.df.npz",
            "nmf_idvrun_params.yaml", "tpm.h5ad", "tpm_stats.df.npz",
        )
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Incomplete preparation: " + ", ".join(missing))


def plot_k_selection(model: cNMF, run: Path) -> None:
    parameters = load_df_from_npz(model.paths["nmf_replicate_parameters"])
    normalized = ad.read_h5ad(model.paths["normalized_counts"])
    rows = []
    for k in sorted(set(parameters["n_components"].astype(int))):
        result = model.consensus(
            k=k, skip_density_and_return_after_stats=True,
            show_clustering=False, close_clustergram_fig=True,
            norm_counts=normalized,
        )
        rows.append(result.stats)
    stats = pd.DataFrame(rows).reset_index(drop=True)
    save_df_to_npz(stats, model.paths["k_selection_stats"])
    stats.to_csv(run / "k-selection.csv", index=False)
    figure, axis = plt.subplots(figsize=(4, 2))
    error_axis = axis.twinx()
    axis.plot(stats["k"], stats["silhouette"], "o-", color="#4DBBD5",
              markersize=3)
    error_axis.plot(stats["k"], stats["prediction_error"], "o-",
                    color="#E64B35", markersize=3)
    axis.set_xlabel("Number of programs (K)")
    axis.set_ylabel("Stability", color="#4DBBD5", labelpad=5)
    error_axis.set_ylabel("Reconstruction error", color="#E64B35")
    axis.set_xticks(stats["k"])
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.set_title("cNMF K selection")
    figure.savefig(model.paths["k_selection_plot"], dpi=300,
                   bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def factorize(model: cNMF, run: Path, input_path: Path | None) -> None:
    if input_path is not None:
        data = read_counts(input_path)
        counts_path = run / f"{run.name}.counts-input.h5ad"
        data.write_h5ad(counts_path)
        print(f"Preparing K={COMPONENTS}, {N_REPEATS} repeats per K",
              flush=True)
        model.prepare(
            counts_fn=str(counts_path), components=COMPONENTS,
            n_iter=N_REPEATS, seed=RANDOM_SEED,
            num_highvar_genes=min(N_TOP_GENES, data.n_vars),
            beta_loss="frobenius", max_NMF_iter=MAX_NMF_ITER,
        )
        del data
    else:
        print("Resuming saved counts, K values, genes and NMF parameters",
              flush=True)
    model.update_nmf_iter_params()
    model.factorize(worker_i=0, total_workers=1, skip_completed_runs=True)
    model.combine(skip_missing_files=False)
    plot_k_selection(model, run)
    print(f"Inspect {model.paths['k_selection_plot']}", flush=True)
    print("Then run this script with --output-dir and --selected-k",
          flush=True)


def fit_consensus(model: cNMF, run: Path, k: int) -> None:
    parameters = load_df_from_npz(model.paths["nmf_replicate_parameters"])
    candidates = set(parameters["n_components"].astype(int))
    if k not in candidates:
        raise ValueError(
            f"K={k} was not tested; choose from {sorted(candidates)}"
        )
    if not Path(model.paths["merged_spectra"] % k).is_file():
        raise ValueError("Factorization is incomplete; finish --resume first")
    threshold = str(DENSITY_THRESHOLD).replace(".", "_")
    outputs = [run / f"usage.k_{k}.csv",
               run / f"program-top-genes.k_{k}.csv",
               run / f"program-density.k_{k}.png"]
    outputs.extend(Path(model.paths[key] % (k, threshold)) for key in (
        "consensus_spectra__txt", "consensus_usages__txt",
        "gene_spectra_score__txt", "gene_spectra_tpm__txt",
    ))
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise ValueError("Consensus outputs already exist: "
                         + ", ".join(existing))
    model.consensus(k=k, density_threshold=DENSITY_THRESHOLD,
                    show_clustering=False, build_ref=False,
                    close_clustergram_fig=True)
    usage, scores, weights, top_genes = model.load_results(
        K=k, density_threshold=DENSITY_THRESHOLD,
        n_top_genes=TOP_GENES_PER_PROGRAM, norm_usage=True,
    )
    prepared = ad.read_h5ad(run / f"{run.name}.counts-input.h5ad",
                            backed="r")
    try:
        metadata = prepared.obs.copy()
        gene_names = (
            prepared.var["feature_name"].fillna("").astype(str).to_dict()
            if "feature_name" in prepared.var else {}
        )
    finally:
        prepared.file.close()
    usage.columns = [f"program_{column}" for column in usage.columns]
    usage.index.name = "spot_id"
    metadata = metadata[[key for key in ("sample", "leiden")
                         if key in metadata]]
    metadata.join(usage, how="right").to_csv(outputs[0], index_label="spot_id")
    rows = []
    for program in top_genes:
        for rank, gene in enumerate(top_genes[program].dropna(), start=1):
            rows.append({
                "program": program, "rank": rank, "gene_id": gene,
                "gene_name": gene_names.get(gene) or gene,
                "gene_score": scores.loc[gene, program],
                "weight_tpm": weights.loc[gene, program],
            })
    pd.DataFrame(rows).to_csv(outputs[1], index=False)
    density = load_df_from_npz(model.paths["local_density_cache"] % k)
    figure, axis = plt.subplots(figsize=(4, 2))
    axis.hist(density.iloc[:, 0], bins=40, color="#4DBBD5")
    axis.axvline(DENSITY_THRESHOLD, color="#E64B35", linewidth=1,
                 label=f"Threshold = {DENSITY_THRESHOLD}")
    axis.set_xlabel("Mean distance to neighboring programs")
    axis.set_ylabel("Number of replicate programs", labelpad=5)
    axis.set_title(f"cNMF program filtering (K={k})")
    axis.legend(frameon=False)
    figure.savefig(outputs[2], dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(f"Saved {len(usage):,} spot usages and K={k} program gene tables",
          flush=True)


def run(args: argparse.Namespace) -> None:
    run_dir = args.output_dir.expanduser().resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_dir.name):
        raise ValueError("Run directory name must use letters, digits, . _ -")
    if args.input is not None:
        if run_dir.exists() and (not run_dir.is_dir()
                                 or any(run_dir.iterdir())):
            raise ValueError("Use an empty output directory for a new run")
    else:
        require_preparation(run_dir)
    configure_plots()
    model = cNMF(output_dir=str(run_dir.parent), name=run_dir.name)
    with threadpool_limits(limits=N_THREADS):
        if args.selected_k is not None:
            fit_consensus(model, run_dir, args.selected_k)
        else:
            input_path = (args.input.expanduser().resolve()
                          if args.input is not None else None)
            factorize(model, run_dir, input_path)
    print(f"cNMF results: {run_dir}", flush=True)


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error
    finally:
        plt.close("all")


if __name__ == "__main__":
    main()
