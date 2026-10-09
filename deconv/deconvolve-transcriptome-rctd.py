#!/usr/bin/env python3
"""Deconvolve spatial RNA counts with rctd-py using a subclass-labelled reference.

Reference dataset: DevVIS scRNA-seq from Gao et al., "Continuous cell-type
diversification in mouse visual cortex development", Nature (2025).
https://www.nature.com/articles/s41586-025-09644-1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Clustered spatial RNA H5AD with obs['leiden'].")
    parser.add_argument(
        "--reference", type=Path, required=True,
        help="Single-cell reference H5AD with annotations.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="New directory for CSV tables; default: <input-parent>/<input-stem>.rctd.",
    )
    parser.add_argument(
        "--clusters", nargs="+",
        help="Leiden cluster IDs to include, e.g. --clusters 0 1 3; default: all clusters.",
    )
    parser.add_argument("--cell-type-column", default="subclass_label")
    parser.add_argument("--mode", choices=("full", "multi"), default="full")
    parser.add_argument("--max-multi-types", type=int, default=10)
    parser.add_argument("--min-reference-cells", type=int, default=25)
    parser.add_argument("--max-reference-cells", type=int, default=10000)
    parser.add_argument("--min-reference-umi", type=int, default=100)
    parser.add_argument("--min-spot-umi", type=int, default=100)
    parser.add_argument("--max-spot-umi", type=int, default=20000000)
    parser.add_argument("--min-umi-sigma", type=int, default=300)
    parser.add_argument("--min-bulk-counts", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    for name in (
        "min_reference_cells", "max_reference_cells", "min_reference_umi",
        "min_spot_umi", "max_spot_umi", "batch_size", "threads", "max_multi_types",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_reference_cells < args.min_reference_cells:
        parser.error("--max-reference-cells must be >= --min-reference-cells")
    if args.max_spot_umi < args.min_spot_umi:
        parser.error("--max-spot-umi must be >= --min-spot-umi")
    if args.min_umi_sigma < 0 or args.min_bulk_counts < 0:
        parser.error("--min-umi-sigma and --min-bulk-counts must be nonnegative")
    return args


def prepare_counts(
    adata: ad.AnnData, layer: str, gene_column: str,
) -> ad.AnnData:
    """Prepare the count matrix and sum columns with duplicate gene symbols."""
    matrix = adata.X if layer == "X" else adata.layers[layer]
    matrix = sparse.csr_matrix(matrix)
    names = pd.Series(adata.var_names if gene_column == "index" else adata.var[gene_column].to_numpy())
    names = names.astype(str)
    codes, unique_names = pd.factorize(names, sort=False)
    n_duplicates = len(names) - len(unique_names)
    if n_duplicates:
        mapping = sparse.csr_matrix(
            (np.ones(len(names), dtype=matrix.dtype), (np.arange(len(names)), codes)),
            shape=(len(names), len(unique_names)),
        )
        matrix = (matrix @ mapping).tocsr()
    return ad.AnnData(
        X=matrix, obs=adata.obs.copy(),
        var=pd.DataFrame(index=pd.Index(unique_names, name="gene_symbol")),
    )


def prepare_reference(
    adata: ad.AnnData, args: argparse.Namespace,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Filter before downsampling and record every retained/excluded subclass."""
    labels = adata.obs[args.cell_type_column]
    if labels.isna().any() or labels.astype(str).str.strip().eq("").any():
        raise ValueError("Reference contains missing/empty cell type labels")
    labels = labels.astype(str).to_numpy()
    totals = np.asarray(adata.X.sum(axis=1, dtype=np.float64)).ravel()
    rng = np.random.default_rng(42)
    selected_rows = []
    records = []
    for label in sorted(set(labels)):
        rows = np.flatnonzero(labels == label)
        valid = rows[totals[rows] >= args.min_reference_umi]
        status = "retained" if len(valid) >= args.min_reference_cells else "too_few_cells"
        selected = valid if status == "retained" else np.array([], dtype=int)
        if len(selected) > args.max_reference_cells:
            selected = rng.choice(selected, args.max_reference_cells, replace=False)
        selected_rows.extend(selected.tolist())
        records.append({
            "cell_type": label, "n_input": len(rows), "n_umi_pass": len(valid),
            "n_used": len(selected), "status": status,
        })
    summary = pd.DataFrame(records)
    if (summary["status"] == "retained").sum() < 2:
        raise ValueError("Fewer than two reference subclasses remain after filtering")
    filtered = adata[np.sort(selected_rows)].copy()
    filtered.obs[args.cell_type_column] = filtered.obs[args.cell_type_column].astype(str)
    return filtered, summary


def prepare_result_tables(
    spot_metadata: pd.DataFrame, selected_rows: np.ndarray, result: object,
    args: argparse.Namespace,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Expand fitted results to original spots; use selected multi weights in multi mode."""
    metadata = spot_metadata.copy()
    n_spots = len(metadata)
    tables = {}
    names = list(result.cell_type_names)
    pixel_mask = np.asarray(result.pixel_mask, dtype=bool)
    if pixel_mask.shape != (len(selected_rows),):
        raise ValueError("RCTD returned an unexpected pixel mask")
    fitted_rows = selected_rows[pixel_mask]
    full_weights = np.full((n_spots, len(names)), np.nan)
    if args.mode == "multi":
        full_weights[fitted_rows] = 0
        confidence = pd.DataFrame(pd.NA, index=metadata.index, columns=names, dtype="boolean")
        for row, n_types in enumerate(result.n_types):
            indices = result.cell_type_indices[row, :n_types]
            full_weights[fitted_rows[row], indices] = result.sub_weights[row, :n_types]
            confidence.iloc[fitted_rows[row], indices] = result.conf_list[row, :n_types]
        tables["rctd_multi_weights"] = pd.DataFrame(full_weights, index=metadata.index, columns=names)
        tables["rctd_multi_confidence"] = confidence
        return tables, metadata

    raw = np.asarray(result.weights, dtype=np.float64)
    if raw.shape != (len(fitted_rows), len(names)):
        raise ValueError("RCTD returned an unexpected weight matrix shape")
    full_weights[fitted_rows] = raw
    tables["rctd_weights"] = pd.DataFrame(full_weights, index=metadata.index, columns=names)
    clipped = np.maximum(raw, 0)
    sums = clipped.sum(axis=1)
    valid = np.isfinite(raw).all(axis=1) & (sums > 0)
    proportions = np.full_like(full_weights, np.nan)
    valid_rows = fitted_rows[valid]
    proportions[valid_rows] = clipped[valid] / sums[valid, None]
    tables["rctd_proportions"] = pd.DataFrame(proportions, index=metadata.index, columns=names)
    status = np.full(n_spots, "umi_filtered", dtype=object)
    status[selected_rows] = "marker_filtered"
    status[fitted_rows] = "fitted"
    converged = np.zeros(n_spots, dtype=bool)
    converged[fitted_rows] = result.converged
    metadata["rctd_converged"] = converged
    status[fitted_rows[~np.asarray(result.converged, dtype=bool)]] = "not_converged"
    status[fitted_rows[~valid]] = "invalid_weights"
    metadata["rctd_status"] = pd.Categorical(status)
    dominant = np.full(n_spots, "unassigned", dtype=object)
    dominant[valid_rows] = np.asarray(names)[proportions[valid_rows].argmax(axis=1)]
    metadata["rctd_dominant_type"] = pd.Categorical(dominant)
    maxima = np.full(n_spots, np.nan)
    maxima[valid_rows] = proportions[valid_rows].max(axis=1)
    metadata["rctd_max_proportion"] = maxima
    return tables, metadata


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    reference_path = args.reference.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir
        else input_path.parent / f"{input_path.stem}.rctd"
    )

    print(f"Spatial input: {input_path}", flush=True)
    spatial = ad.read_h5ad(input_path)
    if args.clusters:
        labels = spatial.obs["leiden"].astype(str)
        missing = set(args.clusters) - set(labels)
        if missing:
            raise ValueError(f"Unknown Leiden cluster IDs: {', '.join(sorted(missing))}")
        spatial = spatial[labels.isin(args.clusters)].copy()
        print(f"Selected clusters: {', '.join(args.clusters)}; spots: {spatial.n_obs}", flush=True)
    working = prepare_counts(spatial, "counts", "feature_name")
    totals = np.asarray(working.X.sum(axis=1, dtype=np.float64)).ravel()
    spot_metadata = spatial.obs.copy()
    spot_metadata["rctd_n_umi"] = totals
    selected_rows = np.flatnonzero((totals >= args.min_spot_umi) & (totals <= args.max_spot_umi))
    if not len(selected_rows):
        raise ValueError("No spots remain after UMI filtering")

    print(f"Reference input: {reference_path}", flush=True)
    original_reference = ad.read_h5ad(reference_path)
    reference_data = prepare_counts(
        original_reference, "X", "index",
    )
    del original_reference
    reference_data, reference_summary = prepare_reference(reference_data, args)

    common = working.var_names.intersection(reference_data.var_names)
    if len(common) < 10:
        raise ValueError(f"Only {len(common)} common genes; check gene identifier columns")
    shared_counts = working[selected_rows, common].X
    if not np.asarray(shared_counts.sum(axis=1)).ravel().any():
        raise ValueError("UMI-passing spots have no counts in shared genes")
    if not np.any(totals[selected_rows] > args.min_umi_sigma):
        raise ValueError("No spot has UMI > --min-umi-sigma for noise estimation")
    retained = reference_summary[reference_summary["status"] == "retained"]
    dropped = reference_summary[reference_summary["status"] != "retained"]
    print(f"UMI-passing spots: {len(selected_rows)}/{spatial.n_obs}; common genes: {len(common)}", flush=True)
    print(f"Reference: {reference_data.n_obs} cells, {len(retained)} subclasses", flush=True)
    if not dropped.empty:
        print("Excluded reference types:\n" + dropped.to_string(index=False), flush=True)

    import torch
    from rctd import RCTDConfig, Reference, run_rctd

    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    config = RCTDConfig(
        UMI_min=args.min_spot_umi, UMI_max=args.max_spot_umi,
        UMI_min_sigma=args.min_umi_sigma, counts_MIN=args.min_bulk_counts,
        MAX_MULTI_TYPES=args.max_multi_types, device=device, compile=False,
    )
    reference = Reference(
        reference_data, cell_type_col=args.cell_type_column,
        cell_min=args.min_reference_cells, n_max_cells=args.max_reference_cells,
        min_UMI=args.min_reference_umi,
    )
    del reference_data
    print(f"RCTD mode: {args.mode}; device: {device}; batch size: {args.batch_size}", flush=True)
    result = run_rctd(
        working[selected_rows].copy(), reference, mode=args.mode, config=config,
        batch_size=args.batch_size,
    )
    del working, reference
    tables, spot_metadata = prepare_result_tables(spot_metadata, selected_rows, result, args)

    output_dir.mkdir(parents=True, exist_ok=False)
    for key, table in tables.items():
        table.to_csv(output_dir / f"{key}.csv")
    if args.mode == "full":
        spot_metadata.to_csv(output_dir / "spot_metadata.csv")
        reference_summary.to_csv(output_dir / "reference_summary.csv", index=False)

    print(f"Output CSV directory: {output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
