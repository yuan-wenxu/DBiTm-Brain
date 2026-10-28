#!/usr/bin/env python3
"""Find marker genes and VMRs independently for each WNN modality."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats

md.set_options(pull_on_update=False)

CLUSTER_KEY = "wnn_leiden"
METHYLATION_LAYER = "methylation"
VALUE_MASK_LAYER = "value_mask"
MARKER_COLUMNS = (
    "cluster",
    "modality",
    "marker_type",
    "rank",
    "feature_id",
    "feature_name",
    "direction",
    "score",
    "effect_size",
    "effect_metric",
    "mean_in_cluster",
    "mean_out_cluster",
    "fraction_in_cluster",
    "fraction_out_cluster",
    "fraction_metric",
    "n_in_cluster",
    "n_out_cluster",
    "p_value",
    "p_value_adj",
    "gene_id",
    "gene_name",
    "gene_type",
    "gene_relation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="WNN-clustered H5MU from 05-joint-cluster-multimodal.py.",
    )
    parser.add_argument(
        "--top-markers-per-cluster",
        type=int,
        default=50,
        help=(
            "Maximum significant markers retained per cluster and modality "
            "(default: 50)."
        ),
    )
    parser.add_argument(
        "--marker-fdr",
        type=float,
        default=0.05,
        help="Maximum adjusted P value for marker results (default: 0.05).",
    )
    parser.add_argument(
        "--min-vmr-observations",
        type=int,
        default=3,
        help=(
            "Minimum observed methylation values required inside and outside a cluster "
            "for marker-VMR testing (default: 3)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing marker table.",
    )
    args = parser.parse_args()
    if args.top_markers_per_cluster < 1:
        parser.error("--top-markers-per-cluster must be at least 1")
    if not 0 < args.marker_fdr <= 1:
        parser.error("--marker-fdr must be greater than 0 and at most 1")
    if args.min_vmr_observations < 2:
        parser.error("--min-vmr-observations must be at least 2")
    return args


def read_clustered_mudata(path: Path) -> md.MuData:
    print(f"Reading {path} ...", flush=True)
    mdata = md.read_h5mu(path)
    missing = {"mrna", "taps", "taps_beta"}.difference(mdata.mod)
    if missing:
        raise ValueError("Missing modalities: " + ", ".join(sorted(missing)))

    mrna = mdata.mod["mrna"]
    for modality_id in ("taps", "taps_beta"):
        if not mrna.obs_names.equals(mdata.mod[modality_id].obs_names):
            raise ValueError(f"mRNA and {modality_id} spot orders are not aligned")
    for modality_id in ("mrna", "taps", "taps_beta"):
        adata = mdata.mod[modality_id]
        if CLUSTER_KEY not in adata.obs:
            raise ValueError(
                f"{modality_id} is missing obs[{CLUSTER_KEY!r}]; run stage 05 first"
            )
        if adata.obs[CLUSTER_KEY].isna().any():
            raise ValueError(f"{modality_id} obs[{CLUSTER_KEY!r}] contains missing values")
        if not isinstance(adata.obs[CLUSTER_KEY].dtype, pd.CategoricalDtype):
            raise ValueError(
                f"{modality_id} obs[{CLUSTER_KEY!r}] must be categorical"
            )

    mrna_labels = mrna.obs[CLUSTER_KEY].astype(str)
    for modality_id in ("taps", "taps_beta"):
        if not mrna_labels.equals(mdata.mod[modality_id].obs[CLUSTER_KEY].astype(str)):
            raise ValueError(f"mRNA and {modality_id} WNN clusters do not match")
    categories = mrna.obs[CLUSTER_KEY].cat.categories
    if len(categories) < 2:
        raise ValueError("At least two WNN clusters are required for marker testing")
    print(
        f"Using {mrna.n_obs:,} aligned spots in {len(categories)} WNN clusters",
        flush=True,
    )
    return mdata


def matrix_feature_means(
    matrix: sparse.spmatrix | np.ndarray,
    spot_mask: np.ndarray,
    feature_indices: np.ndarray,
) -> np.ndarray:
    selected = matrix[spot_mask][:, feature_indices]
    return np.asarray(selected.mean(axis=0)).ravel()


def gene_marker_table(
    adata: ad.AnnData,
    top_markers: int,
    marker_fdr: float,
) -> pd.DataFrame:
    marker_key = "_wnn_marker_genes"
    adata.uns.pop(marker_key, None)
    records: list[pd.DataFrame] = []
    try:
        sc.tl.rank_genes_groups(
            adata,
            groupby=CLUSTER_KEY,
            method="wilcoxon",
            corr_method="benjamini-hochberg",
            use_raw=False,
            n_genes=adata.n_vars,
            pts=True,
            tie_correct=True,
            key_added=marker_key,
        )
        result = adata.uns[marker_key]
        clusters = adata.obs[CLUSTER_KEY]
        for cluster in clusters.cat.categories:
            cluster_name = str(cluster)
            frame = pd.DataFrame(
                {
                    "feature_id": np.asarray(
                        result["names"][cluster_name], dtype=str
                    ),
                    "score": np.asarray(
                        result["scores"][cluster_name], dtype=float
                    ),
                    "effect_size": np.asarray(
                        result["logfoldchanges"][cluster_name], dtype=float
                    ),
                    "p_value": np.asarray(
                        result["pvals"][cluster_name], dtype=float
                    ),
                    "p_value_adj": np.asarray(
                        result["pvals_adj"][cluster_name], dtype=float
                    ),
                }
            )
            frame = frame.loc[
                np.isfinite(frame["effect_size"])
                & np.isfinite(frame["p_value_adj"])
                & frame["effect_size"].gt(0)
                & frame["p_value_adj"].le(marker_fdr)
            ].head(top_markers).copy()
            if frame.empty:
                continue

            feature_ids = frame["feature_id"].to_numpy()
            feature_indices = adata.var_names.get_indexer(feature_ids)
            if np.any(feature_indices < 0):
                raise ValueError("Ranked marker gene is absent from mRNA var_names")
            in_cluster = clusters.astype(str).eq(cluster_name).to_numpy()
            frame["cluster"] = cluster_name
            frame["modality"] = "mrna"
            frame["marker_type"] = "gene"
            frame["rank"] = np.arange(1, len(frame) + 1)
            if "feature_name" in adata.var:
                frame["feature_name"] = adata.var.iloc[feature_indices][
                    "feature_name"
                ].astype(str).to_numpy()
            else:
                frame["feature_name"] = feature_ids
            frame["direction"] = "upregulated"
            frame["effect_metric"] = "log2_fold_change"
            frame["mean_in_cluster"] = matrix_feature_means(
                adata.X, in_cluster, feature_indices
            )
            frame["mean_out_cluster"] = matrix_feature_means(
                adata.X, ~in_cluster, feature_indices
            )
            frame["fraction_in_cluster"] = result["pts"].loc[
                feature_ids, cluster_name
            ].to_numpy(dtype=float)
            frame["fraction_out_cluster"] = result["pts_rest"].loc[
                feature_ids, cluster_name
            ].to_numpy(dtype=float)
            frame["fraction_metric"] = "detected"
            frame["n_in_cluster"] = int(in_cluster.sum())
            frame["n_out_cluster"] = int((~in_cluster).sum())
            frame["gene_id"] = frame["feature_id"]
            frame["gene_name"] = frame["feature_name"]
            frame["gene_type"] = pd.NA
            frame["gene_relation"] = pd.NA
            records.append(frame)
    finally:
        adata.uns.pop(marker_key, None)

    if not records:
        return pd.DataFrame(columns=MARKER_COLUMNS)
    return pd.concat(records, ignore_index=True).loc[:, MARKER_COLUMNS]


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    finite = np.isfinite(p_values)
    if not finite.any():
        return adjusted
    values = p_values[finite]
    order = np.argsort(values)
    ranked = values[order]
    scaled = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    corrected = np.minimum.accumulate(scaled[::-1])[::-1]
    restored = np.empty_like(corrected)
    restored[order] = np.clip(corrected, 0, 1)
    adjusted[finite] = restored
    return adjusted


def validated_methylation_data(
    adata: ad.AnnData,
    modality_id: str,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    missing = {METHYLATION_LAYER, VALUE_MASK_LAYER}.difference(adata.layers)
    if missing:
        raise ValueError(
            f"{modality_id} is missing marker-VMR layer(s): "
            + ", ".join(sorted(missing))
        )
    methylation = adata.layers[METHYLATION_LAYER]
    value_mask = adata.layers[VALUE_MASK_LAYER]
    if not sparse.issparse(methylation) or not sparse.issparse(value_mask):
        raise ValueError(f"{modality_id} methylation and value-mask layers must be sparse")
    methylation = methylation.tocsr(copy=False)
    value_mask = value_mask.tocsr(copy=False)
    if methylation.shape != value_mask.shape:
        raise ValueError(f"{modality_id} methylation and value-mask shapes do not match")
    if methylation.data.size and (
        not np.isfinite(methylation.data).all()
        or methylation.data.min() < 0
        or methylation.data.max() > 1
    ):
        raise ValueError(f"{modality_id} methylation values must be finite fractions from 0 to 1")
    if value_mask.data.size and not np.isin(value_mask.data, (0, 1)).all():
        raise ValueError(f"{modality_id} value-mask layer must contain only 0 and 1")
    return methylation, value_mask


def welch_statistics(
    in_sum: np.ndarray,
    in_squared_sum: np.ndarray,
    in_count: np.ndarray,
    out_sum: np.ndarray,
    out_squared_sum: np.ndarray,
    out_count: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean_in = np.full(in_sum.shape, np.nan, dtype=float)
    mean_out = np.full(in_sum.shape, np.nan, dtype=float)
    variance_in = np.full(in_sum.shape, np.nan, dtype=float)
    variance_out = np.full(in_sum.shape, np.nan, dtype=float)
    mean_in[valid] = in_sum[valid] / in_count[valid]
    mean_out[valid] = out_sum[valid] / out_count[valid]
    variance_in[valid] = (
        in_squared_sum[valid] - np.square(in_sum[valid]) / in_count[valid]
    ) / (in_count[valid] - 1)
    variance_out[valid] = (
        out_squared_sum[valid] - np.square(out_sum[valid]) / out_count[valid]
    ) / (out_count[valid] - 1)
    variance_in[valid] = np.maximum(variance_in[valid], 0)
    variance_out[valid] = np.maximum(variance_out[valid], 0)

    effect = mean_in - mean_out
    with np.errstate(divide="ignore", invalid="ignore"):
        first_term = variance_in / in_count
        second_term = variance_out / out_count
        standard_error_squared = first_term + second_term
        degrees_denominator = (
            np.square(first_term) / (in_count - 1)
            + np.square(second_term) / (out_count - 1)
        )
    score = np.full(in_sum.shape, np.nan, dtype=float)
    p_value = np.full(in_sum.shape, np.nan, dtype=float)
    nonzero_error = valid & (standard_error_squared > 0)
    score[nonzero_error] = effect[nonzero_error] / np.sqrt(
        standard_error_squared[nonzero_error]
    )
    has_degrees = nonzero_error & (degrees_denominator > 0)
    degrees_freedom = np.square(standard_error_squared[has_degrees]) / (
        degrees_denominator[has_degrees]
    )
    p_value[has_degrees] = 2 * stats.t.sf(
        np.abs(score[has_degrees]), degrees_freedom
    )

    zero_error = valid & (standard_error_squared == 0)
    equal_means = zero_error & np.isclose(effect, 0)
    score[equal_means] = 0
    p_value[equal_means] = 1
    different_means = zero_error & ~np.isclose(effect, 0)
    score[different_means] = np.sign(effect[different_means]) * np.inf
    p_value[different_means] = 0
    return mean_in, mean_out, effect, score, p_value


def vmr_marker_table(
    adata: ad.AnnData,
    modality_id: str,
    top_markers: int,
    marker_fdr: float,
    min_observations: int,
) -> pd.DataFrame:
    methylation, value_mask = validated_methylation_data(adata, modality_id)
    squared_methylation = methylation.copy()
    squared_methylation.data = np.square(squared_methylation.data)
    clusters = adata.obs[CLUSTER_KEY]
    total_count = np.asarray(value_mask.sum(axis=0)).ravel().astype(float)
    total_sum = np.asarray(methylation.sum(axis=0)).ravel()
    total_squared_sum = np.asarray(squared_methylation.sum(axis=0)).ravel()
    records: list[pd.DataFrame] = []

    for cluster in clusters.cat.categories:
        cluster_name = str(cluster)
        in_cluster = clusters.astype(str).eq(cluster_name).to_numpy()
        in_count = (
            np.asarray(value_mask[in_cluster].sum(axis=0)).ravel().astype(float)
        )
        in_sum = np.asarray(methylation[in_cluster].sum(axis=0)).ravel()
        in_squared_sum = np.asarray(
            squared_methylation[in_cluster].sum(axis=0)
        ).ravel()
        out_count = total_count - in_count
        out_sum = total_sum - in_sum
        out_squared_sum = total_squared_sum - in_squared_sum
        valid = (in_count >= min_observations) & (out_count >= min_observations)
        mean_in, mean_out, effect, score, p_value = welch_statistics(
            in_sum,
            in_squared_sum,
            in_count,
            out_sum,
            out_squared_sum,
            out_count,
            valid,
        )
        p_value_adj = benjamini_hochberg(p_value)
        selected = np.flatnonzero(
            valid
            & np.isfinite(effect)
            & np.isfinite(p_value_adj)
            & (p_value_adj <= marker_fdr)
        )
        if selected.size == 0:
            continue
        order = np.argsort(-np.abs(score[selected]), kind="stable")
        selected = selected[order[:top_markers]]
        frame = pd.DataFrame(
            {
                "cluster": cluster_name,
                "modality": modality_id,
                "marker_type": "vmr",
                "rank": np.arange(1, len(selected) + 1),
                "feature_id": adata.var_names[selected].astype(str),
                "direction": np.where(
                    effect[selected] > 0,
                    "hypermethylated",
                    "hypomethylated",
                ),
                "score": score[selected],
                "effect_size": effect[selected],
                "effect_metric": "delta_methylation",
                "mean_in_cluster": mean_in[selected],
                "mean_out_cluster": mean_out[selected],
                "fraction_in_cluster": in_count[selected]
                / int(in_cluster.sum()),
                "fraction_out_cluster": out_count[selected]
                / int((~in_cluster).sum()),
                "fraction_metric": "observed",
                "n_in_cluster": in_count[selected].astype(int),
                "n_out_cluster": out_count[selected].astype(int),
                "p_value": p_value[selected],
                "p_value_adj": p_value_adj[selected],
            }
        )
        annotation = adata.var.iloc[selected]
        frame["feature_name"] = (
            annotation["feature_name"].astype(str).to_numpy()
            if "feature_name" in annotation
            else frame["feature_id"]
        )
        for column in ("gene_id", "gene_name", "gene_type", "gene_relation"):
            frame[column] = (
                annotation[column].to_numpy()
                if column in annotation
                else pd.NA
            )
        records.append(frame)

    if not records:
        return pd.DataFrame(columns=MARKER_COLUMNS)
    return pd.concat(records, ignore_index=True).loc[:, MARKER_COLUMNS]


def find_cluster_markers(
    mdata: md.MuData,
    args: argparse.Namespace,
) -> pd.DataFrame:
    genes = gene_marker_table(
        mdata.mod["mrna"], args.top_markers_per_cluster, args.marker_fdr
    )
    vmrs = vmr_marker_table(
        mdata.mod["taps"],
        "taps",
        args.top_markers_per_cluster,
        args.marker_fdr,
        args.min_vmr_observations,
    )
    beta_vmrs = vmr_marker_table(
        mdata.mod["taps_beta"],
        "taps_beta",
        args.top_markers_per_cluster,
        args.marker_fdr,
        args.min_vmr_observations,
    )
    markers = pd.concat((genes, vmrs, beta_vmrs), ignore_index=True).loc[:, MARKER_COLUMNS]
    if markers.empty:
        print(
            f"Warning: no markers passed adjusted P <= {args.marker_fdr:g}",
            flush=True,
        )
    else:
        counts = markers.groupby(["cluster", "modality"], observed=True).size()
        print(
            "Markers retained: "
            + ", ".join(
                f"{cluster} {modality}={count}"
                for (cluster, modality), count in counts.items()
            ),
            flush=True,
        )
    return markers


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = input_path.parent / "wnn-markers-by-modality.tsv"
    try:
        if output_path.exists() and not args.overwrite:
            raise ValueError(f"Marker output already exists: {output_path}")
        mdata = read_clustered_mudata(input_path)
        markers = find_cluster_markers(mdata, args)
        markers.to_csv(output_path, sep="\t", index=False)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(f"Marker table: {output_path}")


if __name__ == "__main__":
    main()
