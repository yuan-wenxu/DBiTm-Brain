"""Export cortical spots with micrometer coordinates for SpaLRFactor."""

import argparse
from pathlib import Path
import tempfile

import anndata as ad
import numpy as np
from scipy import sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=["L2/3/4", "L5", "L6"])
    return parser.parse_args()


def prepare_scoring_input(data: ad.AnnData) -> None:
    """Validate raw counts and derive physical coordinates per sample."""
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError("Spot IDs and feature IDs must be unique")
    if not data.obs.columns.is_unique:
        raise ValueError("Observation column names must be unique")
    required = {"sample", "pxl_col_in_fullres", "pxl_row_in_fullres"}
    missing = required - set(data.obs.columns)
    if missing:
        raise ValueError(f"H5AD is missing obs columns: {sorted(missing)}")
    if data.obs["sample"].isna().any():
        raise ValueError("Sample labels must not contain missing values")
    sample_names = data.obs["sample"].astype(str)
    if sample_names.nunique() != data.obs["sample"].nunique():
        raise ValueError("Sample labels collide when converted to strings")
    if "counts" not in data.layers:
        raise ValueError("H5AD is missing layers['counts']")
    counts = sparse.csr_matrix(data.layers["counts"], dtype=float, copy=True)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    if (not np.isfinite(counts.data).all() or (counts.data < 0).any()
            or not np.allclose(counts.data, np.rint(counts.data),
                               rtol=0, atol=1e-6)):
        raise ValueError("Counts must be finite nonnegative integers")
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("Cortical spots must have positive count libraries")

    coordinates = data.obs[[
        "pxl_col_in_fullres", "pxl_row_in_fullres",
    ]].to_numpy(dtype=float, copy=True)
    spatial = data.uns.get("spatial", {})
    for sample in sample_names.unique():
        metadata = spatial.get(sample, {}).get("metadata", {})
        pixel_size = metadata.get("source_pixel_size_um")
        if (pixel_size is None or not np.isfinite(pixel_size)
                or pixel_size <= 0):
            raise ValueError(
                f"{sample}: metadata['source_pixel_size_um'] "
                "must be finite and positive"
            )
        mask = sample_names.eq(sample).to_numpy()
        coordinates[mask] *= float(pixel_size)
        if not np.isfinite(coordinates[mask]).all():
            raise ValueError(f"{sample}: spatial coordinates must be finite")
        if len(np.unique(coordinates[mask], axis=0)) != int(mask.sum()):
            raise ValueError(f"{sample}: duplicate spatial spot centers")
    data.obs["x_um"] = coordinates[:, 0]
    data.obs["y_um"] = coordinates[:, 1]
    data.obsm["spatial"] = coordinates


def main() -> None:
    args = parse_args()
    output = args.output_dir / f"{args.h5ad.stem}.cortex.h5ad"
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    data = ad.read_h5ad(args.h5ad, backed="r")
    try:
        if "annotation" not in data.obs:
            raise ValueError("H5AD is missing obs['annotation']")
        mask = data.obs["annotation"].isin(args.labels).to_numpy()
        if not mask.any():
            raise ValueError(f"No cortical spots found for labels {args.labels}")
        total_spots = data.n_obs
        cortex = data[mask, :].to_memory()
    finally:
        data.file.close()

    prepare_scoring_input(cortex)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp",
            dir=args.output_dir, delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        cortex.write_h5ad(temporary_path, compression="gzip")
        temporary_path.replace(output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"Cortical spots: {cortex.n_obs} / {total_spots}")
    print(f"Genes retained: {cortex.n_vars}")
    if "sample" in cortex.obs:
        print(cortex.obs["sample"].value_counts().to_string())
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
