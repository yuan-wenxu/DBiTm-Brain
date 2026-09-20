import gzip
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
import tempfile
import sys
import anndata as ad


def read_nonempty_lines(path: Path) -> list[str]:
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        values = [line.rstrip("\r\n") for line in handle]
    if not values or any(not value for value in values):
        raise ValueError(f"Expected non-empty lines in {path}")
    return values


def read_features(path: Path) -> pd.DataFrame:
    features = pd.read_csv(
        path,
        sep="\t",
        header=None,
        dtype=str,
        compression="gzip",
        keep_default_na=False,
    )
    if features.empty:
        raise ValueError(f"No features found in {path}")
    if features.shape[1] != 3:
        raise ValueError(
            f"Expected exactly 3 feature columns in {path}, found {features.shape[1]}"
        )

    features.columns = ["feature_id", "feature_name", "feature_type"]
    if (features["feature_id"] == "").any():
        raise ValueError(f"Empty feature identifier in {path}")
    if features["feature_id"].duplicated().any():
        duplicate = features.loc[
            features["feature_id"].duplicated(), "feature_id"
        ].iloc[0]
        raise ValueError(f"Duplicate feature identifier in {path}: {duplicate}")

    features.index = pd.Index(features.pop("feature_id"), name="feature_id")
    return features


POSITION_COLUMNS = (
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
)


def read_positions(path: Path) -> pd.DataFrame:
    positions = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        dtype=str,
        keep_default_na=False,
    )
    missing = [column for column in POSITION_COLUMNS if column not in positions.columns]
    if missing:
        raise ValueError(
            f"Missing position column(s) in {path}: {', '.join(missing)}"
        )

    integer_columns = (
        "in_tissue",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    )
    for column in integer_columns:
        try:
            positions[column] = pd.to_numeric(
                positions[column], errors="raise", downcast="integer"
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Position column {column!r} is not integer-valued") from error

    for column in ("array_row", "array_col"):
        try:
            pd.to_numeric(positions[column], errors="raise", downcast="integer")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Position column {column!r} is not integer-valued") from error

    if positions["barcode"].duplicated().any():
        raise ValueError(f"Duplicate sequence barcode in {path}")
    if positions.duplicated(["array_row", "array_col"]).any():
        raise ValueError(f"Duplicate array_row/array_col pair in {path}")
    return positions


def read_image(path: Path, imgae_scale_factor: float) -> np.ndarray:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError(f"Expected an 8-bit grayscale image, found mode {image.mode}")
        target_size = tuple(
            max(1, round(length * imgae_scale_factor)) for length in image.size
        )
        resized = image.resize(target_size, resample=Image.Resampling.LANCZOS)
        return np.asarray(resized)


def validate_output(output: Path) -> None:
    if output.exists():
        raise ValueError(f"Output file already exists: {output}.")


def write_anndata(adata: ad.AnnData, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        print(f"Writing {output} ...", file=sys.stderr, flush=True)
        adata.write_h5ad(temporary_path)
        temporary_path.replace(output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def in_tissue_mask(adata: ad.AnnData) -> np.ndarray:
    """Return a validated mask for spots marked as inside tissue."""
    if "in_tissue" not in adata.obs:
        raise ValueError("The input H5AD is missing the obs column 'in_tissue'")

    try:
        values = adata.obs["in_tissue"].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("obs['in_tissue'] must contain only 0 or 1") from error
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError("obs['in_tissue'] must contain only 0 or 1")
    return values == 1


def spatial_plot_data(
    adata: ad.AnnData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    required = {"pxl_row_in_fullres", "pxl_col_in_fullres"}
    missing = required - set(adata.obs.columns)
    if missing:
        raise ValueError(f"Missing spatial obs column(s): {', '.join(sorted(missing))}")

    spatial = adata.uns.get("spatial", {})
    if len(spatial) != 1:
        raise ValueError("Expected exactly one library under adata.uns['spatial']")
    library = next(iter(spatial.values()))
    image = np.asarray(library["images"]["hires"])
    scale = float(library["scalefactors"]["tissue_hires_scalef"])
    hires_pixel_size_um = float(library["metadata"]["hires_pixel_size_um"])
    if not np.isfinite(hires_pixel_size_um) or hires_pixel_size_um <= 0:
        raise ValueError("hires_pixel_size_um must be finite and greater than zero")
    x = adata.obs["pxl_col_in_fullres"].to_numpy(dtype=float) * scale
    y = adata.obs["pxl_row_in_fullres"].to_numpy(dtype=float) * scale
    return image, x, y, hires_pixel_size_um
