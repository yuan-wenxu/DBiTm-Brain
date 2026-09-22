#!/usr/bin/env python3
"""Align three modality matrices by matched spots and write one MuData file."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import anndata as ad
import matplotlib
import mudata as md
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


md.set_options(pull_on_update=False)

MODALITY_COLUMNS = {
    "mrna": "mrna_barcode",
    "taps": "taps_barcode",
    "taps_beta": "taps_beta_barcode",
}
REFERENCE_COLUMN = MODALITY_COLUMNS["taps_beta"]
REFERENCE_ID = "taps_beta"
SPATIAL_COLUMNS = (
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
)
CLUSTER_KEYS = {
    "mrna": "leiden",
    "taps": "cluster",
    "taps_beta": "cluster",
}
MODALITY_LABELS = {
    "mrna": "mRNA",
    "taps": "TAPS",
    "taps_beta": "TAPS-beta",
}
FIGURE_DPI = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrna", type=Path, required=True, help="mRNA H5AD.")
    parser.add_argument("--taps", type=Path, required=True, help="TAPS H5AD.")
    parser.add_argument(
        "--taps-beta",
        dest="taps_beta",
        type=Path,
        required=True,
        help="TAPS-beta H5AD used as the reference modality.",
    )
    parser.add_argument(
        "--spot-overlap",
        type=Path,
        required=True,
        help="TSV containing taps_beta_barcode, mrna_barcode, and taps_barcode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output H5MU path; the file must not already exist.",
    )
    parser.add_argument(
        "--spot-size-um",
        type=float,
        default=50.0,
        help="Side length of spatial spot squares in micrometers (default: 50).",
    )
    parser.add_argument(
        "--crop-margin-um",
        type=float,
        default=500.0,
        help="Margin around cropped spatial plots in micrometers (default: 500).",
    )
    args = parser.parse_args()
    if args.spot_size_um <= 0:
        parser.error("--spot-size-um must be greater than zero")
    if args.crop_margin_um < 0:
        parser.error("--crop-margin-um must be non-negative")
    return args


def read_overlap_table(path: Path) -> pd.DataFrame:
    print(f"Reading spot correspondence: {path}", flush=True)
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing_columns = set(MODALITY_COLUMNS.values()).difference(table.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Spot correspondence table is missing columns: {missing}")
    if table.empty:
        raise ValueError("Spot correspondence table contains no spots")

    spot_columns = list(MODALITY_COLUMNS.values())
    blank_columns = [column for column in spot_columns if table[column].eq("").any()]
    if blank_columns:
        raise ValueError(
            "Spot correspondence table contains blank values in: "
            + ", ".join(blank_columns)
        )
    duplicate_columns = [
        column for column in spot_columns if table[column].duplicated().any()
    ]
    if duplicate_columns:
        raise ValueError(
            "Spot correspondence must be one-to-one; duplicate values found in: "
            + ", ".join(duplicate_columns)
        )
    return table


def read_modality(path: Path, modality_id: str) -> ad.AnnData:
    print(f"Reading {modality_id}: {path}", flush=True)
    adata = ad.read_h5ad(path)
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(f"The {modality_id} matrix is empty: {path}")
    if not adata.obs_names.is_unique:
        raise ValueError(f"The {modality_id} matrix has duplicate spot IDs: {path}")
    print(f"  {adata.n_obs:,} spots x {adata.n_vars:,} features", flush=True)
    return adata


def retain_available_spots(
    table: pd.DataFrame,
    adatas: dict[str, ad.AnnData],
) -> pd.DataFrame:
    available = pd.Series(True, index=table.index)
    for modality_id, barcode_column in MODALITY_COLUMNS.items():
        present = table[barcode_column].isin(adatas[modality_id].obs_names)
        missing_count = int((~present).sum())
        print(
            f"  {modality_id}: {int(present.sum()):,} matched spots present; "
            f"{missing_count:,} absent from H5AD",
            flush=True,
        )
        available &= present

    retained = table.loc[available].reset_index(drop=True)
    if retained.empty:
        raise ValueError("No corresponding spots are present in all three matrices")
    print(
        f"Retaining {len(retained):,} of {len(table):,} correspondence rows present "
        "in all modalities",
        flush=True,
    )
    return retained


def align_modalities(
    adatas: dict[str, ad.AnnData],
    table: pd.DataFrame,
) -> dict[str, ad.AnnData]:
    reference_spots = pd.Index(table[REFERENCE_COLUMN], name="spot_id")
    aligned: dict[str, ad.AnnData] = {}
    for modality_id, barcode_column in MODALITY_COLUMNS.items():
        source_spots = table[barcode_column].to_numpy()
        subset = adatas[modality_id][source_spots].copy()
        subset.obs.insert(0, "source_barcode", source_spots)
        subset.obs_names = reference_spots.copy()
        aligned[modality_id] = subset

    reference_obs = aligned[REFERENCE_ID].obs
    missing_spatial_columns = set(SPATIAL_COLUMNS).difference(reference_obs.columns)
    if missing_spatial_columns:
        missing = ", ".join(sorted(missing_spatial_columns))
        raise ValueError(f"TAPS-beta is missing spatial obs columns: {missing}")
    for modality_id in ("mrna", "taps"):
        modality = aligned[modality_id]
        for column in SPATIAL_COLUMNS:
            if column in modality.obs:
                modality.obs[f"source_{column}"] = modality.obs[column].to_numpy()
            modality.obs[column] = reference_obs[column].to_numpy()
        modality.uns.pop("spatial", None)
    return aligned


def reference_spatial_data(
    adata: ad.AnnData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    spatial = adata.uns.get("spatial", {})
    if len(spatial) != 1:
        raise ValueError(
            "Expected exactly one TAPS-beta library under uns['spatial']"
        )
    library = next(iter(spatial.values()))
    image = np.asarray(library["images"]["hires"])
    scale = float(library["scalefactors"]["tissue_hires_scalef"])
    pixel_size_um = float(library["metadata"]["hires_pixel_size_um"])
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(
            "TAPS-beta hires_pixel_size_um must be finite and greater than zero"
        )
    x = adata.obs["pxl_col_in_fullres"].to_numpy(dtype=float) * scale
    y = adata.obs["pxl_row_in_fullres"].to_numpy(dtype=float) * scale
    return image, x, y, pixel_size_um


def cluster_plot_data(
    adata: ad.AnnData,
    cluster_key: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    if cluster_key not in adata.obs:
        raise ValueError(f"Input matrix is missing obs[{cluster_key!r}]")
    color_key = f"{cluster_key}_colors"
    if color_key not in adata.uns:
        raise ValueError(f"Input matrix is missing uns[{color_key!r}]")

    values = adata.obs[cluster_key]
    if not isinstance(values.dtype, pd.CategoricalDtype):
        values = values.astype("category")
    clusters = values.cat.codes.to_numpy(dtype=int)
    if np.any(clusters < 0):
        raise ValueError(f"obs[{cluster_key!r}] contains missing cluster labels")
    labels = [str(value) for value in values.cat.categories]
    colors = [str(value) for value in np.asarray(adata.uns[color_key]).tolist()]
    if len(colors) != len(labels):
        raise ValueError(
            f"Expected one {color_key!r} entry per category; found "
            f"{len(colors)} colors for {len(labels)} clusters"
        )
    return clusters, labels, colors


def add_cluster_squares(
    axis: plt.Axes,
    clusters: np.ndarray,
    colors: list[str],
    x: np.ndarray,
    y: np.ndarray,
    side_length: float,
) -> None:
    half_size = side_length / 2
    squares = [
        Rectangle(
            (x_value - half_size, y_value - half_size),
            side_length,
            side_length,
        )
        for x_value, y_value in zip(x, y, strict=True)
    ]
    axis.add_collection(
        PatchCollection(
            squares,
            facecolors=[colors[cluster] for cluster in clusters],
            edgecolors="none",
        )
    )


def add_scale_bar(
    axis: plt.Axes,
    view_width_pixels: float,
    view_height_pixels: float,
    pixel_size_um: float,
) -> None:
    target_length_um = view_width_pixels * pixel_size_um / 5
    magnitude = 10 ** np.floor(np.log10(target_length_um))
    scale_bar_um = max(
        multiplier * magnitude
        for multiplier in (1, 2, 5)
        if multiplier * magnitude <= target_length_um
    )
    scale_bar = AnchoredSizeBar(
        axis.transData,
        scale_bar_um / pixel_size_um,
        f"{scale_bar_um:g} µm",
        loc="lower left",
        pad=0.4,
        borderpad=0.5,
        sep=3,
        frameon=True,
        size_vertical=max(2, view_height_pixels * 0.003),
        color="white",
    )
    scale_bar.patch.set_facecolor("black")
    scale_bar.patch.set_alpha(0.6)
    scale_bar.patch.set_edgecolor("none")
    axis.add_artist(scale_bar)


def plot_spatial_clusters(
    path: Path,
    modality_label: str,
    clusters: np.ndarray,
    cluster_labels: list[str],
    colors: list[str],
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    pixel_size_um: float,
    spot_size_um: float,
    crop_margin_um: float | None,
) -> None:
    spot_size_pixels = spot_size_um / pixel_size_um
    half_size = spot_size_pixels / 2
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(image, cmap="gray", origin="upper")
    add_cluster_squares(axis, clusters, colors, x, y, spot_size_pixels)

    if crop_margin_um is None:
        left, right = 0.0, float(image.shape[1])
        top, bottom = 0.0, float(image.shape[0])
        title = f"{modality_label} clusters on TAPS-beta tissue image"
    else:
        margin_pixels = crop_margin_um / pixel_size_um
        left = max(0.0, float(x.min()) - half_size - margin_pixels)
        right = min(float(image.shape[1]), float(x.max()) + half_size + margin_pixels)
        top = max(0.0, float(y.min()) - half_size - margin_pixels)
        bottom = min(float(image.shape[0]), float(y.max()) + half_size + margin_pixels)
        title = f"{modality_label} clusters on cropped TAPS-beta tissue image"

    axis.set_xlim(left, right)
    axis.set_ylim(bottom, top)
    axis.set_aspect("equal")
    axis.set_title(title, fontsize=12)
    add_scale_bar(axis, right - left, bottom - top, pixel_size_um)
    axis.legend(
        handles=[
            Patch(facecolor=color, edgecolor="none", label=label)
            for label, color in zip(cluster_labels, colors, strict=True)
        ],
        title="Cluster",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def cluster_plot_paths(output_path: Path) -> list[Path]:
    return [
        output_path.parent
        / f"{modality_id.replace('_', '-')}-clusters{suffix}.png"
        for modality_id in MODALITY_COLUMNS
        for suffix in ("", "-cropped")
    ]


def write_cluster_plots(
    output_path: Path,
    adatas: dict[str, ad.AnnData],
    spot_size_um: float,
    crop_margin_um: float,
) -> None:
    image, x, y, pixel_size_um = reference_spatial_data(adatas[REFERENCE_ID])
    for modality_id, adata in adatas.items():
        clusters, cluster_labels, colors = cluster_plot_data(
            adata,
            CLUSTER_KEYS[modality_id],
        )
        file_id = modality_id.replace("_", "-")
        for suffix, margin in (
            ("", None),
            ("-cropped", crop_margin_um),
        ):
            plot_spatial_clusters(
                output_path.parent
                / f"{output_path.stem}-{file_id}-clusters{suffix}.png",
                MODALITY_LABELS[modality_id],
                clusters,
                cluster_labels,
                colors,
                image,
                x,
                y,
                pixel_size_um,
                spot_size_um,
                margin,
            )


def write_mudata(mdata: md.MuData, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        print(f"Writing {output_path} ...", flush=True)
        mdata.write_h5mu(temporary_path, compression="gzip")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    args = parse_args()
    paths = {
        "mrna": args.mrna.expanduser().resolve(),
        "taps": args.taps.expanduser().resolve(),
        "taps_beta": args.taps_beta.expanduser().resolve(),
    }
    overlap_path = args.spot_overlap.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise SystemExit(f"Error: output file already exists: {output_path}")
    existing_plots = [path for path in cluster_plot_paths(output_path) if path.exists()]
    if existing_plots:
        raise SystemExit(f"Error: cluster plot already exists: {existing_plots[0]}")

    try:
        table = read_overlap_table(overlap_path)
        adatas = {
            modality_id: read_modality(path, modality_id)
            for modality_id, path in paths.items()
        }
        retained = retain_available_spots(table, adatas)
        aligned = align_modalities(adatas, retained)
        mdata = md.MuData(aligned)
        mdata.uns["spot_correspondence"] = {
            "source": str(overlap_path),
            "reference_modality": "taps_beta",
            "input_rows": len(table),
            "retained_rows": len(retained),
        }
        write_mudata(mdata, output_path)
        write_cluster_plots(
            output_path,
            aligned,
            args.spot_size_um,
            args.crop_margin_um,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(
        f"Wrote {len(retained):,} aligned spots across three modalities to "
        f"{output_path}"
    )
    print(f"Wrote cluster plots next to {output_path}")


if __name__ == "__main__":
    main()
