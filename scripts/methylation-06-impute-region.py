#!/usr/bin/env python3
"""Estimate regional methylation by masked MAGIC diffusion and plot both maps."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import anndata as ad
import magic
import numpy as np
import pandas as pd
from covtools import region_methylation

from utils import spatial_plot_data, validate_output


SpatialData = tuple[np.ndarray, np.ndarray, np.ndarray, float]
SPOT_SIZE_UM = 50.0
CROP_MARGIN_UM = 500.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True,
        help=(
            "Clustered H5AD with X_pca and spatial data from the same "
            "sample and modality."
        ),
    )
    parser.add_argument(
        "--cov", type=Path, required=True,
        help="Coordinate-sorted, uncompressed COV with barcodes in column 7.",
    )
    parser.add_argument(
        "--index", type=Path,
        help="Existing CovTools index (default: <cov>.coi).",
    )
    parser.add_argument("--chrom", required=True, help="Chromosome, e.g. chr1.")
    parser.add_argument(
        "--start", type=int, required=True, help="1-based inclusive start.",
    )
    parser.add_argument(
        "--end", type=int, required=True, help="1-based inclusive end.",
    )
    parser.add_argument(
        "--n-pcs", type=int, default=10,
        help="Number of leading X_pca components (default: %(default)s).",
    )
    parser.add_argument(
        "--knn", type=int, default=5,
        help="MAGIC neighbor parameter (default: %(default)s).",
    )
    parser.add_argument(
        "--decay", type=float, default=1.0,
        help="MAGIC kernel decay (default: %(default)s).",
    )
    parser.add_argument(
        "--t", type=int, default=3,
        help="Number of diffusion steps (default: %(default)s).",
    )
    args = parser.parse_args()
    if not args.chrom or Path(args.chrom).name != args.chrom:
        parser.error("--chrom must be a chromosome name")
    if not 1 <= args.start <= args.end:
        parser.error("coordinates must satisfy 1 <= start <= end")
    if args.n_pcs < 1 or args.knn < 1 or args.t < 0:
        parser.error("--n-pcs and --knn must be positive and --t nonnegative")
    if not np.isfinite(args.decay) or args.decay <= 0:
        parser.error("--decay must be finite and positive")
    return args


def barcode_key(barcode: str) -> str:
    """Match the RRCC and RR_CC forms of DBiT spot barcodes."""
    if len(barcode) == 5 and barcode[2] == "_":
        digits = barcode[:2] + barcode[3:]
        if digits.isascii() and digits.isdigit():
            return digits
    return barcode


def spot_keys(spots: pd.DataFrame, cov_keys: pd.Index) -> pd.Index:
    """Use the DBiT grid coordinates when COV barcodes are grid IDs."""
    keys = spots.index.map(barcode_key)
    use_grid = (
        len(cov_keys) > 0
        and cov_keys.str.fullmatch(r"[0-9]{4}").all()
        and not keys.str.fullmatch(r"[0-9]{4}").all()
    )
    if use_grid:
        columns = ["array_row", "array_col"]
        if not set(columns).issubset(spots.columns):
            raise ValueError("Grid COV barcodes require array_row and array_col")
        coordinates = spots[columns].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(coordinates).all()
            or (coordinates != np.trunc(coordinates)).any()
            or ((coordinates < 0) | (coordinates > 99)).any()
        ):
            raise ValueError("DBiT array coordinates must be integers in 0..99")
        keys = pd.Index([
            f"{row:02d}{column:02d}"
            for row, column in coordinates.astype(int)
        ])
    if keys.has_duplicates:
        raise ValueError("H5AD contains ambiguous spot barcodes")
    return keys


def load_spot_pca(
    path: Path, n_pcs: int,
) -> tuple[pd.DataFrame, np.ndarray, SpatialData]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        if adata.n_obs == 0:
            raise ValueError("H5AD contains no spots")
        if "X_pca" not in adata.obsm:
            raise ValueError("H5AD is missing obsm['X_pca']")
        scores = adata.obsm["X_pca"]
        if scores.ndim != 2 or scores.shape[1] < n_pcs:
            raise ValueError(f"H5AD has fewer than {n_pcs} PCA components")
        embedding = np.asarray(scores[:, :n_pcs], dtype=np.float64)
        if not np.isfinite(embedding).all():
            raise ValueError("Selected PCA components must be finite")
        return adata.obs, embedding, spatial_plot_data(adata)
    finally:
        adata.file.close()


def load_region(
    cov: Path, index: Path, chrom: str, start: int, end: int,
) -> pd.DataFrame:
    if not cov.is_file():
        raise FileNotFoundError(f"COV file does not exist: {cov}")
    if not index.is_file():
        raise FileNotFoundError(
            f"Missing CovTools index {index}. Build it with covtools index."
        )
    region = region_methylation(
        cov, regions=[(chrom, start, end)], index_path=index,
    )[1]["spots"]
    keys = pd.Index([barcode_key(barcode) for barcode in region])
    if keys.has_duplicates:
        raise ValueError("COV contains ambiguous spot barcodes in the interval")
    percentages = np.asarray(
        [spot["methylation_percentage"] for spot in region.values()],
        dtype=np.float64,
    )
    return pd.DataFrame(
        {
            "n_obs": pd.array(
                [spot["site_count"] for spot in region.values()], dtype="Int64",
            ),
            "meth_raw": percentages / 100,
        },
        index=keys,
    )


def diffuse_methylation(
    raw: np.ndarray, embedding: np.ndarray, knn: int, decay: float, t: int,
) -> np.ndarray:
    """Diffuse values and their observation mask, then normalize by support."""
    observed = np.isfinite(raw)
    if t == 0 or not observed.any():
        return raw.copy()
    if knn >= len(raw):
        raise ValueError(f"--knn must be less than the {len(raw)} PCA spots")

    model = magic.MAGIC(
        knn=knn, decay=decay, t=t, n_pca=None, random_state=0, verbose=0,
    )
    model.fit(embedding)
    operator = model.diff_op
    signal = np.column_stack((np.where(observed, raw, 0.0), observed))
    for _ in range(t):
        signal = operator @ signal

    numerator, support = signal.T
    result = np.full_like(raw, np.nan)
    np.divide(numerator, support, out=result, where=support > 0)
    return result


def plot_spatial_methylation(
    values: np.ndarray, spatial: SpatialData, title: str,
) -> bytes:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    image, x, y, pixel_size_um = spatial
    font_family = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Spatial coordinates must be finite")
    side = SPOT_SIZE_UM / pixel_size_um
    half = side / 2
    margin = CROP_MARGIN_UM / pixel_size_um + half
    left = max(0, x.min() - margin)
    right = min(image.shape[1], x.max() + margin)
    top = max(0, y.min() - margin)
    bottom = min(image.shape[0], y.max() + margin)
    if left >= right or top >= bottom:
        raise ValueError("Spatial coordinates do not overlap the tissue image")

    percentages = np.ma.masked_invalid(values * 100)
    observed = percentages.compressed()
    lower, upper = 0.0, 100.0
    if observed.size:
        lower, upper = observed.min(), observed.max()
        if lower == upper:
            lower = max(0, min(lower - 0.5, 99))
            upper = lower + 1

    cmap = LinearSegmentedColormap.from_list(
        "methylation", ["#4DBBD5", "#FFFFFF", "#E64B35"],
    )
    cmap.set_bad("#BDBDBD")
    squares = [
        Rectangle((spot_x - half, spot_y - half), side, side)
        for spot_x, spot_y in zip(x, y, strict=True)
    ]
    points = PatchCollection(
        squares, cmap=cmap, norm=Normalize(lower, upper),
        edgecolors="none", antialiased=False,
    )
    points.set_array(percentages)

    figure = Figure(figsize=(3, 3), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.imshow(image, cmap="gray", origin="upper", interpolation="nearest")
    axis.add_collection(points)
    axis.set_xlim(left, right)
    axis.set_ylim(bottom, top)
    axis.set_aspect("equal")
    axis.set_anchor("E")
    axis.set_title(title, pad=3, fontsize=7, fontfamily=font_family)
    axis.set_axis_off()

    target_length_um = (right - left) * pixel_size_um / 5
    magnitude = 10 ** np.floor(np.log10(target_length_um))
    scale_um = max(
        factor * magnitude for factor in (1, 2, 5)
        if factor * magnitude <= target_length_um
    )
    scale_bar = AnchoredSizeBar(
        axis.transData, scale_um / pixel_size_um, f"{scale_um:g} µm",
        loc="lower left", pad=0.2, borderpad=0.3, sep=2, color="white",
        size_vertical=max(2, (bottom - top) * 0.003),
        fontproperties=FontProperties(size=5, family=font_family),
    )
    scale_bar.patch.set(facecolor="black", alpha=0.6, edgecolor="none")
    axis.add_artist(scale_bar)

    if observed.size:
        colorbar = figure.colorbar(
            points, ax=axis, fraction=0.04, pad=0.025, shrink=0.84,
        )
        colorbar.set_label(
            "Methylation (%)", fontsize=6, fontfamily=font_family,
        )
        colorbar.ax.tick_params(labelsize=5, length=2)
        colorbar.outline.set_linewidth(0.5)
    else:
        axis.set_title(
            f"{title} (no values)", pad=3, fontsize=7, fontfamily=font_family,
        )

    with BytesIO() as png:
        figure.savefig(
            png, format="png", dpi=300, bbox_inches="tight", pad_inches=0.02,
        )
        return png.getvalue()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    region_label = f"{args.chrom}:{args.start:,}–{args.end:,}"
    region_file = f"{args.chrom}_{args.start}_{args.end}.tsv"
    output = input_path.parent / "region-magic" / region_file
    raw_plot = output.with_suffix(".raw.png")
    magic_plot = output.with_suffix(".magic.png")
    cov = args.cov.expanduser().resolve()
    index = args.index.expanduser() if args.index else Path(f"{cov}.coi")
    try:
        for path in (output, raw_plot, magic_plot):
            validate_output(path)
        spots, embedding, spatial = load_spot_pca(input_path, args.n_pcs)
        region = load_region(cov, index, args.chrom, args.start, args.end)
        result = region.reindex(spot_keys(spots, region.index))
        result.insert(0, "barcode", spots.index)
        result["n_obs"] = result["n_obs"].fillna(0)
        raw = result["meth_raw"].to_numpy(dtype=np.float64)
        imputed = diffuse_methylation(
            raw, embedding, args.knn, args.decay, args.t,
        )
        result["meth_magic"] = imputed

        artifacts = {
            output: result.to_csv(sep="\t", index=False, na_rep="NA").encode(),
            raw_plot: plot_spatial_methylation(
                raw, spatial, f"Raw · {region_label}",
            ),
            magic_plot: plot_spatial_methylation(
                imputed, spatial, f"MAGIC · {region_label}",
            ),
        }

        output.parent.mkdir(parents=True, exist_ok=True)
        written = []
        try:
            for path, data in artifacts.items():
                with path.open("xb") as handle:
                    written.append(path)
                    handle.write(data)
        except BaseException:
            for path in written:
                path.unlink(missing_ok=True)
            raise
    except (KeyError, OSError, OverflowError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(
        f"Wrote {output} ({len(spots):,} spots, "
        f"{np.isfinite(raw).sum():,} observed, "
        f"{np.isfinite(imputed).sum():,} with MAGIC estimates)"
    )
    print(f"Spatial plots: {raw_plot}, {magic_plot}")


if __name__ == "__main__":
    main()
