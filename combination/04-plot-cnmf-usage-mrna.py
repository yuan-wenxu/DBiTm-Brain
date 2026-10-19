#!/usr/bin/env python3
"""Plot cNMF usage across P7, P35 and P60 as distributions and tissue maps."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from utils import spatial_plot_data


SAMPLE_ORDER = ("p7-rep1", "p35-rep1", "p60-rep1")
AGE_LABELS = ("P7", "P35", "P60")
CORTEX_ANNOTATIONS = ("L2/3/4", "L5", "L5/6", "L6")
PANEL_SIZE = 4
SPOT_SIZE_UM = 50.0
DPI = 300
COLOR_MAP = LinearSegmentedColormap.from_list(
    "usage", ("#4DBBD5", "#FFFFFF", "#E64B35"),
)


class TissuePanel(NamedTuple):
    sample: str
    image: np.ndarray
    x: np.ndarray
    y: np.ndarray
    side: float
    indices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Integrated H5AD containing tissue images.")
    parser.add_argument("--usage", type=Path, required=True,
                        help="Normalized usage.k_K.csv from step 3.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for PNG figures only.")
    parser.add_argument(
        "--annotations", type=Path,
        help="Optional cluster_annotation.csv; select cortical-layer spots.",
    )
    return parser.parse_args()


def configure_plots() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans",
        ],
        "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 5, "axes.titlepad": 3,
    })


def read_data(input_path: Path, usage_path: Path) -> tuple[
    ad.AnnData, pd.DataFrame,
]:
    source = ad.read_h5ad(input_path, backed="r")
    try:
        required = ["sample", "pxl_row_in_fullres", "pxl_col_in_fullres"]
        missing = set(required).difference(source.obs.columns)
        if missing:
            raise ValueError(f"Missing spatial metadata: {sorted(missing)}")
        if not source.obs_names.is_unique:
            raise ValueError("Input spot IDs must be unique")
        if "leiden" in source.obs:
            required.append("leiden")
        data = ad.AnnData(obs=source.obs[required].copy())
        data.uns["spatial"] = source.uns.get("spatial", {})
    finally:
        source.file.close()
    if set(data.obs["sample"].astype(str)) != set(SAMPLE_ORDER):
        raise ValueError("Input must contain exactly P7, P35 and P60 rep1")
    usage = pd.read_csv(
        usage_path, index_col="spot_id", dtype={"spot_id": str, "sample": str},
    )
    if not usage.index.is_unique or usage.index.isna().any():
        raise ValueError("Usage spot IDs must be unique and nonmissing")
    if set(usage.index) != set(data.obs_names):
        raise ValueError("Usage and H5AD must contain exactly the same spots")
    usage = usage.loc[data.obs_names]
    if "sample" not in usage or not np.array_equal(
        usage["sample"].to_numpy(), data.obs["sample"].astype(str).to_numpy(),
    ):
        raise ValueError("Usage and H5AD sample labels must match by spot ID")
    programs = [key for key in usage if key.startswith("program_")]
    if not programs or not all(
        re.fullmatch(r"program_[1-9][0-9]*", key) for key in programs
    ):
        raise ValueError("Usage must contain program_1, program_2, ...")
    programs.sort(key=lambda key: int(key.split("_")[1]))
    values = usage[programs].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Usage must be finite and nonnegative")
    if not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Expected normalized usage summing to 1 per spot")
    return data, pd.DataFrame(values, index=usage.index, columns=programs)


def select_cortex(
    data: ad.AnnData, usage: pd.DataFrame, annotation_path: Path,
) -> tuple[ad.AnnData, pd.DataFrame]:
    if "leiden" not in data.obs or data.obs["leiden"].isna().any():
        raise ValueError("Cortex selection requires nonmissing leiden labels")
    annotations = pd.read_csv(
        annotation_path, dtype=str, keep_default_na=False, encoding="utf-8-sig",
    )
    if not {"cluster", "annotation"} <= set(annotations):
        raise ValueError("Annotation table requires cluster and annotation")
    annotations = annotations[["cluster", "annotation"]].copy()
    for column in annotations:
        annotations[column] = annotations[column].str.strip()
    if annotations.eq("").any().any():
        raise ValueError("Cluster IDs and annotations must not be blank")
    if annotations["cluster"].duplicated().any():
        raise ValueError("Annotation table has duplicate cluster IDs")
    labels = data.obs["leiden"].astype(str)
    if set(labels) != set(annotations["cluster"]):
        raise ValueError("Annotation cluster IDs must match the input H5AD")
    mapping = annotations.set_index("cluster")["annotation"]
    names = labels.map(mapping)
    selected = names.isin(CORTEX_ANNOTATIONS)
    if not selected.any():
        raise ValueError(f"No cortical annotations found: {CORTEX_ANNOTATIONS}")
    cortex = data[selected].copy()
    cortex.obs["annotation"] = names.loc[cortex.obs_names]
    if set(cortex.obs["sample"].astype(str)) != set(SAMPLE_ORDER):
        raise ValueError("Cortical spots must be present at all three ages")
    # Preserve the fitted program contributions; do not refit or renormalize.
    cortex_usage = usage.loc[cortex.obs_names].copy()
    print("Cortical labels: " + ", ".join(CORTEX_ANNOTATIONS), flush=True)
    counts = cortex.obs["sample"].value_counts().reindex(SAMPLE_ORDER)
    print(f"Selected {cortex.n_obs:,} cortical-layer spots: "
          f"{counts.to_dict()}", flush=True)
    return cortex, cortex_usage


def prepare_tissue_panels(data: ad.AnnData) -> list[TissuePanel]:
    panels = []
    for sample in SAMPLE_ORDER:
        indices = np.flatnonzero(data.obs["sample"].astype(str).eq(sample))
        if sample not in data.uns["spatial"]:
            raise ValueError(f"Missing tissue library: {sample}")
        subset = ad.AnnData(obs=data.obs.iloc[indices].copy())
        subset.uns["spatial"] = {sample: data.uns["spatial"][sample]}
        image, x, y, pixel_size = spatial_plot_data(subset)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Nonfinite spatial coordinates: {sample}")
        if image.ndim not in (2, 3) or min(image.shape[:2]) < 1:
            raise ValueError(f"Invalid tissue image: {sample}")
        panels.append(TissuePanel(
            sample, image, x, y, SPOT_SIZE_UM / pixel_size, indices,
        ))
    return panels


def plot_distribution(
    axis: plt.Axes, values: np.ndarray, panels: list[TissuePanel],
) -> None:
    groups = [values[panel.indices] for panel in panels]
    boxes = axis.boxplot(
        groups, positions=range(3), widths=0.45, patch_artist=True,
        showmeans=True,
        medianprops={"color": "#333333", "linewidth": 0.8},
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markeredgecolor": "#333333", "markersize": 3},
        flierprops={"marker": ".", "markersize": 1, "alpha": 0.25,
                    "markeredgecolor": "#555555"},
        boxprops={"linewidth": 0.6},
        whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6},
    )
    for patch in boxes["boxes"]:
        patch.set_facecolor("#4DBBD5")
        patch.set_alpha(0.7)
    axis.set_xticks(range(3), AGE_LABELS)
    axis.set_ylim(0, 1)
    axis.set_yticks(np.linspace(0, 1, 5))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axis.set_ylabel("Usage (%)", labelpad=5)
    axis.set_title("Usage across spots")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend([boxes["medians"][0], boxes["means"][0]],
                ["Median", "Mean"], frameon=False, loc="upper right")


def plot_tissue(
    axis: plt.Axes, panel: TissuePanel, values: np.ndarray,
    normalization: Normalize,
) -> PatchCollection:
    axis.imshow(panel.image, cmap="gray", origin="upper")
    patches = PatchCollection(
        [Rectangle((x - panel.side / 2, y - panel.side / 2),
                   panel.side, panel.side)
         for x, y in zip(panel.x, panel.y, strict=True)],
        cmap=COLOR_MAP, norm=normalization, edgecolors="none",
    )
    patches.set_array(values[panel.indices])
    axis.add_collection(patches)
    axis.set_xlim(0, panel.image.shape[1])
    axis.set_ylim(panel.image.shape[0], 0)
    axis.set_aspect("equal")
    axis.set_anchor("N")
    axis.set_title(f"{panel.sample} (n={len(panel.indices):,})")
    axis.set_axis_off()
    return patches


def plot_program(
    name: str, values: np.ndarray, panels: list[TissuePanel], output: Path,
    cortex_only: bool = False,
) -> None:
    figure, grid = plt.subplots(
        2, 2, figsize=(2 * PANEL_SIZE, 1.5 * PANEL_SIZE),
    )
    axes = grid.ravel()
    try:
        plot_distribution(axes[0], values, panels)
        upper = float(values.max()) or 1.0
        normalization = Normalize(vmin=0, vmax=upper)
        for axis, panel in zip(axes[1:], panels, strict=True):
            patches = plot_tissue(axis, panel, values, normalization)
        figure.subplots_adjust(
            left=0.075, right=0.89, bottom=0.07, top=0.93,
            wspace=0.12, hspace=0.16,
        )
        color_axis = figure.add_axes((0.92, 0.22, 0.024, 0.55))
        colorbar = figure.colorbar(
            patches, cax=color_axis, ticks=np.linspace(0, upper, 5),
            format=PercentFormatter(xmax=1, decimals=1),
        )
        colorbar.set_label("Usage (%)", fontsize=5)
        colorbar.ax.tick_params(labelsize=5)
        title = name.replace("program_", "Program ")
        if cortex_only:
            title += " (cortical layers)"
        figure.suptitle(f"{title}: P7 / P35 / P60", fontsize=7, y=0.985)
        figure.savefig(output, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    finally:
        plt.close(figure)


def run(args: argparse.Namespace) -> None:
    input_path = args.input.expanduser().resolve()
    usage_path = args.usage.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    data, usage = read_data(input_path, usage_path)
    if args.annotations is not None:
        annotation_path = args.annotations.expanduser().resolve()
        data, usage = select_cortex(data, usage, annotation_path)
    panels = prepare_tissue_panels(data)
    outputs = [output_dir / f"{name.replace('_', '-')}.usage.png"
               for name in usage.columns]
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("Output path is not a directory")
    if any(path.exists() for path in outputs):
        raise ValueError("Output figures exist; choose a new directory")
    configure_plots()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, output in zip(usage.columns, outputs, strict=True):
        plot_program(name, usage[name].to_numpy(), panels, output,
                     cortex_only=args.annotations is not None)
        print(f"Saved {output.name}", flush=True)
    print(f"Saved {len(outputs)} PNG figures in {output_dir}")


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
