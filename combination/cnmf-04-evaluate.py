#!/usr/bin/env python3
"""Describe cNMF usage variance explained by annotation and time/sample."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


SAMPLE_ORDER = ("p7-rep1", "p35-rep1", "p60-rep1")
CORTEX_ANNOTATIONS = ("L2/3/4", "L5", "L6")
SCOPE_NAMES = {
    "all-spots": "All spots", "cortical-layers": "Cortical layers",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Current clustered H5AD with sample and leiden.")
    parser.add_argument("--usage", type=Path, required=True,
                        help="Normalized cNMF usage CSV indexed by spot_id.")
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Current cluster_annotation.csv.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for the variance CSV and PNGs.")
    return parser.parse_args()


def read_data(
    input_path: Path, usage_path: Path, annotation_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = ad.read_h5ad(input_path, backed="r")
    try:
        required = ["sample", "leiden"]
        missing = set(required).difference(source.obs.columns)
        if missing:
            raise ValueError(f"Missing spot metadata: {sorted(missing)}")
        if not source.obs_names.is_unique:
            raise ValueError("H5AD spot IDs must be unique")
        metadata = source.obs[required].copy()
    finally:
        source.file.close()
    if metadata.empty or metadata.isna().any().any():
        raise ValueError("Spot metadata must be nonempty and nonmissing")
    if set(metadata["sample"].astype(str)) != set(SAMPLE_ORDER):
        raise ValueError("Expected P7, P35 and P60 rep1 samples")
    annotations = pd.read_csv(
        annotation_path, dtype=str, keep_default_na=False,
        encoding="utf-8-sig",
    )
    if not {"cluster", "annotation"} <= set(annotations):
        raise ValueError("Annotation table needs cluster and annotation")
    annotations = annotations[["cluster", "annotation"]].copy()
    for column in annotations:
        annotations[column] = annotations[column].str.strip()
    if (annotations.eq("").any().any()
            or annotations["cluster"].duplicated().any()):
        raise ValueError("Annotations must be nonblank with unique cluster IDs")
    labels = metadata["leiden"].astype(str)
    if set(labels) != set(annotations["cluster"]):
        raise ValueError("Annotation IDs must match the current H5AD clusters")
    metadata["annotation"] = labels.map(
        annotations.set_index("cluster")["annotation"],
    )
    usage = pd.read_csv(usage_path, index_col="spot_id", dtype={"sample": str})
    if not usage.index.is_unique or usage.index.isna().any():
        raise ValueError("Usage spot IDs must be unique and nonmissing")
    if set(usage.index) != set(metadata.index):
        raise ValueError("Usage and H5AD must have exactly the same spots")
    usage = usage.loc[metadata.index]
    if "sample" not in usage or not np.array_equal(
        usage["sample"].to_numpy(), metadata["sample"].astype(str).to_numpy(),
    ):
        raise ValueError("Usage and H5AD sample labels must match by spot ID")
    programs = [name for name in usage if name.startswith("program_")]
    if not programs or not all(
        re.fullmatch(r"program_[1-9][0-9]*", name) for name in programs
    ):
        raise ValueError("Usage needs program_1, program_2, ... columns")
    programs.sort(key=lambda name: int(name.split("_")[1]))
    values = usage[programs].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError("Usage must be nonnegative and sum to 1 per spot")
    # The usage CSV can have old cluster IDs; use only the current H5AD labels.
    return metadata, pd.DataFrame(values, index=usage.index, columns=programs)


def indicator_matrix(labels: pd.Series) -> np.ndarray:
    codes, categories = pd.factorize(labels.astype(str), sort=False)
    return np.eye(len(categories), dtype=float)[codes]


def fit_r_squared(
    values: np.ndarray, design: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit OLS with an intercept in the design's column span."""
    coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coefficients
    total = np.square(values - values.mean(axis=0)).sum(axis=0)
    tolerance = np.finfo(float).eps * np.maximum(
        1, np.square(values).sum(axis=0),
    )
    r_squared = np.full(values.shape[1], np.nan)
    valid = total > tolerance
    r_squared[valid] = 1 - np.square(residual).sum(axis=0)[valid] / total[valid]
    r_squared[valid] = np.clip(r_squared[valid], 0, 1)
    adjusted = np.full_like(r_squared, np.nan)
    if len(values) > rank:
        adjusted[valid] = 1 - (1 - r_squared[valid]) * (
            (len(values) - 1) / (len(values) - rank)
        )
    return r_squared, adjusted, int(rank)


def evaluate_scope(
    metadata: pd.DataFrame, usage: pd.DataFrame, scope: str,
) -> pd.DataFrame:
    annotation = indicator_matrix(metadata["annotation"])
    time = indicator_matrix(metadata["sample"])
    interaction = indicator_matrix(
        metadata["annotation"].astype(str) + "|"
        + metadata["sample"].astype(str),
    )
    designs = {
        "annotation_only": annotation,
        "time_sample_only": time,
        "additive": np.column_stack([annotation, time]),
        "interaction": interaction,
    }
    values = usage.to_numpy(dtype=float)
    fits = {name: fit_r_squared(values, design)
            for name, design in designs.items()}
    result = pd.DataFrame({
        "scope": scope, "program": usage.columns,
        "n_spots": len(metadata),
        "n_annotations": metadata["annotation"].nunique(),
        "n_timepoints": metadata["sample"].nunique(),
    })
    for name, (r_squared, adjusted, rank) in fits.items():
        result[f"r2_{name}"] = r_squared
        result[f"adjusted_r2_{name}"] = adjusted
        result[f"rank_{name}"] = rank
    ra, rt, rat, ri = [fits[name][0] for name in (
        "annotation_only", "time_sample_only", "additive", "interaction",
    )]
    result["annotation_unique"] = rat - rt
    result["time_sample_unique"] = rat - ra
    result["shared"] = ra + rt - rat
    result["interaction_gain"] = ri - rat
    result["unexplained_additive"] = 1 - rat
    result["unexplained_full"] = 1 - ri
    result["constant_usage"] = ~np.isfinite(ra)
    valid = ~result["constant_usage"]
    # Shared components can be negative (suppression); never truncate them.
    partition = result.loc[valid, [
        "annotation_unique", "time_sample_unique", "shared",
        "interaction_gain", "unexplained_full",
    ]].sum(axis=1)
    np.testing.assert_allclose(partition, 1, atol=1e-9)
    return result


def overlap_area(first: Bbox, second: Bbox) -> float:
    intersection = Bbox.intersection(first, second)
    if intersection is None:
        return 0
    return intersection.width * intersection.height


def label_programs(axis: plt.Axes, x: np.ndarray, y: np.ndarray,
                   names: list[str]) -> None:
    """Place the few program labels without hiding adjacent points."""
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    points = axis.transData.transform(np.column_stack([x, y]))
    radius = 3 * axis.figure.dpi / 72
    point_boxes = [Bbox.from_bounds(px - radius, py - radius,
                                   2 * radius, 2 * radius)
                   for px, py in points]
    occupied = []
    offsets = [(dx, dy) for dy in (4, -8, 12, -16, 20, -24)
               for dx in (4, -4, 15, -15, 26, -26)]
    for index, name in enumerate(names):
        label = axis.annotate(name.replace("program_", "P"),
                              (x[index], y[index]), fontsize=5,
                              textcoords="offset points", xytext=(4, 4))
        candidates = []
        for dx, dy in offsets:
            label.set_position((dx, dy))
            label.set_ha("left" if dx >= 0 else "right")
            box = label.get_window_extent(renderer).expanded(1.15, 1.2)
            area = box.width * box.height
            outside = area - overlap_area(box, axis.bbox)
            overlap = sum(overlap_area(box, other) for other in occupied)
            hidden_points = sum(overlap_area(box, point_box)
                                for point_box in point_boxes)
            score = 20 * outside + 10 * overlap + 5 * hidden_points
            score += 0.01 * (dx * dx + dy * dy)
            candidates.append((score, dx, dy))
        _, dx, dy = min(candidates)
        label.set_position((dx, dy))
        label.set_ha("left" if dx >= 0 else "right")
        occupied.append(label.get_window_extent(renderer).expanded(1.15, 1.2))
        if abs(dx) > 8 or abs(dy) > 8:
            axis.annotate("", (x[index], y[index]), xytext=(dx, dy),
                          textcoords="offset points",
                          arrowprops={"arrowstyle": "-", "color": "#777777",
                                      "linewidth": 0.35, "shrinkA": 3,
                                      "shrinkB": 3}, zorder=1)


def plot_variance_scatter(
    results: pd.DataFrame, scope: str, output: Path,
) -> None:
    x = results["time_sample_unique"].to_numpy(dtype=float)
    y = results["annotation_unique"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [
            "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans",
        ], "axes.titlesize": 7, "axes.labelsize": 6,
        "xtick.labelsize": 5, "ytick.labelsize": 5,
    })
    figure, axis = plt.subplots(figsize=(2, 2))
    try:
        axis.scatter(x[valid], y[valid], s=16, facecolor="#4DBBD5",
                     edgecolor="#333333", linewidth=0.5, zorder=2)
        axis.set(xlim=(-0.025, 1.025), ylim=(-0.025, 1.025))
        axis.set_box_aspect(1)
        axis.set_xticks(np.linspace(0, 1, 5))
        axis.set_yticks(np.linspace(0, 1, 5))
        axis.set_xlabel("Fraction of variance explained\n"
                        "by time/sample (unique)", labelpad=3)
        axis.set_ylabel("Fraction of variance explained\n"
                        "by annotation (unique)", labelpad=5)
        scope_title = SCOPE_NAMES[scope]
        title = f"{scope_title}\ncNMF programs (K={len(results)})"
        axis.set_title(title, pad=3)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_linewidth(0.6)
        axis.tick_params(width=0.6, length=2, pad=2)
        figure.subplots_adjust(left=0.26, right=0.98, bottom=0.28, top=0.8)
        label_programs(axis, x[valid], y[valid],
                       results.loc[valid, "program"].tolist())
        figure.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.02)
    finally:
        plt.close(figure)


def run(args: argparse.Namespace) -> None:
    for name in ("input", "usage", "annotations", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    metadata, usage = read_data(args.input, args.usage, args.annotations)
    cortex = metadata["annotation"].isin(CORTEX_ANNOTATIONS).to_numpy()
    if not cortex.any():
        raise ValueError("No cortical layers found in the annotations")
    selections = {"all-spots": np.ones(len(metadata), dtype=bool),
                  "cortical-layers": cortex}
    for scope, selected in selections.items():
        samples = set(metadata.loc[selected, "sample"].astype(str))
        if samples != set(SAMPLE_ORDER):
            raise ValueError(f"All three ages must be present in {scope}")
    filenames = ["usage-variance-explained.csv"]
    for scope in selections:
        filenames.append(f"{scope}.variance-explained.png")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise ValueError("Output directory is a file")
    if any((args.output_dir / name).exists() for name in filenames):
        raise ValueError("Outputs exist; choose a new output directory")
    tables = []
    for scope, selected in selections.items():
        table = evaluate_scope(
            metadata.loc[selected], usage.loc[selected], scope,
        )
        tables.append(table)
        print(f"{scope}: {selected.sum():,} spots; "
              f"{table['n_annotations'].iloc[0]} annotations", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.concat(tables, ignore_index=True)
    results.to_csv(args.output_dir / filenames[0], index=False)
    for scope, table in zip(selections, tables, strict=True):
        plot_variance_scatter(
            table, scope,
            args.output_dir / f"{scope}.variance-explained.png",
        )
    print(f"Saved two PNGs and variance table in {args.output_dir}")


def main() -> None:
    try:
        run(parse_args())
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
