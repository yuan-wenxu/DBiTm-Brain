#!/usr/bin/env python3
"""Plot multiple CpG saturation curves at fixed sequencing-depth intervals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOT_TITLE = "CpG saturation curves"
CURVE_FIELDS = [
    "sample",
    "target_gbp",
    "downsample_fraction",
    "median_unique_cpgs",
    "q1_unique_cpgs",
    "q3_unique_cpgs",
    "hq_spot_count",
    "reads_threshold",
    "source_clean_gbp",
    "sequencing_gbp_source",
]
METADATA_FIELDS = [
    "sample",
    "source_clean_gbp",
    "maximum_plotted_gbp",
    "sampling_step_gbp",
    "marker_step_gbp",
    "sampling_point_count",
    "reads_threshold",
    "hq_spot_count",
]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-spot CpG depth histograms into saturation curves at "
            "fixed Gbp intervals."
        )
    )
    parser.add_argument(
        "sample_dirs",
        nargs="+",
        type=Path,
        help=(
            "Sample directories that each contain fastp/fastp.json and "
            "saturation/per_spot_cpg_depth_histogram.tsv.gz."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for the combined PNG, PDF, curve TSV, and metadata TSV.",
    )
    parser.add_argument(
        "--sampling-step-gbp",
        type=float,
        default=5.0,
        help="Interval between calculated curve points in Gbp. Default: 5.",
    )
    parser.add_argument(
        "--marker-step-gbp",
        type=float,
        default=50.0,
        help="Interval between visible plot markers in Gbp. Default: 50.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for name in ("sampling_step_gbp", "marker_step_gbp"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            option = name.replace("_", "-")
            raise ValueError(f"--{option} must be a finite value > 0")
    ratio = args.marker_step_gbp / args.sampling_step_gbp
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "--marker-step-gbp must be an integer multiple of --sampling-step-gbp"
        )


def discover_samples(sample_dirs: list[Path]) -> list[dict[str, Path | str]]:
    samples: list[dict[str, Path | str]] = []
    seen_names: set[str] = set()
    for sample_dir in sample_dirs:
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"sample directory not found: {sample_dir}")

        sample_name = sample_dir.name
        if sample_name in seen_names:
            raise ValueError(f"duplicate sample name: {sample_name}")

        histogram_path = (
            sample_dir / "saturation" / "per_spot_cpg_depth_histogram.tsv.gz"
        )
        threshold_path = sample_dir / "saturation" / "reads_threshold.tsv"
        fastp_path = sample_dir / "fastp" / "fastp.json"
        required_paths = (histogram_path, threshold_path, fastp_path)
        missing_paths = [str(path) for path in required_paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(
                f"required input file(s) missing for {sample_dir}: "
                + ", ".join(missing_paths)
            )

        seen_names.add(sample_name)
        samples.append(
            {
                "sample": sample_name,
                "histogram": histogram_path,
                "fastp": fastp_path,
                "threshold": threshold_path,
            }
        )
    return samples


def read_clean_gbp(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"fastp JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_bases = payload.get("summary", {}).get("after_filtering", {}).get("total_bases")
    try:
        total_bases = int(raw_bases)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"invalid fastp after_filtering.total_bases in {path}: {raw_bases}"
        ) from error
    if total_bases <= 0:
        raise ValueError(f"fastp total bases must be > 0: {path}")
    return total_bases / 1e9


def read_reads_threshold(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"reads threshold not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        row = next(reader, None)
    if row is None or "reads_threshold" not in row:
        raise ValueError(f"invalid reads threshold file: {path}")
    try:
        threshold = float(row["reads_threshold"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid reads threshold value: {path}") from error
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"reads threshold must be > 0: {path}")
    return threshold


def calculate_sample_curve(
    sample: str,
    histogram_path: Path,
    source_clean_gbp: float,
    reads_threshold: float,
    sampling_step_gbp: float,
) -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(
        histogram_path,
        sep="\t",
        dtype={
            "spot": "string",
            "reads": np.int64,
            "depth": np.int64,
            "site_count": np.int64,
        },
    )
    required = {"spot", "reads", "depth", "site_count"}
    if not required.issubset(frame.columns):
        missing = ", ".join(sorted(required - set(frame.columns)))
        raise ValueError(f"histogram lacks columns ({missing}): {histogram_path}")
    if frame.empty:
        raise ValueError(f"empty histogram: {histogram_path}")
    if (
        frame["reads"].lt(0).any()
        or frame["depth"].le(0).any()
        or frame["site_count"].le(0).any()
    ):
        raise ValueError(f"histogram contains invalid values: {histogram_path}")
    if frame.groupby("spot", observed=True)["reads"].nunique().max() != 1:
        raise ValueError(f"histogram has inconsistent reads per spot: {histogram_path}")

    spot_codes, spot_names = pd.factorize(frame["spot"], sort=True)
    reads_by_spot = (
        frame.groupby("spot", observed=True)["reads"]
        .first()
        .reindex(spot_names)
        .to_numpy(dtype=np.int64)
    )
    hq_mask = reads_by_spot > reads_threshold
    hq_spot_count = int(hq_mask.sum())
    if hq_spot_count == 0:
        raise ValueError(f"no HQ spots above the reads threshold for {sample}")

    maximum_step = math.floor(source_clean_gbp / sampling_step_gbp)
    if maximum_step < 1:
        raise ValueError(
            f"sample {sample} has less than one sampling interval "
            f"({source_clean_gbp:g} Gbp)"
        )
    targets_gbp = sampling_step_gbp * np.arange(1, maximum_step + 1)
    depths = frame["depth"].to_numpy(dtype=np.float64)
    site_counts = frame["site_count"].to_numpy(dtype=np.float64)
    rows: list[dict[str, float | int | str]] = []
    for target_gbp in targets_gbp:
        fraction = float(target_gbp / source_clean_gbp)
        detection_probability = -np.expm1(depths * np.log1p(-fraction))
        expected_by_spot = np.bincount(
            spot_codes,
            weights=site_counts * detection_probability,
            minlength=len(spot_names),
        )[hq_mask]
        q1, median, q3 = np.percentile(expected_by_spot, [25, 50, 75])
        rows.append(
            {
                "sample": sample,
                "target_gbp": float(target_gbp),
                "downsample_fraction": fraction,
                "median_unique_cpgs": float(median),
                "q1_unique_cpgs": float(q1),
                "q3_unique_cpgs": float(q3),
                "hq_spot_count": hq_spot_count,
                "reads_threshold": reads_threshold,
                "source_clean_gbp": source_clean_gbp,
                "sequencing_gbp_source": "fastp.after_filtering.total_bases",
            }
        )
    return pd.DataFrame(rows, columns=CURVE_FIELDS), hq_spot_count


def atomic_write_table(frame: pd.DataFrame, path: Path, compression: object = None) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        frame.to_csv(
            temporary,
            sep="\t",
            index=False,
            float_format="%.9f",
            compression=compression,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_figure_atomic(figure: plt.Figure, path: Path, file_format: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        save_args: dict[str, object] = {"format": file_format, "bbox_inches": "tight"}
        if file_format == "png":
            save_args["dpi"] = 300
        figure.savefig(temporary, **save_args)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_plot(
    curves: pd.DataFrame,
    output_dir: Path,
    title: str,
    sampling_step_gbp: float,
    marker_step_gbp: float,
) -> None:
    sample_names = list(dict.fromkeys(curves["sample"]))
    colors = plt.get_cmap("tab10").colors
    figure, axis = plt.subplots(figsize=(5, 4))
    for index, sample in enumerate(sample_names):
        data = curves.loc[curves["sample"] == sample].sort_values("target_gbp")
        color = colors[index % len(colors)]
        axis.plot(
            data["target_gbp"],
            data["median_unique_cpgs"] / 1e4,
            color=color,
            linewidth=2.0,
            label=sample,
        )
        marker_mask = np.isclose(
            np.mod(data["target_gbp"].to_numpy(), marker_step_gbp),
            0.0,
            rtol=0.0,
            atol=1e-9,
        )
        axis.scatter(
            data.loc[marker_mask, "target_gbp"],
            data.loc[marker_mask, "median_unique_cpgs"] / 1e4,
            color=color,
            s=22,
            linewidths=0,
            zorder=3,
        )

    axis.set_title(f"{title} ({sampling_step_gbp:g}-Gbp increments)", fontsize=12)
    axis.set_xlabel("Sequencing depth (Gbp)", fontsize=10)
    axis.set_ylabel("Median unique CpGs per HQ spot (×10⁴)", fontsize=10)
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.25)
    axis.legend(
        title="Sample",
        loc="upper left",
        frameon=True,
        fontsize=6,
        title_fontsize=8,
        labelspacing=0.35,
        borderpad=0.4,
        handlelength=2.2,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    save_figure_atomic(
        figure, output_dir / "combined_saturation_curve_5gb.png", "png"
    )
    plt.close(figure)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
        samples = discover_samples(args.sample_dirs)
        curve_frames: list[pd.DataFrame] = []
        metadata_rows: list[dict[str, float | int | str]] = []
        for paths in samples:
            sample = str(paths["sample"])
            histogram_path = Path(paths["histogram"])
            source_clean_gbp = read_clean_gbp(Path(paths["fastp"]))
            reads_threshold = read_reads_threshold(Path(paths["threshold"]))
            curve, hq_spot_count = calculate_sample_curve(
                sample,
                histogram_path,
                source_clean_gbp,
                reads_threshold,
                args.sampling_step_gbp,
            )
            curve_frames.append(curve)
            metadata_rows.append(
                {
                    "sample": sample,
                    "source_clean_gbp": source_clean_gbp,
                    "maximum_plotted_gbp": float(curve["target_gbp"].iloc[-1]),
                    "sampling_step_gbp": args.sampling_step_gbp,
                    "marker_step_gbp": args.marker_step_gbp,
                    "sampling_point_count": len(curve),
                    "reads_threshold": reads_threshold,
                    "hq_spot_count": hq_spot_count,
                }
            )
            print(
                f"sample={sample} clean-gbp={source_clean_gbp:.9f} "
                f"points={len(curve)} hq-spots={hq_spot_count}"
            )

        curves = pd.concat(curve_frames, ignore_index=True)
        metadata = pd.DataFrame(metadata_rows, columns=METADATA_FIELDS)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_table(
            curves,
            args.output_dir / "combined_saturation_5gb.tsv.gz",
            compression="gzip",
        )
        atomic_write_table(
            metadata,
            args.output_dir / "combined_saturation_5gb_metadata.tsv",
        )
        write_plot(
            curves,
            args.output_dir,
            PLOT_TITLE,
            args.sampling_step_gbp,
            args.marker_step_gbp,
        )
        print(f"[saturation] output-dir={args.output_dir}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as error:
        print(f"[saturation] error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
