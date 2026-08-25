#!/usr/bin/env python3
"""Summarize and filter MethSCAn spot-by-VMR matrices."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import re
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONTEXT_ORDER = ("CA", "CC", "CG", "CT")
MATRIX_NAMES = {
    "fraction": "methylation_fractions.csv.gz",
    "methylated": "methylated_sites.csv.gz",
    "total": "total_sites.csv.gz",
    "residual": "mean_shrunken_residuals.csv.gz",
}
VMR_PATTERN = re.compile(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")


@dataclass
class ContextResult:
    context: str
    vmrs: list[str]
    spot_rows: list[dict[str, object]]
    vmr_observed_spots: list[int]
    summary: dict[str, object]


@dataclass
class FilterResult:
    context: str
    spot_cutoff: int
    kept_spots: set[str]
    kept_vmr_indices: list[int]
    kept_vmr_observed_spots: list[int]
    spot_filter_rows: list[dict[str, object]]
    vmr_filter_rows: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream MethSCAn fraction/count/residual matrices, report matrix "
            "sparsity, and write filtered matrix copies."
        )
    )
    parser.add_argument(
        "--methscan-dir",
        type=Path,
        required=True,
        help="MethSCAn output root containing CONTEXT/matrix directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: METHSCAN_DIR/matrix_qc).",
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        help="Contexts to inspect (default: auto-detect, ordered as CA CC CG CT).",
    )
    parser.add_argument(
        "--spot-filter-quantile",
        type=float,
        default=0.05,
        help=(
            "Per-context lower quantile of observed-VMR counts used as the spot "
            "cutoff; 0 keeps all spots (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--min-vmr-observed-spots",
        type=int,
        default=10,
        help=(
            "Minimum retained spots with coverage required to keep a VMR "
            "(default: 10)."
        ),
    )
    args = parser.parse_args()
    if not 0 <= args.spot_filter_quantile < 1:
        parser.error("--spot-filter-quantile must be >= 0 and < 1")
    if args.min_vmr_observed_spots < 1:
        parser.error("--min-vmr-observed-spots must be >= 1")
    return args


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open(mode="r", encoding="utf-8", newline="")


def matrix_paths(methscan_dir: Path, context: str) -> dict[str, Path]:
    matrix_dir = methscan_dir / context / "matrix"
    return {key: matrix_dir / name for key, name in MATRIX_NAMES.items()}


def discover_contexts(methscan_dir: Path) -> list[str]:
    contexts = []
    for path in methscan_dir.iterdir():
        if not path.is_dir():
            continue
        if any(
            candidate.is_file()
            for candidate in matrix_paths(methscan_dir, path.name).values()
        ):
            contexts.append(path.name)
    order = {context: index for index, context in enumerate(CONTEXT_ORDER)}
    return sorted(contexts, key=lambda context: (order.get(context, len(order)), context))


def canonical_spot(row_id: str, context: str) -> str:
    suffix = f".{context}"
    return row_id[: -len(suffix)] if row_id.endswith(suffix) else row_id


def parse_count(raw: str, path: Path, row_number: int, vmr: str) -> int:
    if raw == "":
        return 0
    try:
        value = int(raw)
    except ValueError:
        try:
            numeric_value = float(raw)
        except ValueError as error:
            raise ValueError(
                f"Invalid count in {path} row {row_number}, VMR {vmr}: {raw!r}"
            ) from error
        if not math.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(
                f"Count must be a non-negative integer in {path} row {row_number}, "
                f"VMR {vmr}: {raw!r}"
            )
        value = int(numeric_value)
    if value < 0:
        raise ValueError(
            f"Count must be a non-negative integer in {path} row {row_number}, "
            f"VMR {vmr}: {raw!r}"
        )
    return value


def parse_float(raw: str, path: Path, row_number: int, vmr: str) -> float:
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(
            f"Invalid numeric value in {path} row {row_number}, VMR {vmr}: {raw!r}"
        ) from error


def percentile(values: list[int], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def format_float(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.8f}"


def inspect_context(
    methscan_dir: Path,
    context: str,
) -> ContextResult:
    paths = matrix_paths(methscan_dir, context)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing matrix file(s): " + ", ".join(str(path) for path in missing)
        )

    with ExitStack() as stack:
        readers = {
            key: csv.reader(stack.enter_context(open_text(path)))
            for key, path in paths.items()
        }
        headers: dict[str, list[str]] = {}
        for key, reader in readers.items():
            try:
                headers[key] = next(reader)
            except StopIteration as error:
                raise ValueError(f"Matrix is empty: {paths[key]}") from error

        reference_header = headers["fraction"]
        if len(reference_header) < 2:
            raise ValueError(f"Matrix has no VMR columns: {paths['fraction']}")
        for key, header in headers.items():
            if header != reference_header:
                raise ValueError(
                    f"Matrix header differs between {paths['fraction']} and {paths[key]}"
                )

        vmrs = reference_header[1:]
        if len(vmrs) != len(set(vmrs)):
            raise ValueError(f"Duplicate VMR columns in {paths['fraction']}")
        for vmr in vmrs:
            match = VMR_PATTERN.fullmatch(vmr)
            if match is None or int(match.group("end")) <= int(match.group("start")):
                raise ValueError(f"Invalid VMR coordinate in {paths['fraction']}: {vmr}")

        vmr_count = len(vmrs)
        vmr_observed_spots = [0] * vmr_count # Count the number of spots observed for each VMR across all spots.
        spot_rows: list[dict[str, object]] = []
        spot_names: set[str] = set()
        row_ids: set[str] = set()

        rows = zip_longest(
            readers["fraction"],
            readers["methylated"],
            readers["total"],
            readers["residual"],
        )
        for row_number, matrices in enumerate(rows, start=2):
            if any(row is None for row in matrices):
                raise ValueError(f"Matrix row counts differ for context {context}")
            fraction_row, methylated_row, total_row, residual_row = matrices
            assert fraction_row is not None
            assert methylated_row is not None
            assert total_row is not None
            assert residual_row is not None
            for key, row in zip(
                ("fraction", "methylated", "total", "residual"), matrices
            ):
                assert row is not None
                if len(row) != len(reference_header):
                    raise ValueError(
                        f"Expected {len(reference_header)} columns in {paths[key]} "
                        f"row {row_number}, found {len(row)}"
                    )

            row_id = fraction_row[0]
            row_ids_this_line = {
                fraction_row[0], methylated_row[0], total_row[0], residual_row[0]
            }
            if len(row_ids_this_line) != 1:
                raise ValueError(
                    f"Matrix row identifiers differ at context {context}, row {row_number}: "
                    + ", ".join(sorted(row_ids_this_line))
                )
            if not row_id:
                raise ValueError(f"Empty matrix row identifier at {context} row {row_number}")
            if row_id in row_ids:
                raise ValueError(f"Duplicate matrix row identifier in {context}: {row_id}")
            row_ids.add(row_id)
            spot = canonical_spot(row_id, context)
            if spot in spot_names:
                raise ValueError(f"Duplicate canonical spot in {context}: {spot}")
            spot_names.add(spot)

            spot_observed = 0 # Count the number of VMR sites observed in a single spot.
            for index, (fraction_raw, methylated_raw, total_raw, residual_raw) in enumerate(
                zip(
                    fraction_row[1:],
                    methylated_row[1:],
                    total_row[1:],
                    residual_row[1:],
                )
            ):
                vmr = vmrs[index]
                total = parse_count(total_raw, paths["total"], row_number, vmr)
                methylated = parse_count(
                    methylated_raw, paths["methylated"], row_number, vmr
                )

                has_coverage = total > 0
                if has_coverage:
                    spot_observed += 1
                    vmr_observed_spots[index] += 1

                    if fraction_raw != "":
                        parse_float(
                            fraction_raw, paths["fraction"], row_number, vmr
                        )
                    if residual_raw != "":
                        residual = parse_float(
                            residual_raw, paths["residual"], row_number, vmr
                        )
                        if not math.isfinite(residual):
                            raise ValueError(
                                f"Non-finite residual in {paths['residual']} row "
                                f"{row_number}, VMR {vmr}: {residual_raw!r}"
                            )
                else:
                    if methylated != 0:
                        raise ValueError(
                            f"Methylated count without coverage in "
                            f"{paths['methylated']} row {row_number}, VMR {vmr}: "
                            f"methylated={methylated}"
                        )

            spot_rows.append(
                {
                    "spot": spot,
                    "vmr_count": vmr_count,
                    "observed_vmrs": spot_observed, # Count of VMRs observed for this spot.
                    "observed_fraction": spot_observed / vmr_count,
                }
            )

    if not spot_rows:
        raise ValueError(f"Matrix has no spot rows: {paths['fraction']}")

    observed_size = sum(vmr_observed_spots)
    matrix_size = len(spot_rows) * len(vmrs)
    summary: dict[str, object] = {
        "context": context,
        "spot_count": len(spot_rows),
        "vmr_count": len(vmrs),
        "matrix_size": matrix_size,
        "observed_size": observed_size,
        "missing_size": matrix_size - observed_size,
        "observed_fraction": observed_size / matrix_size if matrix_size else math.nan,
    }

    return ContextResult(
        context=context,
        vmrs=vmrs,
        spot_rows=spot_rows,
        vmr_observed_spots=vmr_observed_spots,
        summary=summary,
    )


def make_filter_result(
    methscan_dir: Path,
    result: ContextResult,
    spot_filter_quantile: float,
    min_vmr_observed_spots: int,
) -> FilterResult:
    context = result.context
    observed_counts = [int(row["observed_vmrs"]) for row in result.spot_rows]
    spot_cutoff = math.ceil(percentile(observed_counts, spot_filter_quantile))
    kept_spots = {
        str(row["spot"])
        for row in result.spot_rows
        if int(row["observed_vmrs"]) >= spot_cutoff
    }
    if not kept_spots:
        raise ValueError(
            f"Spot filter removed every {context} spot at cutoff {spot_cutoff}"
        )

    spot_filter_rows = []
    for row in result.spot_rows:
        keep = str(row["spot"]) in kept_spots
        spot_filter_rows.append(
            {
                "context": context,
                "spot": row["spot"],
                "observed_vmrs": row["observed_vmrs"],
                "total_vmrs": row["vmr_count"],
                "observed_fraction": format_float(float(row["observed_fraction"])),
                "observed_vmrs_cutoff": spot_cutoff,
                "keep": int(keep),
                "filter_reason": "kept" if keep else "low_observed_vmr_count",
            }
        )

    total_path = matrix_paths(methscan_dir, context)["total"]
    kept_vmr_observed_spots = [0] * len(result.vmrs)
    with open_text(total_path) as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"Matrix is empty: {total_path}") from error
        if header[1:] != result.vmrs:
            raise ValueError(f"VMR header changed while filtering: {total_path}")
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"Expected {len(header)} columns in {total_path} row "
                    f"{row_number}, found {len(row)}"
                )
            spot = canonical_spot(row[0], context)
            if spot not in kept_spots:
                continue
            for index, raw in enumerate(row[1:]):
                if parse_count(raw, total_path, row_number, result.vmrs[index]) > 0:
                    kept_vmr_observed_spots[index] += 1

    kept_vmr_indices = [
        index
        for index, count in enumerate(kept_vmr_observed_spots)
        if count >= min_vmr_observed_spots
    ]
    if not kept_vmr_indices:
        raise ValueError(
            f"VMR filter removed every {context} VMR at minimum support "
            f"{min_vmr_observed_spots}"
        )

    vmr_filter_rows = []
    for index, vmr in enumerate(result.vmrs):
        observed = kept_vmr_observed_spots[index]
        keep = observed >= min_vmr_observed_spots
        vmr_filter_rows.append(
            {
                "context": context,
                "vmr": vmr,
                "observed_spots_after_spot_filter": observed,
                "retained_spots": len(kept_spots),
                "observed_fraction_after_spot_filter": format_float(
                    observed / len(kept_spots)
                ),
                "minimum_observed_spots": min_vmr_observed_spots,
                "keep": int(keep),
                "filter_reason": "kept" if keep else "low_observed_spot_count",
            }
        )

    return FilterResult(
        context=context,
        spot_cutoff=spot_cutoff,
        kept_spots=kept_spots,
        kept_vmr_indices=kept_vmr_indices,
        kept_vmr_observed_spots=kept_vmr_observed_spots,
        spot_filter_rows=spot_filter_rows,
        vmr_filter_rows=vmr_filter_rows,
    )


def write_filtered_matrices(
    methscan_dir: Path,
    filtered_root: Path,
    result: ContextResult,
    filtering: FilterResult,
) -> None:
    input_paths = matrix_paths(methscan_dir, result.context)
    output_dir = filtered_root / result.context / "matrix"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_columns = [index + 1 for index in filtering.kept_vmr_indices]

    for key, input_path in input_paths.items():
        output_path = output_dir / MATRIX_NAMES[key]
        temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
        try:
            with open_text(input_path) as input_handle, gzip.open(
                temporary, mode="wt", encoding="utf-8", newline=""
            ) as output_handle:
                reader = csv.reader(input_handle)
                writer = csv.writer(output_handle, lineterminator="\n")
                try:
                    header = next(reader)
                except StopIteration as error:
                    raise ValueError(f"Matrix is empty: {input_path}") from error
                if header[1:] != result.vmrs:
                    raise ValueError(
                        f"VMR header changed while writing filtered matrix: {input_path}"
                    )
                writer.writerow([header[0], *(header[index] for index in selected_columns)])
                for row_number, row in enumerate(reader, start=2):
                    if len(row) != len(header):
                        raise ValueError(
                            f"Expected {len(header)} columns in {input_path} row "
                            f"{row_number}, found {len(row)}"
                        )
                    if canonical_spot(row[0], result.context) not in filtering.kept_spots:
                        continue
                    writer.writerow([row[0], *(row[index] for index in selected_columns)])
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open(mode="w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_sparsity(path: Path, results: list[ContextResult]) -> None:
    contexts = [result.context for result in results]
    spot_values = [
        [100 * int(row["observed_vmrs"]) / len(result.vmrs) for row in result.spot_rows]
        for result in results
    ]
    vmr_values = [
        [100 * count / len(result.spot_rows) for count in result.vmr_observed_spots]
        for result in results
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, values, title, ylabel in (
        (
            axes[0],
            spot_values,
            "Observed VMRs per spot",
            "Observed matrix entries per spot (%)",
        ),
        (
            axes[1],
            vmr_values,
            "Observed spots per VMR",
            "Observed matrix entries per VMR (%)",
        ),
    ):
        boxes = axis.boxplot(
            values,
            tick_labels=contexts,
            showfliers=False,
            patch_artist=True,
        )
        for box in boxes["boxes"]:
            box.set_facecolor("#8491B4")
            box.set_alpha(0.8)
        axis.set_title(title, fontsize=12)
        axis.set_ylabel(ylabel, fontsize=10)
        axis.tick_params(axis="x", labelsize=10)
        axis.tick_params(axis="y", labelsize=10)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("MethSCAn matrix sparsity (blank fraction = no usable coverage)", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_filter_distributions(
    path: Path, filter_results: list[FilterResult], min_vmr_observed_spots: int
) -> None:
    """Plot lower-tail support distributions used by the two filters."""

    figure, axes = plt.subplots(
        2,
        len(filter_results),
        figsize=(max(8.0, 4 * len(filter_results)), 6.4),
        squeeze=False,
        sharey="row",
    )

    def draw_histogram(
        axis: plt.Axes,
        values: list[int],
        cutoff: int,
        title: str,
        xlabel: str,
    ) -> None:
        lower = min(values)
        upper = max(lower, math.ceil(percentile(values, 0.995)))
        upper = max(upper, cutoff + 1)
        displayed = [value for value in values if value <= upper]
        span = upper - lower + 1
        bin_count = min(80, span, max(10, round(math.sqrt(len(displayed)))))
        bin_count = max(1, bin_count)
        axis.hist(
            displayed,
            bins=bin_count,
            range=(lower - 0.5, upper + 0.5),
            color="#4DBBD5",
            edgecolor="white",
            linewidth=0.35,
        )
        axis.axvspan(lower - 0.5, cutoff, color="#E64B35", alpha=0.16)
        axis.axvline(cutoff, color="#B2182B", linestyle="--", linewidth=1.5)
        axis.set_yscale("log")
        axis.set_title(title, fontsize=12)
        axis.set_xlabel(xlabel, fontsize=10)
        axis.grid(axis="y", alpha=0.2)

        removed = sum(value < cutoff for value in values)
        omitted = len(values) - len(displayed)
        annotation = (
            f"keep >= {cutoff:,}\n"
            f"below cutoff: {removed:,}/{len(values):,} ({removed / len(values):.1%})"
        )
        if omitted:
            annotation += f"\nright tail hidden: {omitted:,} (> {upper:,})"
        axis.text(
            0.98,
            0.95,
            annotation,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )

    for column, filtering in enumerate(filter_results):
        spot_values = [
            int(row["observed_vmrs"]) for row in filtering.spot_filter_rows
        ]
        vmr_values = [
            int(row["observed_spots_after_spot_filter"])
            for row in filtering.vmr_filter_rows
        ]
        draw_histogram(
            axes[0][column],
            spot_values,
            filtering.spot_cutoff,
            f"{filtering.context}: spot support",
            "Observed VMRs per spot",
        )
        draw_histogram(
            axes[1][column],
            vmr_values,
            min_vmr_observed_spots,
            f"{filtering.context}: VMR support",
            "Observed spots per VMR after spot filtering",
        )

    axes[0][0].set_ylabel("Spots (log scale)", fontsize = 10)
    axes[1][0].set_ylabel("VMRs (log scale)", fontsize = 10)
    figure.suptitle(
        "Distributions used to choose spot and VMR filtering thresholds",
        fontsize=12,
    )
    figure.text(
        0.5,
        0.01,
        "Red shading marks values below the current cutoff; x-axes focus on the "
        "lower 99.5% to make the filtering boundary visible.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    methscan_dir = args.methscan_dir.expanduser().resolve()
    if not methscan_dir.is_dir():
        raise SystemExit(f"MethSCAn directory does not exist: {methscan_dir}")

    contexts = (
        [context.upper() for context in args.contexts]
        if args.contexts
        else discover_contexts(methscan_dir)
    )
    if not contexts:
        raise SystemExit(f"No complete MethSCAn matrix sets found under {methscan_dir}")
    if len(contexts) != len(set(contexts)):
        raise SystemExit("--contexts contains duplicate values")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else methscan_dir / "matrix_qc"
    )
    results: list[ContextResult] = []
    try:
        for context in contexts:
            print(f"Inspecting {context} matrices...", flush=True)
            result = inspect_context(
                methscan_dir,
                context,
            )
            results.append(result)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        raise SystemExit(str(error)) from error

    filter_results: list[FilterResult] = []
    try:
        for result in results:
            print(f"Filtering {result.context} matrices...", flush=True)
            filtering = make_filter_result(
                methscan_dir,
                result,
                args.spot_filter_quantile,
                args.min_vmr_observed_spots,
            )
            filter_results.append(filtering)
            retained_observed_size = sum(
                filtering.kept_vmr_observed_spots[index]
                for index in filtering.kept_vmr_indices
            )
            retained_matrix_size = len(filtering.kept_spots) * len(
                filtering.kept_vmr_indices
            )
            result.summary.update(
                {
                    "spot_filter_quantile": args.spot_filter_quantile,
                    "spot_observed_vmrs_cutoff": filtering.spot_cutoff,
                    "retained_spots": len(filtering.kept_spots),
                    "filtered_spots": len(result.spot_rows)
                    - len(filtering.kept_spots),
                    "minimum_vmr_observed_spots": args.min_vmr_observed_spots,
                    "retained_vmrs": len(filtering.kept_vmr_indices),
                    "filtered_vmrs": len(result.vmrs)
                    - len(filtering.kept_vmr_indices),
                    "filtered_matrix_size": retained_matrix_size,
                    "filtered_observed_size": retained_observed_size,
                    "filtered_observed_fraction": (
                        retained_observed_size / retained_matrix_size
                        if retained_matrix_size
                        else math.nan
                    ),
                }
            )
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        raise SystemExit(str(error)) from error

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_fields = list(results[0].summary)
    extra_summary_fields = sorted(
        set().union(*(result.summary for result in results)) - set(summary_fields)
    )
    summary_fields.extend(extra_summary_fields)
    summary_rows = []
    for result in results:
        row = dict(result.summary)
        for field in summary_fields:
            row.setdefault(field, "NA")
        for field in (
            "observed_fraction",
            "spot_filter_quantile",
            "filtered_observed_fraction",
        ):
            row[field] = format_float(float(row[field]))
        summary_rows.append(row)

    summary_path = output_dir / "matrix_qc_summary.tsv"
    spot_filter_path = output_dir / "spot_filter.tsv.gz"
    vmr_filter_path = output_dir / "vmr_filter.tsv.gz"
    sparsity_figure = output_dir / "matrix_sparsity.png"
    filter_distribution_figure = output_dir / "filter_threshold_distributions.png"
    filtered_root = output_dir / "filtered_methscan"

    write_tsv(summary_path, summary_rows, summary_fields)
    write_gzip_tsv(
        spot_filter_path,
        [row for filtering in filter_results for row in filtering.spot_filter_rows],
        [
            "context",
            "spot",
            "observed_vmrs",
            "total_vmrs",
            "observed_fraction",
            "observed_vmrs_cutoff",
            "keep",
            "filter_reason",
        ],
    )
    write_gzip_tsv(
        vmr_filter_path,
        [row for filtering in filter_results for row in filtering.vmr_filter_rows],
        [
            "context",
            "vmr",
            "observed_spots_after_spot_filter",
            "retained_spots",
            "observed_fraction_after_spot_filter",
            "minimum_observed_spots",
            "keep",
            "filter_reason",
        ],
    )
    try:
        for result, filtering in zip(results, filter_results):
            print(f"Writing filtered {result.context} matrices...", flush=True)
            write_filtered_matrices(
                methscan_dir, filtered_root, result, filtering
            )
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        raise SystemExit(str(error)) from error

    plot_sparsity(sparsity_figure, results)
    plot_filter_distributions(
        filter_distribution_figure, filter_results, args.min_vmr_observed_spots
    )

    print(f"Contexts: {', '.join(contexts)}")
    for result in results:
        filtering = next(
            item for item in filter_results if item.context == result.context
        )
        print(
            f"{result.context}: spots={len(result.spot_rows):,}; "
            f"VMRs={len(result.vmrs):,}; "
            f"observed={float(result.summary['observed_fraction']):.2%}; "
            f"retained={len(filtering.kept_spots):,} spots x "
            f"{len(filtering.kept_vmr_indices):,} VMRs"
        )
    print(f"Summary: {summary_path}")
    print(f"Spot filter: {spot_filter_path}")
    print(f"VMR filter: {vmr_filter_path}")
    print(f"Sparsity figure: {sparsity_figure}")
    print(f"Filter threshold distributions: {filter_distribution_figure}")


if __name__ == "__main__":
    main()
