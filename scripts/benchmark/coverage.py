#!/usr/bin/env python3
"""Plot genomic and spatial coverage for CG, CA, CC, and CT sites."""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import math
import statistics
from collections import defaultdict
from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONTEXTS = ("CG", "CA", "CC", "CT")
WINDOW_SIZES = (100_000, 1_000_000)
DEFAULT_CHROMOSOMES = tuple(
    [f"chr{number}" for number in range(1, 20)] + ["chrX"]
)
PLOT_COLOR = "#56B4E9"
FASTA_CHUNK_BASES = 4 * 1024 * 1024


@dataclass(frozen=True)
class SpotRecord:
    path: Path
    spot: str
    x_index: int
    y_index: int
    observed_sites: int
    retained: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host-dir",
        type=Path,
        required=True,
        help="Directory recursively containing <spot>.<context>.cov files.",
    )
    parser.add_argument(
        "--reference-fasta",
        type=Path,
        required=True,
        help="Reference FASTA used by the methylation caller; plain or gzip.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--chromosomes",
        default=",".join(DEFAULT_CHROMOSOMES),
        help=(
            "Comma-separated reference chromosomes to plot (default: "
            "chr1-chr19,chrX)."
        ),
    )
    parser.add_argument(
        "--min-total-sites",
        type=int,
        default=100,
        help="Minimum unique sites required to retain a spot (default: 100).",
    )
    args = parser.parse_args()
    if args.min_total_sites < 0:
        parser.error("--min-total-sites cannot be negative")
    chromosomes = [item.strip() for item in args.chromosomes.split(",")]
    if not chromosomes or any(not item for item in chromosomes):
        parser.error("--chromosomes must contain non-empty names")
    if len(chromosomes) != len(set(chromosomes)):
        parser.error("--chromosomes cannot contain duplicate names")
    args.chromosomes = chromosomes
    return args


def open_text(path: Path, *, encoding: str = "utf-8") -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding=encoding)
    return path.open(mode="r", encoding=encoding)


def add_reference_sites(
    chrom: str,
    context: str,
    mask: np.ndarray,
    coordinate_offset: int,
    data_start: int,
    chromosome_counts: dict[str, dict[str, int]],
    window_counts: dict[int, dict[tuple[str, str, int], int]],
) -> None:
    local_indices = np.flatnonzero(mask)
    if not local_indices.size:
        return
    chromosome_counts[chrom][context] += int(local_indices.size)
    coordinates = local_indices.astype(np.int64, copy=False)
    coordinates += data_start + coordinate_offset
    for size in WINDOW_SIZES:
        indices, counts = np.unique(coordinates // size, return_counts=True)
        for window_index, count in zip(indices, counts):
            window_counts[size][(chrom, context, int(window_index))] += int(count)


def count_reference_chunk(
    chrom: str,
    sequence: bytes,
    carry: bytes,
    bases_processed: int,
    chromosome_counts: dict[str, dict[str, int]],
    window_counts: dict[int, dict[tuple[str, str, int], int]],
) -> bytes:
    if not sequence:
        return carry
    data = carry + sequence.upper()
    if len(data) < 2:
        return data
    data_start = bases_processed - len(carry)
    bases = np.frombuffer(data, dtype=np.uint8)
    previous = bases[:-1]
    current = bases[1:]
    a, c, g, t = (ord(base) for base in "ACGT")
    masks = {
        "CG": (((previous == c) & (current == g)), None),
        "CA": (
            ((previous == c) & (current == a)),
            ((previous == t) & (current == g)),
        ),
        "CC": (
            ((previous == c) & (current == c)),
            ((previous == g) & (current == g)),
        ),
        "CT": (
            ((previous == c) & (current == t)),
            ((previous == a) & (current == g)),
        ),
    }
    for context, (forward_mask, reverse_mask) in masks.items():
        add_reference_sites(
            chrom,
            context,
            forward_mask,
            0,
            data_start,
            chromosome_counts,
            window_counts,
        )
        if reverse_mask is not None:
            add_reference_sites(
                chrom,
                context,
                reverse_mask,
                1,
                data_start,
                chromosome_counts,
                window_counts,
            )
    return data[-1:]


def scan_reference(
    path: Path, selected_chromosomes: list[str]
) -> tuple[
    dict[str, int],
    dict[str, dict[str, int]],
    dict[int, dict[tuple[str, str, int], int]],
]:
    selected = set(selected_chromosomes)
    chromosome_lengths: dict[str, int] = {}
    chromosome_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {context: 0 for context in CONTEXTS}
    )
    window_counts: dict[int, dict[tuple[str, str, int], int]] = {
        size: defaultdict(int) for size in WINDOW_SIZES
    }
    current_chrom: str | None = None
    include_current = False
    buffer = bytearray()
    carry = b""
    bases_processed = 0

    def flush_buffer() -> None:
        nonlocal carry, bases_processed
        if not buffer or current_chrom is None or not include_current:
            return
        sequence = bytes(buffer)
        carry = count_reference_chunk(
            current_chrom,
            sequence,
            carry,
            bases_processed,
            chromosome_counts,
            window_counts,
        )
        bases_processed += len(sequence)
        buffer.clear()

    def finish_chromosome() -> None:
        if current_chrom is None or not include_current:
            return
        flush_buffer()
        chromosome_lengths[current_chrom] = bases_processed
        counts = chromosome_counts[current_chrom]
        print(
            f"Reference {current_chrom}: {bases_processed:,} bp; "
            + ", ".join(f"{context}={counts[context]:,}" for context in CONTEXTS),
            flush=True,
        )

    with open_text(path, encoding="ascii") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if raw_line.startswith(">"):
                finish_chromosome()
                header = raw_line[1:].strip()
                if not header:
                    raise ValueError(f"Empty FASTA header at {path}:{line_number}")
                current_chrom = header.split()[0]
                if current_chrom in chromosome_lengths:
                    raise ValueError(f"Duplicate FASTA record: {current_chrom}")
                include_current = current_chrom in selected
                buffer.clear()
                carry = b""
                bases_processed = 0
                continue
            if current_chrom is None:
                if raw_line.strip():
                    raise ValueError(f"Sequence before first FASTA header: {path}")
                continue
            if not include_current:
                continue
            sequence = raw_line.strip().encode("ascii").upper()
            if not sequence:
                continue
            invalid = set(sequence) - set(b"ACGTN")
            if invalid:
                symbols = "".join(sorted(chr(value) for value in invalid))
                raise ValueError(
                    f"Unsupported FASTA symbols {symbols!r} at {path}:{line_number}"
                )
            buffer.extend(sequence)
            if len(buffer) >= FASTA_CHUNK_BASES:
                flush_buffer()
        finish_chromosome()

    missing = selected - set(chromosome_lengths)
    if missing:
        raise ValueError(
            "Selected chromosomes absent from FASTA: " + ", ".join(sorted(missing))
        )
    return chromosome_lengths, chromosome_counts, window_counts


def discover_files(host_dir: Path, context: str) -> list[Path]:
    paths = set(host_dir.rglob(f"*.{context}.cov"))
    paths.update(host_dir.rglob(f"*.{context}.cov.gz"))
    return sorted(paths)


def sample_name(path: Path, context: str) -> str:
    for suffix in (f".{context}.cov.gz", f".{context}.cov"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    raise ValueError(f"Unexpected {context} coverage filename: {path.name}")


def parse_spot_coordinates(spot: str) -> tuple[int, int]:
    fields = spot.rsplit("_", maxsplit=1)
    if len(fields) != 2:
        raise ValueError(f"Spot name does not encode X_Y coordinates: {spot}")
    try:
        x_index, y_index = int(fields[0]), int(fields[1])
    except ValueError as error:
        raise ValueError(
            f"Spot name does not encode integer X_Y coordinates: {spot}"
        ) from error
    if x_index < 0 or y_index < 0:
        raise ValueError(f"Spot coordinates cannot be negative: {spot}")
    return x_index, y_index


def iter_coverage_sites(
    path: Path,
    context: str,
    chromosome_ranks: dict[str, int],
) -> Iterator[tuple[int, int, str]]:
    """Yield unique selected sites from one coordinate-sorted coverage file."""
    previous_key: tuple[int, int] | None = None
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            minimum_fields = 6 if context == "CG" else 8
            if len(fields) < minimum_fields:
                raise ValueError(
                    f"Expected at least {minimum_fields} columns at "
                    f"{path}:{line_number}"
                )
            if context != "CG":
                if fields[6] != context:
                    raise ValueError(
                        f"Expected {context} in column 7 at {path}:{line_number}"
                    )
                if fields[7] not in {"+", "-"}:
                    raise ValueError(
                        f"Invalid strand in column 8 at {path}:{line_number}"
                    )
            chrom = fields[0]
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError as error:
                raise ValueError(
                    f"Invalid coverage coordinate at {path}:{line_number}"
                ) from error
            if start < 0 or start != end:
                raise ValueError(
                    f"Coverage start/end must be the same non-negative coordinate "
                    f"at {path}:{line_number}"
                )
            rank = chromosome_ranks.get(chrom)
            if rank is None:
                continue
            key = (rank, start)
            if previous_key is not None and key < previous_key:
                raise ValueError(
                    f"Coverage sites are not coordinate-sorted at {path}:{line_number}"
                )
            if key != previous_key:
                yield rank, start, chrom
            previous_key = key


def count_spot_sites(
    path: Path, context: str, chromosome_ranks: dict[str, int]
) -> int:
    return sum(1 for _ in iter_coverage_sites(path, context, chromosome_ranks))


def read_spot_records(
    host_dir: Path,
    context: str,
    chromosome_ranks: dict[str, int],
    min_total_sites: int,
) -> list[SpotRecord]:
    paths = discover_files(host_dir, context)
    if not paths:
        raise ValueError(f"No *.{context}.cov files found below {host_dir}")
    records = []
    seen_spots: set[str] = set()
    for file_number, path in enumerate(paths, start=1):
        spot = sample_name(path, context)
        if spot in seen_spots:
            raise ValueError(f"Duplicate {context} coverage files for spot: {spot}")
        seen_spots.add(spot)
        x_index, y_index = parse_spot_coordinates(spot)
        observed_sites = count_spot_sites(path, context, chromosome_ranks)
        records.append(
            SpotRecord(
                path=path,
                spot=spot,
                x_index=x_index,
                y_index=y_index,
                observed_sites=observed_sites,
                retained=observed_sites >= min_total_sites,
            )
        )
        if file_number % 100 == 0:
            print(f"Counted {file_number:,} {context} coverage files", flush=True)
    return records


def merge_observed_sites(
    records: list[SpotRecord],
    context: str,
    chromosome_ranks: dict[str, int],
) -> tuple[int, dict[str, int], dict[int, dict[tuple[str, int], int]]]:
    """Merge retained sorted files without holding all observed sites in memory."""
    chromosome_observed: dict[str, int] = defaultdict(int)
    window_observed: dict[int, dict[tuple[str, int], int]] = {
        size: defaultdict(int) for size in WINDOW_SIZES
    }
    retained = [record for record in records if record.retained]
    heap: list[tuple[int, int, int, str, Iterator[tuple[int, int, str]]]] = []
    with ExitStack() as stack:
        for file_index, record in enumerate(retained):
            iterator = stack.enter_context(
                closing(iter_coverage_sites(record.path, context, chromosome_ranks))
            )
            try:
                rank, position, chrom = next(iterator)
            except StopIteration:
                continue
            heapq.heappush(heap, (rank, position, file_index, chrom, iterator))

        previous_site: tuple[int, int] | None = None
        unique_sites = 0
        while heap:
            rank, position, file_index, chrom, iterator = heapq.heappop(heap)
            site = (rank, position)
            if site != previous_site:
                unique_sites += 1
                chromosome_observed[chrom] += 1
                for size in WINDOW_SIZES:
                    window_observed[size][(chrom, position // size)] += 1
                previous_site = site
            try:
                next_rank, next_position, next_chrom = next(iterator)
            except StopIteration:
                continue
            heapq.heappush(
                heap,
                (next_rank, next_position, file_index, next_chrom, iterator),
            )
    return unique_sites, chromosome_observed, window_observed


def percent(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else math.nan


def build_genomic_rows(
    context: str,
    chromosomes: list[str],
    chromosome_reference: dict[str, dict[str, int]],
    window_reference: dict[int, dict[tuple[str, str, int], int]],
    chromosome_observed: dict[str, int],
    window_observed: dict[int, dict[tuple[str, int], int]],
) -> tuple[list[tuple[str, float]], dict[int, list[tuple[str, float]]]]:
    chromosome_rows = []
    for chrom in chromosomes:
        observed = chromosome_observed[chrom]
        reference = chromosome_reference[chrom][context]
        if observed > reference:
            raise ValueError(
                f"Observed {context} sites exceed reference sites on {chrom}: "
                f"{observed:,} > {reference:,}"
            )
        chromosome_rows.append((chrom, percent(observed, reference)))

    window_rows: dict[int, list[tuple[str, float]]] = {}
    for size in WINDOW_SIZES:
        for (chrom, window_index), observed in window_observed[size].items():
            reference = window_reference[size][
                (chrom, context, window_index)
            ]
            if observed > reference:
                raise ValueError(
                    f"Observed {context} sites exceed reference sites in "
                    f"{chrom}:{window_index * size}-{(window_index + 1) * size}"
                )
        rows = []
        for chrom in chromosomes:
            keys = sorted(
                (
                    key
                    for key in window_reference[size]
                    if key[0] == chrom
                    and key[1] == context
                    and window_reference[size][key]
                ),
                key=lambda key: key[2],
            )
            for _, _, window_index in keys:
                observed = window_observed[size][(chrom, window_index)]
                reference = window_reference[size][
                    (chrom, context, window_index)
                ]
                rows.append((chrom, percent(observed, reference)))
        if not rows:
            raise ValueError(
                f"Reference contains no {context} sites in {size} bp windows"
            )
        window_rows[size] = rows
    return chromosome_rows, window_rows


def context_label(context: str) -> str:
    return "CpG" if context == "CG" else context


def plot_spatial_coverage(
    path: Path,
    context: str,
    records: list[SpotRecord],
    reference_sites: int,
) -> None:
    max_x = max(record.x_index for record in records)
    max_y = max(record.y_index for record in records)
    matrix = np.zeros((max_y + 1, max_x + 1), dtype=float)
    for record in records:
        if record.retained:
            matrix[record.y_index, record.x_index] = percent(
                record.observed_sites, reference_sites
            )

    label = context_label(context)
    with plt.rc_context({"font.family": "sans-serif", "font.size": 10}):
        figure, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(
            matrix,
            origin="lower",
            interpolation="nearest",
            aspect="equal",
            cmap="magma",
            vmin=0,
            extent=(-0.5, max_x + 0.5, -0.5, max_y + 0.5),
        )
        axis.set_title(f"Spatial {label} coverage (retained spots)", fontsize=12)
        axis.set_xlabel("X index", fontsize=10)
        axis.set_ylabel("Y index", fontsize=10)
        colorbar = figure.colorbar(image, ax=axis, shrink=0.9)
        colorbar.set_label(f"Reference {label} sites covered per spot (%)")
        figure.tight_layout()
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)


def plot_genomic_coverage(
    path: Path,
    context: str,
    chromosome_rows: list[tuple[str, float]],
    window_rows: dict[int, list[tuple[str, float]]],
) -> None:
    label = context_label(context)
    with plt.rc_context({"font.family": "sans-serif", "font.size": 9}):
        figure, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
        chromosome_labels = [chrom for chrom, _ in chromosome_rows]
        values = [coverage for _, coverage in chromosome_rows]
        axes[0].bar(
            range(len(chromosome_labels)), values, color=PLOT_COLOR, width=0.75
        )
        axes[0].set_xticks(
            range(len(chromosome_labels)),
            chromosome_labels,
            rotation=45,
            ha="right",
        )
        axes[0].set_ylabel("Coverage (%)", fontsize=10)
        axes[0].set_title(f"Unique {label} coverage by chromosome", fontsize=12)
        axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axes[0].set_axisbelow(True)

        for axis, size in zip(axes[1:], WINDOW_SIZES):
            rows = window_rows[size]
            x_values = list(range(len(rows)))
            axis.plot(
                x_values,
                [coverage for _, coverage in rows],
                color=PLOT_COLOR,
                linewidth=0.6,
            )
            tick_positions = []
            tick_labels = []
            start = 0
            previous_chrom = rows[0][0]
            for index, (chrom, _) in enumerate(rows + [("", math.nan)]):
                if chrom != previous_chrom:
                    tick_positions.append((start + index - 1) / 2)
                    tick_labels.append(previous_chrom)
                    if index < len(rows):
                        axis.axvline(index - 0.5, color="#D9D9D9", linewidth=0.5)
                    start = index
                    previous_chrom = chrom
            axis.set_xticks(tick_positions, tick_labels, rotation=45, ha="right")
            axis.set_xlim(-0.5, max(0.5, len(rows) - 0.5))
            axis.set_ylim(bottom=0)
            axis.set_ylabel("Coverage (%)", fontsize=10)
            axis.set_title(
                f"Unique {label} coverage in {size // 1_000:,} kb windows",
                fontsize=12,
            )
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
            axis.set_axisbelow(True)
        axes[-1].set_xlabel("Genomic windows ordered by chromosome", fontsize=10)
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)


def build_spot_coverage_rows(
    context: str,
    records: list[SpotRecord],
    reference_sites: int,
) -> list[dict[str, object]]:
    rows = []
    for record in sorted(records, key=lambda item: (item.x_index, item.y_index)):
        coverage_fraction = record.observed_sites / reference_sites
        rows.append(
            {
                "context": context,
                "spot": record.spot,
                "X_index": record.x_index,
                "Y_index": record.y_index,
                "observed_sites": record.observed_sites,
                "reference_sites": reference_sites,
                "coverage_fraction": f"{coverage_fraction:.12f}",
                "coverage_percent": f"{coverage_fraction * 100:.9f}",
                "retained": record.retained,
            }
        )
    return rows


def build_summary_row(
    context: str,
    records: list[SpotRecord],
    minimum_total_sites: int,
    chromosomes: list[str],
    reference_sites: int,
    unique_sites: int,
) -> dict[str, object]:
    retained = [record for record in records if record.retained]
    retained_coverage = [
        percent(record.observed_sites, reference_sites) for record in retained
    ]
    coverage_fraction = unique_sites / reference_sites
    return {
        "context": context,
        "coverage_files": len(records),
        "retained_spots": len(retained),
        "minimum_total_sites": minimum_total_sites,
        "chromosomes": ",".join(chromosomes),
        "reference_sites": reference_sites,
        "unique_sites_covered": unique_sites,
        "pooled_coverage_fraction": f"{coverage_fraction:.9f}",
        "pooled_coverage_percent": f"{coverage_fraction * 100:.6f}",
        "retained_spot_mean_coverage_percent": (
            f"{statistics.fmean(retained_coverage):.9f}"
        ),
        "retained_spot_median_coverage_percent": (
            f"{statistics.median(retained_coverage):.9f}"
        ),
    }


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open(mode="w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    host_dir = args.host_dir.expanduser().resolve()
    reference_path = args.reference_fasta.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not host_dir.is_dir():
        raise SystemExit(f"Host directory does not exist: {host_dir}")
    if not reference_path.is_file():
        raise SystemExit(f"Reference FASTA does not exist: {reference_path}")

    try:
        _, chromosome_reference, window_reference = scan_reference(
            reference_path, args.chromosomes
        )
    except (OSError, EOFError, UnicodeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    chromosome_ranks = {
        chrom: rank for rank, chrom in enumerate(args.chromosomes)
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    spot_rows: list[dict[str, object]] = []
    for context in CONTEXTS:
        try:
            records = read_spot_records(
                host_dir,
                context,
                chromosome_ranks,
                args.min_total_sites,
            )
            retained = [record for record in records if record.retained]
            if not retained:
                raise ValueError(
                    f"No {context} spots passed --min-total-sites"
                )
            unique_sites, chromosome_observed, window_observed = (
                merge_observed_sites(records, context, chromosome_ranks)
            )
            chromosome_rows, window_rows = build_genomic_rows(
                context,
                args.chromosomes,
                chromosome_reference,
                window_reference,
                chromosome_observed,
                window_observed,
            )
            reference_sites = sum(
                chromosome_reference[chrom][context] for chrom in args.chromosomes
            )
            plot_genomic_coverage(
                output_dir / f"{context}_genomic_coverage.png",
                context,
                chromosome_rows,
                window_rows,
            )
            plot_spatial_coverage(
                output_dir / f"{context}_spatial_coverage_heatmap.png",
                context,
                records,
                reference_sites,
            )
            spot_rows.extend(
                build_spot_coverage_rows(context, records, reference_sites)
            )
            summary_rows.append(
                build_summary_row(
                    context,
                    records,
                    args.min_total_sites,
                    args.chromosomes,
                    reference_sites,
                    unique_sites,
                )
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise SystemExit(str(error)) from error

        print(f"{context} coverage files: {len(records):,}")
        print(f"{context} spots retained: {len(retained):,}")
        print(f"{context} reference sites: {reference_sites:,}")
        print(f"{context} unique sites covered: {unique_sites:,}")
        print(
            f"{context} pooled coverage: "
            f"{percent(unique_sites, reference_sites):.6f}%"
        )
    try:
        write_tsv(output_dir / "context_coverage_by_spot.tsv", spot_rows)
        write_tsv(output_dir / "context_coverage_summary.tsv", summary_rows)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
