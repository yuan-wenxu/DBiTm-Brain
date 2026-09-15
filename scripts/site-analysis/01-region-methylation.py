#!/usr/bin/env python3
"""Summarize per-spot mCG across mm10 ChromHMM states."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import math
import struct
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


YAME_SIGNATURE = 266563789635
PLOT_STATE_ORDER = (
    "Enh",
    "EnhG",
    "EnhLo",
    "EnhPois",
    "EnhPr",
    "Het",
    "Quies",
    "Quies3",
    "Quies4",
    "QuiesG",
    "ReprPC",
    "ReprPCWk",
    "Tss",
    "TssBiv",
    "TssFlnk",
    "Tx",
    "TxWk",
)
# Keep these sample colors aligned with scripts/benchmark/saturation.py.
SAMPLE_COLORS = {
    "P35-TAPS-50μm": "#A73030",
    "P35-TAPS-beta-50μm": "#E64B35",
    "SRR29496780-20μm": "#2F5597",
    "SRR29496782-50μm": "#3C5488",
    "SRR29496784-50μm": "#4DBBD5",
    "SRR32867346-10μm": "#70B7D2",
    "SRR32867348-50μm": "#9DCBE1",
}


@dataclass(frozen=True)
class ChromHMMIndex:
    """CpG starts by chromosome plus aligned categorical state codes."""

    positions: dict[str, array]
    row_offsets: dict[str, int]
    state_codes: bytearray
    labels: tuple[str, ...]

    def label_at(self, chrom: str, cpg_start: int) -> str | None:
        chrom_positions = self.positions.get(chrom)
        if chrom_positions is None:
            return None
        index = bisect.bisect_left(chrom_positions, cpg_start)
        if index == len(chrom_positions) or chrom_positions[index] != cpg_start:
            return None
        code = self.state_codes[self.row_offsets[chrom] + index]
        return self.labels[code]


@dataclass
class SiteStats:
    sites: int = 0
    methylation_fraction_sum: float = 0.0

    @property
    def rate(self) -> float:
        """Arithmetic mean of per-site methylation fractions."""
        return self.methylation_fraction_sum / self.sites

    def add(self, methylation_fraction: float) -> None:
        self.sites += 1
        self.methylation_fraction_sum += methylation_fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate one or more samples of per-spot CG coverage over mm10 "
            "ChromHMM states and draw a combined boxplot."
        )
    )
    parser.add_argument(
        "sample_dirs",
        type=Path,
        nargs="+",
        help=(
            "Sample result directories that each contain "
            "dbitm/coverage/host/host.CG.cov."
        ),
    )
    parser.add_argument(
        "--chromhmm-cm",
        type=Path,
        required=True,
        help="mm10 YAME categorical mask, normally ChromHMM.20220414.cm.",
    )
    parser.add_argument(
        "--cpg-reference",
        type=Path,
        required=True,
        help="Matching mm10 YAME CpG coordinates, normally cpg_nocontig.cr.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-sites",
        type=int,
        default=10,
        help=(
            "Minimum matched CpGs for a spot-state value in the plot and summary "
            "(default: 10)."
        ),
    )
    parser.add_argument(
        "--min-total-sites",
        type=int,
        default=100,
        help="Minimum matched CG sites required to retain a spot (default: 100).",
    )
    args = parser.parse_args()
    if args.min_sites < 1:
        parser.error("--min-sites must be at least 1")
    if args.min_total_sites < 0:
        parser.error("--min-total-sites cannot be negative")
    return args


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="r", encoding="utf-8")


def read_yame_payload(path: Path, expected_format: str) -> bytes:
    """Read one BGZF/YAME record and return its encoded payload."""
    with gzip.open(path, mode="rb") as handle:
        header = handle.read(17)
        if len(header) != 17:
            raise ValueError(f"Truncated YAME header: {path}")
        signature, format_byte, payload_size = struct.unpack("<QcQ", header)
        if signature != YAME_SIGNATURE:
            raise ValueError(f"Invalid YAME signature: {path}")
        observed_format = format_byte.decode("ascii")
        if observed_format != expected_format:
            raise ValueError(
                f"Expected YAME format {expected_format}, found {observed_format}: {path}"
            )
        payload = handle.read()
    if len(payload) != payload_size:
        raise ValueError(
            f"YAME payload length mismatch in {path}: "
            f"expected {payload_size:,}, found {len(payload):,}"
        )
    return payload


def decode_state_mask(path: Path) -> tuple[tuple[str, ...], bytearray]:
    """Expand a format-2 categorical mask to one compact byte per CpG."""
    payload = read_yame_payload(path, "2")
    key_boundary = payload.find(b"\0\0")
    if key_boundary < 0:
        raise ValueError(f"Missing categorical key terminator: {path}")
    try:
        labels = tuple(payload[:key_boundary].decode("utf-8").split("\0"))
    except UnicodeDecodeError as error:
        raise ValueError(f"Invalid categorical state labels: {path}") from error
    offset = key_boundary + 2
    if offset >= len(payload):
        raise ValueError(f"Missing categorical state records: {path}")
    unit = payload[offset]
    offset += 1
    if unit not in (1, 2, 3, 8):
        raise ValueError(f"Unsupported YAME categorical unit width {unit}: {path}")
    record_size = unit + 2
    if (len(payload) - offset) % record_size:
        raise ValueError(f"Truncated categorical run in {path}")

    state_codes = bytearray()
    for record_start in range(offset, len(payload), record_size):
        code = int.from_bytes(
            payload[record_start : record_start + unit], byteorder="little"
        )
        run_length = int.from_bytes(
            payload[record_start + unit : record_start + record_size],
            byteorder="little",
        )
        if code >= len(labels) or code > 255 or run_length == 0:
            raise ValueError(f"Invalid categorical run in {path}")
        state_codes.extend(bytes((code,)) * run_length)
    return labels, state_codes


def decode_cpg_reference(path: Path) -> tuple[dict[str, array], dict[str, int], int]:
    """Decode format-7 mm10 CpG coordinates into compact chromosome arrays."""
    payload = read_yame_payload(path, "7")
    positions: dict[str, array] = {}
    row_offsets: dict[str, int] = {}
    offset = 0
    row_count = 0
    while offset < len(payload):
        chromosome_end = payload.find(b"\0", offset)
        if chromosome_end < 0:
            raise ValueError(f"Truncated chromosome name in {path}")
        try:
            chrom = payload[offset:chromosome_end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Invalid chromosome name in {path}") from error
        if not chrom or chrom in positions:
            raise ValueError(f"Invalid or duplicate chromosome {chrom!r} in {path}")
        offset = chromosome_end + 1
        chrom_positions = array("I")
        positions[chrom] = chrom_positions
        row_offsets[chrom] = row_count
        coordinate = 0

        while offset < len(payload) and payload[offset] != 0xFF:
            lead = payload[offset]
            if lead < 0x80:
                delta = lead
                offset += 1
            elif lead < 0xC0:
                if offset + 2 > len(payload):
                    raise ValueError(f"Truncated two-byte coordinate in {path}")
                delta = ((lead & 0x3F) << 8) | payload[offset + 1]
                offset += 2
            else:
                if offset + 8 > len(payload):
                    raise ValueError(f"Truncated eight-byte coordinate in {path}")
                delta = int.from_bytes(
                    payload[offset : offset + 8], byteorder="big"
                ) & ((1 << 62) - 1)
                offset += 8
            if delta <= 0:
                raise ValueError(f"Non-increasing CpG coordinate in {path}")
            coordinate += delta
            # YAME stores the 1-based cytosine coordinate; normalize it to the
            # zero-based cytosine start used by the host coverage files.
            cpg_start = coordinate - 1
            if cpg_start < 0 or cpg_start > 0xFFFFFFFF:
                raise ValueError(f"CpG coordinate outside uint32 range in {path}")
            chrom_positions.append(cpg_start)
            row_count += 1

        if offset < len(payload):
            if payload[offset] != 0xFF:
                raise ValueError(f"Invalid chromosome terminator in {path}")
            offset += 1
    return positions, row_offsets, row_count


def read_chromhmm_index(cm_path: Path, cpg_path: Path) -> ChromHMMIndex:
    labels, state_codes = decode_state_mask(cm_path)
    positions, row_offsets, coordinate_count = decode_cpg_reference(cpg_path)
    if coordinate_count != len(state_codes):
        raise ValueError(
            "ChromHMM mask and CpG reference row counts differ: "
            f"{len(state_codes):,} states versus {coordinate_count:,} coordinates"
        )
    missing_states = [state for state in PLOT_STATE_ORDER if state not in labels]
    if missing_states:
        raise ValueError(
            "ChromHMM mask lacks reference-figure states: " + ", ".join(missing_states)
        )
    return ChromHMMIndex(positions, row_offsets, state_codes, labels)


def resolve_sample_input(input_dir: Path) -> tuple[str, Path]:
    """Resolve a sample root and its fixed DBiTM host-CG coverage path."""
    sample_dir = input_dir.expanduser().resolve()
    if not sample_dir.is_dir():
        raise ValueError(f"Sample directory does not exist: {sample_dir}")

    coverage_path = sample_dir / "dbitm" / "coverage" / "host" / "host.CG.cov"
    if not coverage_path.is_file():
        raise ValueError(f"Host CG coverage file does not exist: {coverage_path}")
    return sample_dir.name, coverage_path


def process_coverage_file(
    path: Path, chromhmm_index: ChromHMMIndex
) -> tuple[
    dict[str, dict[str, SiteStats]],
    dict[str, int],
]:
    """Aggregate coverage rows by the spot name stored in column seven."""
    stats_by_spot: dict[str, dict[str, SiteStats]] = {}
    matched_sites_by_spot: dict[str, int] = defaultdict(int)
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 7:
                raise ValueError(
                    f"Expected at least 7 coverage columns at {path}:{line_number}"
                )
            chrom = fields[0]
            cpg_start = int(fields[1])
            methylation_percent = float(fields[3])
            spot = fields[6]
            if (
                cpg_start < 0
                or not math.isfinite(methylation_percent)
                or not 0 <= methylation_percent <= 100
                or not spot
            ):
                raise ValueError(f"Invalid coverage values at {path}:{line_number}")

            matched_sites_by_spot[spot] += 0
            state = chromhmm_index.label_at(chrom, cpg_start)
            if state is None:
                continue
            spot_stats = stats_by_spot.setdefault(spot, {})
            spot_stats.setdefault(state, SiteStats()).add(methylation_percent / 100)
            matched_sites_by_spot[spot] += 1
    return stats_by_spot, dict(matched_sites_by_spot)


def summarize_rows(
    detail_rows: list[dict[str, object]],
    min_sites: int,
    sample_order: list[str],
    state_order: list[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        grouped[(str(row["sample"]), str(row["state"]))].append(row)

    summary: list[dict[str, object]] = []
    for sample in sample_order:
        for state in state_order:
            rows = grouped.get((sample, state), [])
            passing = [row for row in rows if int(row["site_count"]) >= min_sites]
            site_count = sum(int(row["site_count"]) for row in passing)
            site_methylation_sum = sum(
                float(row["methylation_rate"]) * int(row["site_count"])
                for row in passing
            )
            summary.append(
                {
                    "sample": sample,
                    "context": "CG",
                    "state": state,
                    "plotted": state in PLOT_STATE_ORDER,
                    "total_spots": len(rows),
                    "spots_passing_filter": len(passing),
                    "total_sites": site_count,
                    "pooled_methylation_rate": (
                        site_methylation_sum / site_count if site_count else ""
                    ),
                }
            )
    return summary


def write_detail(path: Path, rows: Iterable[dict[str, object]]) -> None:
    fields = [
        "sample",
        "spot",
        "context",
        "state",
        "site_count",
        "methylation_rate",
    ]
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open(mode="w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_boxplots(
    path: Path,
    detail_rows: list[dict[str, object]],
    min_sites: int,
    sample_order: list[str],
) -> None:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in detail_rows:
        if int(row["site_count"]) >= min_sites:
            values[(str(row["sample"]), str(row["state"]))].append(
                float(row["methylation_rate"])
            )

    box_width = min(0.72 / len(sample_order), 0.48)
    offsets = [
        (index - (len(sample_order) - 1) / 2) * box_width
        for index in range(len(sample_order))
    ]

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.edgecolor": "#666666",
            "axes.linewidth": 0.8,
            "xtick.color": "#2B2B2B",
            "ytick.color": "#2B2B2B",
        }
    ):
        figure, axis = plt.subplots(figsize=(12, 5))
        for sample_index, sample in enumerate(sample_order):
            color = SAMPLE_COLORS[sample]
            for state_position, state in enumerate(PLOT_STATE_ORDER, start=1):
                observations = values.get((sample, state), [])
                if not observations:
                    continue
                axis.boxplot(
                    observations,
                    positions=[state_position + offsets[sample_index]],
                    widths=box_width * 0.88,
                    whis=1.5,
                    showfliers=False,
                    patch_artist=True,
                    medianprops={"color": "#1F1F1F", "linewidth": 0.8},
                    whiskerprops={"color": color, "linewidth": 0.8},
                    capprops={"color": color, "linewidth": 0.8},
                    boxprops={
                        "facecolor": color,
                        "edgecolor": color,
                        "linewidth": 0.9,
                    },
                )

        axis.set_title(
            "DNA methylation under different chromatin states", fontsize=12, pad=8
        )
        axis.set_ylabel("Methylation Levels", fontsize=12)
        axis.set_ylim(-0.05, 1.03)
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.set_xticks(
            range(1, len(PLOT_STATE_ORDER) + 1),
            PLOT_STATE_ORDER,
            rotation=48,
            ha="right",
            rotation_mode="anchor",
        )
        axis.grid(axis="both", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.legend(
            handles=[
                Patch(
                    facecolor=SAMPLE_COLORS[sample],
                    edgecolor=SAMPLE_COLORS[sample],
                    label=sample,
                )
                for sample in sample_order
            ],
            title="Sample",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=True,
            fontsize=8,
            title_fontsize=8,
        )
        figure.tight_layout()
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)


def plot_spot_filtering(
    path: Path,
    matched_sites_by_sample: dict[str, list[tuple[str, int]]],
    min_total_sites: int,
    sample_order: list[str],
) -> None:
    """Plot per-sample matched-CG ranks used by the spot coverage filter."""
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.edgecolor": "#666666",
            "axes.linewidth": 0.8,
            "xtick.color": "#2B2B2B",
            "ytick.color": "#2B2B2B",
        }
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        maximum_spot_count = 0
        for sample in sample_order:
            ordered = sorted(
                matched_sites_by_sample[sample],
                key=lambda item: (-item[1], item[0]),
            )
            ranks = list(range(1, len(ordered) + 1))
            site_counts = [site_count for _, site_count in ordered]
            retained_count = sum(
                site_count >= min_total_sites for site_count in site_counts
            )
            maximum_spot_count = max(maximum_spot_count, len(ordered))
            axis.scatter(
                ranks,
                site_counts,
                color=SAMPLE_COLORS[sample],
                s=8,
                alpha=0.75,
                linewidths=0,
                label=f"{sample} ({retained_count:,}/{len(ordered):,} retained)",
            )
        axis.axhline(
            min_total_sites,
            color="#1F1F1F",
            linestyle="--",
            linewidth=1.2,
            label=f"Threshold = {min_total_sites:,}",
        )
        axis.set_yscale("symlog", linthresh=1)
        axis.set_ylim(bottom=0)
        axis.set_title("Spot filtering by matched CG coverage", fontsize=12, pad=8)
        axis.set_xlabel("Spot rank (highest coverage first)", fontsize=10)
        axis.set_ylabel("Matched CG sites", fontsize=10)
        axis.set_xlim(0.5, maximum_spot_count + 0.5)
        axis.grid(axis="both", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=8,
        )
        figure.tight_layout()
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    args = parse_args()
    chromhmm_cm = args.chromhmm_cm.expanduser().resolve()
    cpg_reference = args.cpg_reference.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path, description in (
        (chromhmm_cm, "ChromHMM mask"),
        (cpg_reference, "CpG reference"),
    ):
        if not path.is_file():
            raise SystemExit(f"{description} does not exist: {path}")

    try:
        sample_inputs = [resolve_sample_input(path) for path in args.sample_dirs]
    except ValueError as error:
        raise SystemExit(str(error)) from error
    sample_order = [sample for sample, _ in sample_inputs]
    duplicate_samples = sorted(
        sample for sample in set(sample_order) if sample_order.count(sample) > 1
    )
    if duplicate_samples:
        raise SystemExit("Duplicate sample names: " + ", ".join(duplicate_samples))
    missing_colors = [sample for sample in sample_order if sample not in SAMPLE_COLORS]
    if missing_colors:
        raise SystemExit("SAMPLE_COLORS lacks entries for: " + ", ".join(missing_colors))

    try:
        chromhmm_index = read_chromhmm_index(chromhmm_cm, cpg_reference)
    except (OSError, EOFError, UnicodeError, ValueError, struct.error) as error:
        raise SystemExit(str(error)) from error
    print(
        f"Loaded {len(chromhmm_index.state_codes):,} mm10 CpGs "
        f"and {len(chromhmm_index.labels)} ChromHMM labels",
        flush=True,
    )

    detail_rows: list[dict[str, object]] = []
    files_processed = 0
    matched_sites_by_sample: dict[str, list[tuple[str, int]]] = {
        sample: [] for sample in sample_order
    }
    for sample, coverage_path in sample_inputs:
        try:
            (
                stats_by_spot,
                sample_matched_by_spot,
            ) = process_coverage_file(coverage_path, chromhmm_index)
        except (OSError, UnicodeError, ValueError) as error:
            raise SystemExit(str(error)) from error
        files_processed += 1

        for spot in sorted(sample_matched_by_spot):
            matched_sites = sample_matched_by_spot[spot]
            matched_sites_by_sample[sample].append((spot, matched_sites))
            if matched_sites < args.min_total_sites:
                continue
            for state in chromhmm_index.labels:
                stats = stats_by_spot.get(spot, {}).get(state)
                if stats is None:
                    continue
                detail_rows.append(
                    {
                        "sample": sample,
                        "spot": spot,
                        "context": "CG",
                        "state": state,
                        "site_count": stats.sites,
                        "methylation_rate": stats.rate,
                    }
                )

        print(
            f"Sample={sample} spots={len(sample_matched_by_spot):,} ",
            flush=True,
        )

    if not detail_rows:
        raise SystemExit("No spots passed --min-total-sites")

    extra_states = [
        state for state in chromhmm_index.labels if state not in PLOT_STATE_ORDER
    ]
    summary_order = list(PLOT_STATE_ORDER) + extra_states
    summary_rows = summarize_rows(
        detail_rows, args.min_sites, sample_order, summary_order
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "chromhmm_methylation_by_spot.tsv.gz"
    summary_path = output_dir / "chromhmm_methylation_summary.tsv"
    figure_path = output_dir / "chromhmm_methylation_boxplots.png"
    filter_figure_path = output_dir / "spot_filtering_rank.png"

    write_detail(detail_path, detail_rows)
    write_tsv(summary_path, summary_rows)
    plot_boxplots(figure_path, detail_rows, args.min_sites, sample_order)
    plot_spot_filtering(
        filter_figure_path,
        matched_sites_by_sample,
        args.min_total_sites,
        sample_order,
    )

    print("Context: CG")
    print(f"Coverage files: {files_processed:,}")
    print(f"Figure: {figure_path}")
    print(f"Filter figure: {filter_figure_path}")
    print(f"Summary: {summary_path}")
    print(f"Per-spot table: {detail_path}")


if __name__ == "__main__":
    main()
