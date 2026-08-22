#!/usr/bin/env python3
"""Summarize per-spot mCG across mm10 ChromHMM states."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import math
import statistics
import struct
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
STATE_DESCRIPTIONS = {
    "Enh": "Active enhancer",
    "EnhG": "Genic enhancer",
    "EnhLo": "Low-signal enhancer",
    "EnhPois": "Poised enhancer",
    "EnhPr": "Primed enhancer",
    "Het": "Constitutive heterochromatin",
    "Quies": "Quiescent or low-signal chromatin",
    "Quies2": "Quiescent substate; retained in tables but omitted from the reference figure",
    "Quies3": "Quiescent substate",
    "Quies4": "Quiescent substate",
    "QuiesG": "Genic quiescent state",
    "ReprPC": "Polycomb-repressed state",
    "ReprPCWk": "Weak Polycomb-repressed state",
    "Tss": "Active transcription start site",
    "TssBiv": "Bivalent transcription start site",
    "TssFlnk": "Transcription-start-site flanking state",
    "Tx": "Strong transcription",
    "TxWk": "Weak transcription",
    "NA": "No ChromHMM state",
}
BOX_COLOR = "#56B4E9"


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
            "Aggregate per-spot CG coverage over mm10 ChromHMM states and draw a "
            "boxplot matching Extended Data Fig. 4b of Deng et al."
        )
    )
    parser.add_argument(
        "--host-dir",
        type=Path,
        required=True,
        help="Directory recursively containing <spot>.CG.cov files.",
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
        help="Minimum matched CpGs for a spot-state value in the plot (default: 10).",
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


def sample_name(path: Path) -> str:
    suffix = ".CG.cov"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected CG coverage filename: {path.name}")
    return path.name[: -len(suffix)]


def discover_files(host_dir: Path) -> list[Path]:
    paths = sorted(host_dir.rglob("*.CG.cov"))
    return paths


def process_coverage_file(
    path: Path, chromhmm_index: ChromHMMIndex
) -> tuple[dict[str, SiteStats], int, int]:
    stats_by_state: dict[str, SiteStats] = defaultdict(SiteStats)
    matched_sites = 0
    unmatched_sites = 0
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                raise ValueError(
                    f"Expected at least 4 coverage columns at {path}:{line_number}"
                )
            chrom = fields[0]
            cpg_start = int(fields[1])
            methylation_percent = float(fields[3])
            if (
                cpg_start < 0
                or not math.isfinite(methylation_percent)
                or not 0 <= methylation_percent <= 100
            ):
                raise ValueError(f"Invalid coverage values at {path}:{line_number}")
            state = chromhmm_index.label_at(chrom, cpg_start)
            if state is None:
                unmatched_sites += 1
                continue
            stats_by_state[state].add(methylation_percent / 100)
            matched_sites += 1
    return stats_by_state, matched_sites, unmatched_sites


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def summarize_rows(
    detail_rows: list[dict[str, object]], min_sites: int, state_order: list[str]
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        grouped[str(row["state"])].append(row)

    summary: list[dict[str, object]] = []
    for state in state_order:
        rows = grouped.get(state, [])
        passing = [row for row in rows if int(row["site_count"]) >= min_sites]
        values = [float(row["methylation_rate"]) for row in passing]
        site_count = sum(int(row["site_count"]) for row in rows)
        site_methylation_sum = sum(
            float(row["methylation_rate"]) * int(row["site_count"])
            for row in rows
        )
        summary.append(
            {
                "context": "CG",
                "state": state,
                "plotted_in_reference_order": state in PLOT_STATE_ORDER,
                "spots_observed": len(rows),
                "spots_passing_min_sites": len(passing),
                "spot_site_observations": site_count,
                "pooled_methylation_rate": (
                    site_methylation_sum / site_count if site_count else ""
                ),
                "spot_mean": statistics.fmean(values) if values else "",
                "spot_q1": percentile(values, 0.25) if values else "",
                "spot_median": statistics.median(values) if values else "",
                "spot_q3": percentile(values, 0.75) if values else "",
            }
        )
    return summary


def write_detail(path: Path, rows: Iterable[dict[str, object]]) -> None:
    fields = [
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
) -> None:
    values: dict[str, list[float]] = defaultdict(list)
    for row in detail_rows:
        if int(row["site_count"]) >= min_sites:
            values[str(row["state"])].append(float(row["methylation_rate"]))

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.edgecolor": "#666666",
            "axes.linewidth": 0.8,
            "xtick.color": "#2B2B2B",
            "ytick.color": "#2B2B2B",
        }
    ):
        figure, axis = plt.subplots(figsize=(10, 4))
        for position, state in enumerate(PLOT_STATE_ORDER, start=1):
            observations = values.get(state, [])
            if not observations:
                continue
            axis.boxplot(
                observations,
                positions=[position],
                widths=0.48,
                whis=1.5,
                showfliers=False,
                patch_artist=True,
                medianprops={"color": "#1F1F1F", "linewidth": 1.1},
                whiskerprops={"color": BOX_COLOR, "linewidth": 1.0},
                capprops={"color": BOX_COLOR, "linewidth": 1.0},
                boxprops={
                    "facecolor": "white",
                    "edgecolor": BOX_COLOR,
                    "linewidth": 1.3,
                },
            )

        axis.set_title("DNA methylation under different chromatin states", fontsize=12, pad=8)
        axis.set_ylabel("Methylation Levels", fontsize=10)
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
        figure.tight_layout()
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    args = parse_args()
    host_dir = args.host_dir.expanduser().resolve()
    chromhmm_cm = args.chromhmm_cm.expanduser().resolve()
    cpg_reference = args.cpg_reference.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path, description in (
        (host_dir, "host directory"),
        (chromhmm_cm, "ChromHMM mask"),
        (cpg_reference, "CpG reference"),
    ):
        if not path.exists():
            raise SystemExit(f"{description} does not exist: {path}")

    try:
        chromhmm_index = read_chromhmm_index(chromhmm_cm, cpg_reference)
    except (OSError, EOFError, UnicodeError, ValueError, struct.error) as error:
        raise SystemExit(str(error)) from error
    print(
        f"Loaded {len(chromhmm_index.state_codes):,} mm10 CpGs "
        f"and {len(chromhmm_index.labels)} ChromHMM labels",
        flush=True,
    )

    paths = discover_files(host_dir)
    if not paths:
        raise SystemExit(f"No *.CG.cov files found below {host_dir}")

    detail_rows: list[dict[str, object]] = []
    files_processed = 0
    spots_retained = 0
    matched_sites_total = 0
    unmatched_sites_total = 0
    for path in paths:
        try:
            stats_by_state, matched_sites, unmatched_sites = process_coverage_file(
                path, chromhmm_index
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise SystemExit(str(error)) from error
        files_processed += 1
        matched_sites_total += matched_sites
        unmatched_sites_total += unmatched_sites
        if matched_sites >= args.min_total_sites:
            spot = sample_name(path)
            for state, stats in stats_by_state.items():
                detail_rows.append(
                    {
                        "spot": spot,
                        "context": "CG",
                        "state": state,
                        "site_count": stats.sites,
                        "methylation_rate": stats.rate,
                    }
                )
            spots_retained += 1
        if files_processed % 100 == 0:
            print(f"Processed {files_processed:,} CG coverage files", flush=True)

    if not detail_rows:
        raise SystemExit("No spots passed --min-total-sites")

    extra_states = [
        state for state in chromhmm_index.labels if state not in PLOT_STATE_ORDER
    ]
    summary_order = list(PLOT_STATE_ORDER) + extra_states
    summary_rows = summarize_rows(detail_rows, args.min_sites, summary_order)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "chromhmm_methylation_by_spot.tsv.gz"
    summary_path = output_dir / "chromhmm_methylation_summary.tsv"
    figure_path = output_dir / "chromhmm_methylation_boxplots.png"
    definition_path = output_dir / "chromhmm_state_definitions.tsv"

    write_detail(detail_path, detail_rows)
    write_tsv(summary_path, summary_rows)
    write_tsv(
        definition_path,
        [
            {
                "state": state,
                "description": STATE_DESCRIPTIONS.get(state, "ChromHMM categorical state"),
                "plotted_in_reference_order": state in PLOT_STATE_ORDER,
            }
            for state in summary_order
        ],
    )
    plot_boxplots(figure_path, detail_rows, args.min_sites)

    print("Context: CG")
    print(f"Coverage files: {files_processed:,}")
    print(f"Spots retained: {spots_retained:,}")
    print(f"Matched CG sites: {matched_sites_total:,}")
    print(f"Unmatched CG sites: {unmatched_sites_total:,}")
    print(f"Figure: {figure_path}")
    print(f"Summary: {summary_path}")
    print(f"Per-spot table: {detail_path}")
    print(f"Definitions: {definition_path}")


if __name__ == "__main__":
    main()
