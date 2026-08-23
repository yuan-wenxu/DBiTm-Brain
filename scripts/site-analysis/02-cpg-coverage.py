#!/usr/bin/env python3
"""Quantify spatial CpG coverage against the mm10 CpG reference."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import math
import statistics
import struct
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


YAME_SIGNATURE = 266563789635
WINDOW_SIZES = (100_000, 1_000_000)
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
PLOT_COLOR = "#56B4E9"


@dataclass(frozen=True)
class CpGAnnotationIndex:
    """Reference CpG coordinates and aligned ChromHMM state codes."""

    positions: dict[str, array]
    row_offsets: dict[str, int]
    state_codes: bytearray
    labels: tuple[str, ...]

    @property
    def total_sites(self) -> int:
        return len(self.state_codes)

    def row_at(self, chrom: str, cpg_start: int) -> int | None:
        chrom_positions = self.positions.get(chrom)
        if chrom_positions is None:
            return None
        local_index = bisect.bisect_left(chrom_positions, cpg_start)
        if (
            local_index == len(chrom_positions)
            or chrom_positions[local_index] != cpg_start
        ):
            return None
        return self.row_offsets[chrom] + local_index


@dataclass(frozen=True)
class SpotRecord:
    path: Path
    spot: str
    x_index: int
    y_index: int
    matched_sites: int
    unmatched_rows: int
    retained: bool
    state_counts: dict[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate per-spot, cumulative, genomic-window, spatial, and "
            "ChromHMM-normalized CpG coverage."
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
        "--min-total-sites",
        type=int,
        default=100,
        help="Minimum matched reference CpGs required to retain a spot (default: 100).",
    )
    args = parser.parse_args()
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
                f"Expected YAME format {expected_format}, found "
                f"{observed_format}: {path}"
            )
        payload = handle.read()
    if len(payload) != payload_size:
        raise ValueError(
            f"YAME payload length mismatch in {path}: expected "
            f"{payload_size:,}, found {len(payload):,}"
        )
    return payload


def decode_state_mask(path: Path) -> tuple[tuple[str, ...], bytearray]:
    """Expand a format-2 categorical mask to one byte per reference CpG."""
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
    """Decode format-7 CpG coordinates into compact chromosome arrays."""
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
            cpg_start = coordinate - 1
            if cpg_start < 0 or cpg_start > 0xFFFFFFFF:
                raise ValueError(f"CpG coordinate outside uint32 range in {path}")
            chrom_positions.append(cpg_start)
            row_count += 1

        if offset < len(payload):
            offset += 1
    return positions, row_offsets, row_count


def read_annotation_index(cm_path: Path, cpg_path: Path) -> CpGAnnotationIndex:
    labels, state_codes = decode_state_mask(cm_path)
    positions, row_offsets, coordinate_count = decode_cpg_reference(cpg_path)
    if coordinate_count != len(state_codes):
        raise ValueError(
            "ChromHMM mask and CpG reference row counts differ: "
            f"{len(state_codes):,} states versus {coordinate_count:,} coordinates"
        )
    return CpGAnnotationIndex(positions, row_offsets, state_codes, labels)


def discover_files(host_dir: Path) -> list[Path]:
    paths = set(host_dir.rglob("*.CG.cov"))
    paths.update(host_dir.rglob("*.CG.cov.gz"))
    return sorted(paths)


def sample_name(path: Path) -> str:
    for suffix in (".CG.cov.gz", ".CG.cov"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    raise ValueError(f"Unexpected CG coverage filename: {path.name}")


def parse_spot_coordinates(spot: str) -> tuple[int, int]:
    fields = spot.rsplit("_", maxsplit=1)
    if len(fields) != 2:
        raise ValueError(f"Spot name does not encode X_Y coordinates: {spot}")
    try:
        return int(fields[0]), int(fields[1])
    except ValueError as error:
        raise ValueError(
            f"Spot name does not encode integer X_Y coordinates: {spot}"
        ) from error


def chromosome_sort_key(chrom: str) -> tuple[int, int | str]:
    """Order canonical mouse chromosomes numerically, followed by X, Y, and M."""
    label = chrom[3:] if chrom.lower().startswith("chr") else chrom
    if label.isdigit():
        return 0, int(label)
    special_order = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    upper_label = label.upper()
    if upper_label in special_order:
        return 0, special_order[upper_label]
    return 1, upper_label


def read_matched_sites(
    path: Path, annotation: CpGAnnotationIndex
) -> tuple[dict[str, set[int]], int]:
    """Return unique matched reference row indices grouped by chromosome."""
    matched: dict[str, set[int]] = defaultdict(set)
    unmatched_rows = 0
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(
                    f"Expected at least 3 coverage columns at {path}:{line_number}"
                )
            chrom = fields[0]
            try:
                cpg_start = int(fields[1])
            except ValueError as error:
                raise ValueError(
                    f"Invalid CpG coordinate at {path}:{line_number}"
                ) from error
            if cpg_start < 0:
                raise ValueError(f"Negative CpG coordinate at {path}:{line_number}")
            row_index = annotation.row_at(chrom, cpg_start)
            if row_index is None:
                unmatched_rows += 1
            else:
                matched[chrom].add(row_index)
    return matched, unmatched_rows


def count_matched_sites(matched: dict[str, set[int]]) -> int:
    return sum(len(indices) for indices in matched.values())


def reference_counts(
    annotation: CpGAnnotationIndex,
) -> tuple[dict[str, int], dict[int, dict[tuple[str, int], int]], list[int]]:
    """Count denominator CpGs by chromosome, fixed window, and state."""
    chromosome_counts = {
        chrom: len(chrom_positions)
        for chrom, chrom_positions in annotation.positions.items()
    }
    window_counts: dict[int, dict[tuple[str, int], int]] = {
        size: {} for size in WINDOW_SIZES
    }
    for chrom, chrom_positions in annotation.positions.items():
        coordinates = np.frombuffer(chrom_positions, dtype=np.uint32)
        for size in WINDOW_SIZES:
            counts = np.bincount(coordinates // size)
            for window_index, count in enumerate(counts):
                if count:
                    window_counts[size][(chrom, window_index)] = int(count)

    codes = np.frombuffer(annotation.state_codes, dtype=np.uint8)
    state_counts = np.bincount(codes, minlength=len(annotation.labels)).tolist()
    return chromosome_counts, window_counts, state_counts


def ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else math.nan


def percent(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * ratio(numerator, denominator)


def state_order(annotation: CpGAnnotationIndex) -> list[int]:
    label_to_code = {label: code for code, label in enumerate(annotation.labels)}
    ordered = [
        label_to_code[label]
        for label in PLOT_STATE_ORDER
        if label in label_to_code
    ]
    ordered.extend(
        code
        for code, label in enumerate(annotation.labels)
        if label not in PLOT_STATE_ORDER
    )
    return ordered


def write_tsv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, object]],
    *,
    compressed: bool = False,
) -> None:
    opener = gzip.open if compressed else Path.open
    kwargs = {"mode": "wt" if compressed else "w", "encoding": "utf-8", "newline": ""}
    with opener(path, **kwargs) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def plot_spatial_coverage(
    path: Path, records: list[SpotRecord], total_sites: int
) -> None:
    retained = [record for record in records if record.retained]
    max_x = max(record.x_index for record in records)
    max_y = max(record.y_index for record in records)
    matrix = np.zeros((max_y + 1, max_x + 1), dtype=float)
    for record in retained:
        matrix[record.y_index, record.x_index] = percent(
            record.matched_sites, total_sites
        )

    with plt.rc_context({"font.family": "sans-serif", "font.size": 10}):
        figure, axis = plt.subplots(figsize=(5, 4))
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad("#F2F2F2")
        image = axis.imshow(
            matrix,
            origin="lower",
            interpolation="nearest",
            aspect="equal",
            cmap=cmap,
            vmin=0,
            extent=(-0.5, max_x + 0.5, -0.5, max_y + 0.5),
        )
        axis.set_title("Spatial CpG coverage (retained spots)", fontsize=12)
        axis.set_xlabel("X index", fontsize=10)
        axis.set_ylabel("Y index", fontsize=10)
        colorbar = figure.colorbar(image, ax=axis, shrink=0.9)
        colorbar.set_label("Reference CpGs covered per spot (%)")
        figure.tight_layout()
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)


def plot_cumulative_coverage(path: Path, rows: list[dict[str, object]]) -> None:
    with plt.rc_context({"font.family": "sans-serif", "font.size": 10}):
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.plot(
            [0] + [int(row["spot_rank"]) for row in rows],
            [0.0] + [float(row["cumulative_coverage_percent"]) for row in rows],
            color=PLOT_COLOR,
            linewidth=1.5,
        )
        axis.set_title("Cumulative unique CpG coverage", fontsize=12)
        axis.set_xlabel("Retained spots (descending spot coverage)", fontsize=10)
        axis.set_ylabel("Reference CpGs covered (%)", fontsize=10)
        axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)
        axis.grid(color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        figure.tight_layout()
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)


def plot_genomic_coverage(
    path: Path,
    chromosome_rows: list[dict[str, object]],
    window_rows: dict[int, list[dict[str, object]]],
) -> None:
    with plt.rc_context({"font.family": "sans-serif", "font.size": 9}):
        figure, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)

        labels = [str(row["chrom"]) for row in chromosome_rows]
        values = [float(row["coverage_percent"]) for row in chromosome_rows]
        axes[0].bar(range(len(labels)), values, color=PLOT_COLOR, width=0.75)
        axes[0].set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axes[0].set_ylabel("Coverage (%)", fontsize=10)
        axes[0].set_title("Unique CpG coverage by chromosome", fontsize=12)
        axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axes[0].set_axisbelow(True)

        for axis, size in zip(axes[1:], WINDOW_SIZES):
            rows = window_rows[size]
            x_values = list(range(len(rows)))
            axis.plot(
                x_values,
                [float(row["coverage_percent"]) for row in rows],
                color=PLOT_COLOR,
                linewidth=0.6,
            )
            chromosome_midpoints: list[float] = []
            chromosome_labels: list[str] = []
            start = 0
            previous_chrom = str(rows[0]["chrom"])
            for index, row in enumerate(rows + [{"chrom": None}]):
                chrom = row["chrom"]
                if chrom != previous_chrom:
                    end = index
                    chromosome_midpoints.append((start + end - 1) / 2)
                    chromosome_labels.append(previous_chrom)
                    if index < len(rows):
                        axis.axvline(index - 0.5, color="#D9D9D9", linewidth=0.5)
                    start = index
                    previous_chrom = str(chrom)
            axis.set_xticks(
                chromosome_midpoints,
                chromosome_labels,
                rotation=45,
                ha="right",
            )
            axis.set_xlim(0, max(x_values))
            axis.set_ylim(bottom=0)
            axis.set_ylabel("Coverage (%)", fontsize=10)
            axis.set_title(f"Unique CpG coverage in {size // 1_000:,} kb windows", fontsize=12)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
            axis.set_axisbelow(True)
        axes[-1].set_xlabel("Genomic windows ordered by chromosome", fontsize=10)
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)


def plot_chromhmm_coverage(path: Path, rows: list[dict[str, object]]) -> None:
    rows = [row for row in rows if str(row["state"]) in PLOT_STATE_ORDER]
    with plt.rc_context({"font.family": "sans-serif", "font.size": 9}):
        figure, axis = plt.subplots(figsize=(6.5, 5.5))
        positions = np.arange(len(rows))
        axis.barh(
            positions,
            [float(row["pooled_coverage_percent"]) for row in rows],
            color=PLOT_COLOR,
            height=0.7,
        )
        axis.set_yticks(positions, [str(row["state"]) for row in rows], fontsize=10)
        axis.invert_yaxis()
        axis.set_xlabel("Reference CpGs covered across retained spots (%)", fontsize=10)
        axis.set_title("ChromHMM-normalized CpG coverage", fontsize=12)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        figure.tight_layout()
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)


def plot_chromhmm_spot_coverage(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    rows = [row for row in rows if str(row["state"]) in PLOT_STATE_ORDER]
    spots = list(dict.fromkeys(str(row["spot"]) for row in rows))
    states = list(dict.fromkeys(str(row["state"]) for row in rows))
    spot_indices = {spot: index for index, spot in enumerate(spots)}
    state_indices = {state: index for index, state in enumerate(states)}
    matrix = np.zeros((len(states), len(spots)), dtype=float)
    for row in rows:
        matrix[state_indices[str(row["state"])], spot_indices[str(row["spot"])]] = float(
            row["coverage_percent"]
        )

    positive_values = matrix[matrix > 0]
    color_maximum = (
        float(np.percentile(positive_values, 99)) if positive_values.size else None
    )
    tick_count = min(6, len(spots))
    tick_positions = np.linspace(0, len(spots) - 1, tick_count, dtype=int)
    with plt.rc_context({"font.family": "sans-serif", "font.size": 9}):
        figure, axis = plt.subplots(figsize=(10, 5))
        image = axis.imshow(
            matrix,
            origin="upper",
            interpolation="nearest",
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=color_maximum,
        )
        axis.set_yticks(
            range(len(states)),
            states,
            fontsize=9,
        )
        axis.set_xticks(
            tick_positions,
            [str(position + 1) for position in tick_positions],
        )
        axis.set_xlabel("Spot rank (descending total CpG coverage)", fontsize=10)
        axis.set_ylabel("ChromHMM state", fontsize=10)
        axis.set_title("ChromHMM CpG coverage by spot", fontsize=12)
        colorbar = figure.colorbar(image, ax=axis, shrink=0.9)
        colorbar.set_label(
            "Coverage per spot (%)\n(color scale capped at 99th percentile)",
            fontsize=10,
        )
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
        annotation = read_annotation_index(chromhmm_cm, cpg_reference)
        chromosome_reference, window_reference, state_reference = reference_counts(
            annotation
        )
    except (OSError, EOFError, UnicodeError, ValueError, struct.error) as error:
        raise SystemExit(str(error)) from error
    print(
        f"Loaded {annotation.total_sites:,} mm10 CpGs and "
        f"{len(annotation.labels)} ChromHMM labels",
        flush=True,
    )

    paths = discover_files(host_dir)
    if not paths:
        raise SystemExit(f"No *.CG.cov files found below {host_dir}")

    records: list[SpotRecord] = []
    observed_spot_names: set[str] = set()
    for file_number, path in enumerate(paths, start=1):
        try:
            spot = sample_name(path)
            if spot in observed_spot_names:
                raise ValueError(f"Duplicate CG coverage files for spot: {spot}")
            observed_spot_names.add(spot)
            x_index, y_index = parse_spot_coordinates(spot)
            matched, unmatched_rows = read_matched_sites(path, annotation)
        except (OSError, UnicodeError, ValueError) as error:
            raise SystemExit(str(error)) from error
        matched_sites = count_matched_sites(matched)
        state_counts = Counter(
            annotation.state_codes[row_index]
            for indices in matched.values()
            for row_index in indices
        )
        records.append(
            SpotRecord(
                path=path,
                spot=spot,
                x_index=x_index,
                y_index=y_index,
                matched_sites=matched_sites,
                unmatched_rows=unmatched_rows,
                retained=matched_sites >= args.min_total_sites,
                state_counts=dict(state_counts),
            )
        )
        if file_number % 100 == 0:
            print(f"Counted {file_number:,} CG coverage files", flush=True)

    retained = sorted(
        (record for record in records if record.retained),
        key=lambda record: (-record.matched_sites, record.spot),
    )
    if not retained:
        raise SystemExit("No spots passed --min-total-sites")

    observed_mask = bytearray(annotation.total_sites)
    chromosome_observed: dict[str, int] = defaultdict(int)
    window_observed: dict[int, dict[tuple[str, int], int]] = {
        size: defaultdict(int) for size in WINDOW_SIZES
    }
    state_observed: Counter[int] = Counter()
    cumulative_rows: list[dict[str, object]] = []
    cumulative_sites = 0
    for rank, record in enumerate(retained, start=1):
        try:
            matched, _ = read_matched_sites(record.path, annotation)
        except (OSError, UnicodeError, ValueError) as error:
            raise SystemExit(str(error)) from error
        new_sites = 0
        for chrom, indices in matched.items():
            row_offset = annotation.row_offsets[chrom]
            chrom_positions = annotation.positions[chrom]
            for row_index in indices:
                if observed_mask[row_index]:
                    continue
                observed_mask[row_index] = 1
                new_sites += 1
                chromosome_observed[chrom] += 1
                state_observed[annotation.state_codes[row_index]] += 1
                cpg_start = chrom_positions[row_index - row_offset]
                for size in WINDOW_SIZES:
                    window_observed[size][(chrom, cpg_start // size)] += 1
        cumulative_sites += new_sites
        cumulative_rows.append(
            {
                "spot_rank": rank,
                "spot": record.spot,
                "spot_cpg_sites": record.matched_sites,
                "new_cpg_sites": new_sites,
                "cumulative_cpg_sites": cumulative_sites,
                "reference_cpg_sites": annotation.total_sites,
                "cumulative_coverage_fraction": ratio(
                    cumulative_sites, annotation.total_sites
                ),
                "cumulative_coverage_percent": percent(
                    cumulative_sites, annotation.total_sites
                ),
            }
        )
        if rank % 100 == 0:
            print(f"Accumulated {rank:,} retained spots", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    spot_rows = [
        {
            "spot": record.spot,
            "X_index": record.x_index,
            "Y_index": record.y_index,
            "matched_cpg_sites": record.matched_sites,
            "reference_cpg_sites": annotation.total_sites,
            "coverage_fraction": ratio(record.matched_sites, annotation.total_sites),
            "coverage_percent": percent(record.matched_sites, annotation.total_sites),
            "unmatched_rows": record.unmatched_rows,
            "retained": record.retained,
        }
        for record in sorted(
            records, key=lambda record: (record.x_index, record.y_index)
        )
    ]
    write_tsv(
        output_dir / "cpg_coverage_by_spot.tsv",
        list(spot_rows[0]),
        spot_rows,
    )
    write_tsv(
        output_dir / "cumulative_cpg_coverage.tsv",
        list(cumulative_rows[0]),
        cumulative_rows,
    )

    chromosome_order = sorted(annotation.positions, key=chromosome_sort_key)
    chromosome_rows = [
        {
            "chrom": chrom,
            "observed_cpg_sites": chromosome_observed[chrom],
            "reference_cpg_sites": chromosome_reference[chrom],
            "coverage_fraction": ratio(
                chromosome_observed[chrom], chromosome_reference[chrom]
            ),
            "coverage_percent": percent(
                chromosome_observed[chrom], chromosome_reference[chrom]
            ),
        }
        for chrom in chromosome_order
    ]
    write_tsv(
        output_dir / "cpg_coverage_by_chromosome.tsv",
        list(chromosome_rows[0]),
        chromosome_rows,
    )

    window_rows_by_size: dict[int, list[dict[str, object]]] = {}
    for size in WINDOW_SIZES:
        rows = []
        for chrom in chromosome_order:
            keys = sorted(
                (
                    key
                    for key in window_reference[size]
                    if key[0] == chrom
                ),
                key=lambda key: key[1],
            )
            for key in keys:
                observed = window_observed[size][key]
                reference = window_reference[size][key]
                rows.append(
                    {
                        "chrom": chrom,
                        "start": key[1] * size,
                        "end": (key[1] + 1) * size,
                        "window_size": size,
                        "observed_cpg_sites": observed,
                        "reference_cpg_sites": reference,
                        "coverage_fraction": ratio(observed, reference),
                        "coverage_percent": percent(observed, reference),
                    }
                )
        window_rows_by_size[size] = rows
        label = "100kb" if size == 100_000 else "1mb"
        write_tsv(
            output_dir / f"cpg_coverage_by_{label}_window.tsv.gz",
            list(rows[0]),
            rows,
            compressed=True,
        )

    ordered_codes = state_order(annotation)
    chromhmm_spot_rows = []
    spot_state_percentages: dict[int, list[float]] = defaultdict(list)
    for record in retained:
        for code in ordered_codes:
            reference = state_reference[code]
            observed = record.state_counts.get(code, 0)
            coverage_fraction = ratio(observed, reference)
            coverage_percent = percent(observed, reference)
            spot_state_percentages[code].append(coverage_percent)
            chromhmm_spot_rows.append(
                {
                    "spot": record.spot,
                    "X_index": record.x_index,
                    "Y_index": record.y_index,
                    "state": annotation.labels[code],
                    "observed_cpg_sites": observed,
                    "reference_cpg_sites": reference,
                    "coverage_fraction": coverage_fraction,
                    "coverage_percent": coverage_percent,
                }
            )
    write_tsv(
        output_dir / "chromhmm_cpg_coverage_by_spot.tsv.gz",
        list(chromhmm_spot_rows[0]),
        chromhmm_spot_rows,
        compressed=True,
    )

    chromhmm_summary_rows = []
    for code in ordered_codes:
        values = spot_state_percentages[code]
        reference = state_reference[code]
        observed = state_observed[code]
        chromhmm_summary_rows.append(
            {
                "state": annotation.labels[code],
                "observed_cpg_sites": observed,
                "reference_cpg_sites": reference,
                "pooled_coverage_fraction": ratio(observed, reference),
                "pooled_coverage_percent": percent(observed, reference),
                "retained_spots": len(retained),
                "spot_mean_coverage_percent": statistics.fmean(values),
                "spot_q1_coverage_percent": percentile(values, 0.25),
                "spot_median_coverage_percent": statistics.median(values),
                "spot_q3_coverage_percent": percentile(values, 0.75),
            }
        )
    write_tsv(
        output_dir / "chromhmm_cpg_coverage_summary.tsv",
        list(chromhmm_summary_rows[0]),
        chromhmm_summary_rows,
    )

    plot_spatial_coverage(
        output_dir / "spatial_cpg_coverage_heatmap.png",
        records,
        annotation.total_sites,
    )
    plot_cumulative_coverage(
        output_dir / "cumulative_cpg_coverage.png", cumulative_rows
    )
    plot_genomic_coverage(
        output_dir / "genomic_cpg_coverage.png",
        chromosome_rows,
        window_rows_by_size,
    )
    plot_chromhmm_coverage(
        output_dir / "chromhmm_cpg_coverage.png", chromhmm_summary_rows
    )
    plot_chromhmm_spot_coverage(
        output_dir / "chromhmm_cpg_coverage_by_spot.png",
        chromhmm_spot_rows,
    )

    print("Context: CG")
    print(f"Coverage files: {len(records):,}")
    print(f"Spots retained: {len(retained):,}")
    print(f"Reference CpGs: {annotation.total_sites:,}")
    print(f"Unique CpGs covered: {cumulative_sites:,}")
    print(
        "Cumulative CpG coverage: "
        f"{percent(cumulative_sites, annotation.total_sites):.6f}%"
    )
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
