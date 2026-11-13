#!/usr/bin/env python3
"""Create compressed Parquet tables of spot-level gene-bin methylation."""

from __future__ import annotations

import argparse
import gzip
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = pa.schema([
    ("spot", pa.string()), ("gene_id", pa.string()),
    ("gene_name", pa.string()), ("chrom", pa.string()),
    ("strand", pa.string()), ("bin_index", pa.int32()),
    ("bin", pa.string()), ("methylation_rate", pa.float64()),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write spot x gene-bin methylation rates to ZSTD Parquet."
    )
    parser.add_argument("--gtf", type=Path, required=True,
                        help="GTF/GTF.GZ annotation containing gene features")
    parser.add_argument("--cov", type=Path, required=True,
                        help="Seven-column, coordinate-sorted COV file")
    parser.add_argument("--output", type=Path,
                        help="Output Parquet path; default: beside the input")
    parser.add_argument("--upstream-bp", type=int, default=2000)
    parser.add_argument("--downstream-bp", type=int, default=2000)
    parser.add_argument("--upstream-bins", type=int, default=20)
    parser.add_argument("--downstream-bins", type=int, default=20)
    parser.add_argument("--body-bins", type=int, default=100)
    args = parser.parse_args()
    if args.output is None:
        source_path = args.cov
        if source_path.suffix == ".gz":
            source_path = source_path.with_suffix("")
        args.output = source_path.with_suffix(".genebin.parquet")
    return args


def write_parquet_table(writer: pq.ParquetWriter, table: pa.Table) -> None:
    table = table.sort_by([
        ("gene_id", "ascending"), ("spot", "ascending"),
        ("bin_index", "ascending"),
    ])
    writer.write_table(table, row_group_size=100_000)


@dataclass
class ParquetRows:
    writer: pq.ParquetWriter
    rows: list[list[object]] = field(default_factory=list)

    def writerow(self, row: list[object]) -> None:
        self.rows.append(row)
        if len(self.rows) >= 100_000:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        arrays = [pa.array(values, type=column.type)
                  for values, column in zip(zip(*self.rows), SCHEMA)]
        write_parquet_table(self.writer, pa.Table.from_arrays(arrays, schema=SCHEMA))
        self.rows.clear()


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def gtf_attributes(text: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for field_text in text.strip().rstrip(";").split(";"):
        field_text = field_text.strip()
        if not field_text:
            continue
        key, _, value = field_text.partition(" ")
        attributes[key] = value.strip().strip('"')
    return attributes


def read_genes(
    gtf_path: Path, args: argparse.Namespace
) -> dict[str, list["GeneState"]]:
    by_chrom: dict[str, list[GeneState]] = {}
    seen: set[str] = set()
    gene_index = 0
    with open_text(gtf_path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            chrom, _, _, start_text, end_text, _, strand, _, attrs = fields
            attributes = gtf_attributes(attrs)
            gene_id = attributes.get("gene_id")
            if not gene_id:
                raise ValueError(f"GTF line {line_number} has no gene_id")
            if gene_id in seen:
                raise ValueError(f"Duplicate gene_id in GTF: {gene_id}")
            start, end = int(start_text) - 1, int(end_text)
            if start < 0 or end <= start or strand not in {"+", "-"}:
                raise ValueError(f"Invalid gene interval at GTF line {line_number}")
            seen.add(gene_id)
            gene = {
                "gene_id": gene_id,
                "gene_name": attributes.get("gene_name", gene_id),
                "chrom": chrom,
                "start": start,
                "end": end,
                "strand": strand,
            }
            if strand == "+":
                lower = start - args.upstream_bp
                upper = end + args.downstream_bp
            else:
                lower = start - args.downstream_bp
                upper = end + args.upstream_bp
            by_chrom.setdefault(chrom, []).append(
                GeneState(gene_index, gene, lower, upper)
            )
            gene_index += 1
    if not gene_index:
        raise ValueError(f"No gene features found in {gtf_path}")
    for chrom_genes in by_chrom.values():
        chrom_genes.sort(key=lambda state: (state.lower, state.index))
    return by_chrom


@dataclass
class GeneState:
    index: int
    gene: dict[str, object]
    lower: int
    upper: int
    # key=(spot barcode, bin); value=[sum of CpG rates, number of CpGs]
    values: dict[tuple[str, int], list[float | int]] = field(default_factory=dict)


def bin_index(
    state: GeneState,
    position: int,
    upstream_bp: int,
    downstream_bp: int,
    upstream_bins: int,
    downstream_bins: int,
    body_bins: int,
) -> int:
    gene = state.gene
    start, end = int(gene["start"]), int(gene["end"])
    strand = str(gene["strand"])
    gene_length = end - start

    if start <= position < end:
        body = (position - start) * body_bins // gene_length
        if strand == "-":
            body = body_bins - 1 - body
        return upstream_bins + body

    if strand == "+":
        if position < start:
            return (position - (start - upstream_bp)) // (
                upstream_bp // upstream_bins
            )
        return upstream_bins + body_bins + (position - end) // (
            downstream_bp // downstream_bins
        )

    if position >= end:
        distance = position - end
        return upstream_bins - 1 - distance // (upstream_bp // upstream_bins)
    distance = position - (start - downstream_bp)
    return (upstream_bins + body_bins + downstream_bins - 1
            - distance // (downstream_bp // downstream_bins))


def add_position(
    position: int,
    site_rates: dict[str, list[float | int]],
    active: list[GeneState],
    args: argparse.Namespace,
) -> None:
    if not site_rates:
        return
    n_bins = args.upstream_bins + args.body_bins + args.downstream_bins
    for state in active:
        if not state.lower <= position < state.upper:
            continue
        current_bin = bin_index(
            state, position, args.upstream_bp, args.downstream_bp,
            args.upstream_bins, args.downstream_bins, args.body_bins,
        )
        if not 0 <= current_bin < n_bins:
            raise ValueError("A CpG was assigned outside the gene-bin range")
        for spot, (rate_sum, row_count) in site_rates.items():
            rate = rate_sum / row_count
            key = (spot, current_bin)
            value = state.values.get(key)
            if value is None:
                state.values[key] = [rate, 1]
            else:
                value[0] += rate
                value[1] += 1


def write_gene_rows(
    state: GeneState,
    writer: ParquetRows,
    args: argparse.Namespace,
) -> None:
    labels = (
        [f"up_{index + 1}" for index in range(args.upstream_bins)]
        + [f"body_{index + 1}" for index in range(args.body_bins)]
        + [f"down_{index + 1}" for index in range(args.downstream_bins)]
    )
    gene = state.gene
    for (spot, current_bin), (rate_sum, site_count) in sorted(state.values.items()):
        writer.writerow([
            spot, gene["gene_id"], gene["gene_name"], gene["chrom"],
            gene["strand"], current_bin, labels[current_bin],
            rate_sum / site_count,
        ])
    state.values.clear()


def main() -> None:
    args = parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        bin_values = (
            args.upstream_bp, args.downstream_bp, args.upstream_bins,
            args.downstream_bins, args.body_bins,
        )
        if any(value <= 0 for value in bin_values):
            raise ValueError("Flank lengths and bin counts must be positive")
        if args.upstream_bins != args.downstream_bins:
            raise ValueError("Upstream and downstream bin counts must match")
        if args.upstream_bp % args.upstream_bins:
            raise ValueError("Upstream length must divide evenly into bins")
        if args.downstream_bp % args.downstream_bins:
            raise ValueError("Downstream length must divide evenly into bins")
        by_chrom = read_genes(args.gtf, args)
        active: list[GeneState] = []
        current_chrom = ""
        closed_chroms: set[str] = set()
        chromosome_genes: list[GeneState] = []
        next_gene = 0
        current_position: int | None = None
        site_rates: dict[str, list[float | int]] = {}

        with open_text(args.cov) as cov_handle, args.output.open(
            "xb"
        ) as output, pq.ParquetWriter(
            output, SCHEMA, compression="zstd", compression_level=3,
            use_dictionary=True, write_statistics=True,
        ) as parquet:
            writer = ParquetRows(parquet)

            for line_number, line in enumerate(cov_handle, 1):
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) < 7:
                    raise ValueError(f"COV line {line_number} has fewer than 7 columns")
                chrom = fields[0]
                # The project's per-spot COV files use 0-based positions.
                position = int(fields[1])
                end = int(fields[2])
                if position < 0 or end != position:
                    raise ValueError(
                        f"COV line {line_number} is not a single-base position"
                    )
                try:
                    rate = float(fields[3]) / 100.0
                except ValueError as error:
                    raise ValueError(
                        f"Invalid methylation percentage on COV line {line_number}"
                    ) from error
                if not 0.0 <= rate <= 1.0:
                    raise ValueError(
                        f"Methylation percentage outside 0-100 on COV line {line_number}"
                    )

                if chrom != current_chrom:
                    if current_position is not None:
                        add_position(current_position, site_rates, active, args)
                    for state in active:
                        write_gene_rows(state, writer, args)
                    active.clear()
                    if current_chrom:
                        closed_chroms.add(current_chrom)
                    if chrom in closed_chroms:
                        raise ValueError("COV chromosomes are not contiguous")
                    current_chrom = chrom
                    chromosome_genes = by_chrom.get(chrom, [])
                    next_gene = 0
                    current_position = None
                    site_rates = {}
                elif current_position is not None and position < current_position:
                    raise ValueError("COV must be coordinate-sorted")

                if current_position != position:
                    if current_position is not None:
                        add_position(current_position, site_rates, active, args)
                    current_position = position
                    site_rates = {}
                    expired = [state for state in active if state.upper <= position]
                    for state in expired:
                        write_gene_rows(state, writer, args)
                    active = [state for state in active if state.upper > position]
                    while (next_gene < len(chromosome_genes)
                           and chromosome_genes[next_gene].lower <= position):
                        state = chromosome_genes[next_gene]
                        next_gene += 1
                        if state.upper > position:
                            active.append(state)

                values = site_rates.get(fields[6])
                if values is None:
                    site_rates[fields[6]] = [rate, 1]
                else:
                    values[0] += rate
                    values[1] += 1
                if line_number % 5_000_000 == 0:
                    print(
                        f"Processed {line_number:,} COV lines",
                        file=sys.stderr, flush=True,
                    )

            if current_position is not None:
                add_position(current_position, site_rates, active, args)
            for state in active:
                write_gene_rows(state, writer, args)
            writer.flush()

    except (OSError, ValueError, pa.ArrowException) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"Wrote spot-by-gene-bin Parquet: {args.output}")


if __name__ == "__main__":
    main()
