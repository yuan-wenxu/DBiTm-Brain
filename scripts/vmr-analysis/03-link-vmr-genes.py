#!/usr/bin/env python3
"""Annotate MethSCAn VMRs with GENCODE genes using pyranges."""

from __future__ import annotations

import argparse
import csv
import gzip
import re
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
import pyranges as pr


VMR_PATTERN = re.compile(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
GENE_ATTR_PATTERN = re.compile(
    r'gene_id "(?P<gene_id>[^"]+)";.*?'
    r'gene_type "(?P<gene_type>[^"]+)";.*?'
    r'gene_name "(?P<gene_name>[^"]+)";'
)
GTF_COLUMNS = (
    "chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtf", type=Path, required=True,
                        help="GENCODE GTF (gzipped or plain).")
    parser.add_argument("--matrix", type=Path,
                        help="MethSCAn methylation_fractions.csv.gz; VMRs from header.")
    parser.add_argument("--bed", type=Path,
                        help="BED file of VMRs (0-based half-open).")
    parser.add_argument("--context", default="CG",
                        help="Context label for output rows (default: CG).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--promoter-upstream", type=int, default=2000,
                        help="Promoter window upstream of TSS (default: 2000).")
    parser.add_argument("--promoter-downstream", type=int, default=500,
                        help="Promoter window downstream of TSS (default: 500).")
    args = parser.parse_args()
    if bool(args.matrix) == bool(args.bed):
        parser.error("Provide exactly one of --matrix or --bed.")
    return args


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open(mode="r", encoding="utf-8", newline="")


def read_genes(path: Path) -> pd.DataFrame:
    """Read gene rows from a GENCODE GTF into a 0-based half-open DataFrame."""
    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=GTF_COLUMNS,
        dtype=str,
        compression="gzip" if path.suffix == ".gz" else "infer",
    )
    genes = df[df["feature"] == "gene"].copy()
    if genes.empty:
        raise ValueError(f"No gene records found in {path}")

    # Keep the full gene span (BED style) and the strand as a plain column,
    # then build an unstranded PyRanges so VMRs are matched on both strands.
    genes["start0"] = genes["start"].astype(int) - 1
    genes["end0"] = genes["end"].astype(int)
    matches = genes["attr"].str.extract(GENE_ATTR_PATTERN)
    if matches.isna().any().any():
        raise ValueError(f"Some gene records lack gene_id/type/name attributes: {path}")
    result = pd.DataFrame(
        {
            "Chromosome": genes["chrom"],
            "Start": genes["start0"],
            "End": genes["end0"],
            "gene_strand": genes["strand"],
            "gene_id": matches["gene_id"],
            "gene_name": matches["gene_name"],
            "gene_type": matches["gene_type"],
            "gene_start": genes["start0"],
            "gene_end": genes["end0"],
        }
    )
    return result


def parse_vmr_regions(matrix: Path) -> pd.DataFrame:
    """Extract VMR coordinates from the MethSCAn matrix header."""
    with open_text(matrix) as handle:
        header = next(csv.reader(handle))
    rows: list[dict[str, str | int]] = []
    for column_number, column in enumerate(header[1:], start=2):
        match = VMR_PATTERN.fullmatch(column)
        if match is None:
            raise ValueError(f"Invalid VMR header in column {column_number}: {column!r}")
        start = int(match.group("start"))
        end = int(match.group("end"))
        if end <= start:
            raise ValueError(f"Invalid VMR span in column {column_number}: {column!r}")
        rows.append(
            {
                "Chromosome": match.group("chrom"),
                "Start": start,
                "End": end,
                "vmr": f"{match.group('chrom')}:{start}-{end}",
            }
        )
    return pd.DataFrame(rows)


def parse_bed(file: Path) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    with open_text(file) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"Malformed BED line {line_number}: {file}")
            start = int(fields[1])
            end = int(fields[2])
            if end <= start:
                raise ValueError(f"Non-positive interval at {file}:{line_number}")
            rows.append(
                {
                    "Chromosome": fields[0],
                    "Start": start,
                    "End": end,
                    "vmr": f"{fields[0]}:{start}-{end}",
                }
            )
    return pd.DataFrame(rows)


def tss_of(row: pd.Series) -> int:
    if row["gene_strand"] == "+":
        return int(row["gene_start"])
    return int(row["gene_end"]) - 1


def distance_to_gene(start: int, end: int, gene_start: int, gene_end: int) -> int:
    if end <= gene_start:
        return gene_start - end
    if start >= gene_end:
        return start - gene_end
    return 0


def distance_to_tss(start: int, end: int, tss: int) -> int:
    """Distance in bp from [start, end) to a TSS point (0 if it contains TSS)."""
    if start <= tss < end:
        return 0
    return min(abs(start - tss), abs(end - 1 - tss))


def promoter_span(row: pd.Series, upstream: int, downstream: int) -> tuple[int, int]:
    tss = tss_of(row)
    if row["gene_strand"] == "+":
        return (max(0, tss - upstream), tss + downstream)
    return (max(0, tss - downstream), tss + upstream)


def write_output(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    fields = [
        "context", "vmr", "chrom", "start", "end", "length_bp",
        "gene_id", "gene_name", "gene_type", "gene_strand", "gene_tss",
        "relation", "overlap_bp", "distance_to_gene_bp", "distance_to_tss_bp",
    ]
    with path.open(mode="w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    gtf_path = args.gtf.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not gtf_path.is_file():
        raise SystemExit(f"GTF does not exist: {gtf_path}")

    print(f"Parsing gene annotations: {gtf_path}", flush=True)
    gene_df = read_genes(gtf_path)
    gr_genes = pr.PyRanges(gene_df)
    print(
        f"Loaded {len(gene_df):,} genes across "
        f"{gene_df['Chromosome'].nunique():,} chromosomes",
        flush=True,
    )

    if args.matrix:
        vmr_df = parse_vmr_regions(args.matrix.expanduser().resolve())
        context = args.context
    else:
        vmr_df = parse_bed(args.bed.expanduser().resolve())
        context = "BED"
    if vmr_df.empty:
        raise SystemExit("No VMRs found in input")
    gr_vmrs = pr.PyRanges(vmr_df)
    print(f"VMRs: {len(vmr_df):,}", flush=True)

    body = gr_vmrs.join(gr_genes).df
    body["overlap_bp"] = (
        np.minimum(body["End"].astype(int), body["End_b"].astype(int))
        - np.maximum(body["Start"].astype(int), body["Start_b"].astype(int))
    )
    print(f"Gene-body overlaps: {len(body):,}", flush=True)

    promoter = pd.DataFrame(
        {
            "Chromosome": gene_df["Chromosome"],
            "gene_id": gene_df["gene_id"],
            "gene_name": gene_df["gene_name"],
            "gene_type": gene_df["gene_type"],
            "gene_strand": gene_df["gene_strand"],
            "gene_start": gene_df["gene_start"],
            "gene_end": gene_df["gene_end"],
        }
    )
    spans = promoter.apply(
        lambda row: promoter_span(row, args.promoter_upstream, args.promoter_downstream),
        axis=1,
    )
    promoter["Start"] = [span[0] for span in spans]
    promoter["End"] = [span[1] for span in spans]
    gr_promoters = pr.PyRanges(promoter)

    body_ids = set(body["vmr"])
    remaining_vmr_df = vmr_df[~vmr_df["vmr"].isin(body_ids)]
    # If every VMR is accounted for we can skip the promoter step entirely.
    prom_rows: pd.DataFrame = pd.DataFrame()
    if not remaining_vmr_df.empty:
        gr_remaining = pr.PyRanges(remaining_vmr_df)
        prom = gr_remaining.join(gr_promoters).df
        prom_rows = prom
    print(f"Promoter overlaps: {len(prom_rows):,}", flush=True)

    prom_vmr_keys = set(body["vmr"]) | set(prom_rows["vmr"]) if not prom_rows.empty else set(body["vmr"])
    intergenic_vmr_df = vmr_df[~vmr_df["vmr"].isin(prom_vmr_keys)]
    near_rows: pd.DataFrame = pd.DataFrame()
    if not intergenic_vmr_df.empty:
        gr_intergenic = pr.PyRanges(intergenic_vmr_df)
        near = gr_intergenic.nearest(gr_genes).df
        near["overlap_bp"] = 0
        near["Distance"] = near["Distance"].astype(int)
        near_rows = near
    print(f"Intergenic (nearest gene): {len(near_rows):,}", flush=True)

    collected: dict[tuple[str, str], dict[str, object]] = {}

    def add(
        chromosome: str,
        start: int,
        end: int,
        vmr: str,
        gene_id: str,
        gene_name: str,
        gene_type: str,
        gene_strand: str,
        gene_start: int,
        gene_end: int,
        relation: str,
        overlap_bp: int,
    ) -> None:
        key = (vmr, gene_id)
        if relation == "gene" and key in collected:
            return
        if relation == "promoter" and key in collected:
            return
        tss = gene_end - 1 if gene_strand == "-" else gene_start
        collected[key] = {
            "context": context,
            "vmr": vmr,
            "chrom": chromosome,
            "start": start,
            "end": end,
            "length_bp": end - start,
            "gene_id": gene_id,
            "gene_name": gene_name,
            "gene_type": gene_type,
            "gene_strand": gene_strand,
            "gene_tss": tss,
            "relation": relation,
            "overlap_bp": overlap_bp,
            "distance_to_gene_bp": 0 if overlap_bp > 0 else distance_to_gene(
                start, end, gene_start, gene_end
            ),
            "distance_to_tss_bp": distance_to_tss(start, end, tss),
        }

    if not body.empty:
        for _, row in body.iterrows():
            add(
                row["Chromosome"], int(row["Start"]), int(row["End"]), row["vmr"],
                row["gene_id"], row["gene_name"], row["gene_type"],
                row["gene_strand"], int(row["gene_start"]), int(row["gene_end"]),
                "gene", int(row["overlap_bp"]),
            )
    if not prom_rows.empty:
        for _, row in prom_rows.iterrows():
            add(
                row["Chromosome"], int(row["Start"]), int(row["End"]), row["vmr"],
                row["gene_id"], row["gene_name"], row["gene_type"],
                row["gene_strand"], int(row["gene_start"]), int(row["gene_end"]),
                "promoter", 0,
            )
    if not near_rows.empty:
        for _, row in near_rows.iterrows():
            add(
                row["Chromosome"], int(row["Start"]), int(row["End"]), row["vmr"],
                row["gene_id"], row["gene_name"], row["gene_type"],
                row["gene_strand"], int(row["gene_start"]), int(row["gene_end"]),
                "intergenic", 0,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_output(output_path, list(collected.values()))
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
