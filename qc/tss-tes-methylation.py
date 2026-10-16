#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_COLORS = {"TAPS": "#4DBBD5", "TAPS-beta": "#E64B35"}
FLANK_BP = 3000
FLANK_BIN_BP = 100
BODY_BINS = 100
OUTPUT_NAME = "tss-tes-methylation.png"
GENE_ID_PATTERN = re.compile(r'(?:^|;)\s*gene_id\s+"([^"]+)"')
COV_COLUMNS = ["chrom", "start", "end", "percent"]


@dataclass
class GeneBins:
    edges: np.ndarray
    strand: str
    numerator: np.ndarray
    observations: np.ndarray

    @property
    def lower(self) -> np.ndarray:
        return self.edges[:, :-1] if self.strand == "+" else self.edges[:, 1:]

    @property
    def upper(self) -> np.ndarray:
        return self.edges[:, 1:] if self.strand == "+" else self.edges[:, :-1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gtf", required=True, type=Path,
                        help="GTF or GTF.gz with gene features; same assembly "
                             "and chromosome names as the COV files.")
    parser.add_argument("--taps-cov", required=True, type=Path,
                        help="TAPS COV or COV.gz to plot.")
    parser.add_argument("--beta-cov", required=True, type=Path,
                        help="TAPS-beta COV or COV.gz to plot. "
                             "Extra columns are ignored.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=250000,
                        help="COV rows per read chunk (default: 250000).")
    args = parser.parse_args(argv)
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be > 0")
    return args


def read_genes(path: Path) -> pd.DataFrame:
    """Read genes once, without multiplying transcript isoforms."""
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"GTF line {line_number}: expected 9 columns")
            if fields[2] != "gene":
                continue
            gene_id = GENE_ID_PATTERN.search(fields[8])
            start, end = int(fields[3]) - 1, int(fields[4])
            if (gene_id is None or start < 0 or end <= start
                    or fields[6] not in ("+", "-")):
                raise ValueError(f"Invalid gene at GTF line {line_number}")
            rows.append((fields[0], start, end, fields[6], gene_id[1]))
    genes = pd.DataFrame(
        rows, columns=["chrom", "start", "end", "strand", "gene_id"]
    )
    if genes.empty:
        raise ValueError("GTF contains no gene features")
    if genes["gene_id"].duplicated().any():
        raise ValueError("GTF contains duplicate gene_id values")
    return genes


def make_gene_bins(genes: pd.DataFrame) -> dict[str, list[GeneBins]]:
    groups: dict[str, list[GeneBins]] = {}
    upstream = np.arange(-FLANK_BP, 0, FLANK_BIN_BP)
    downstream = np.arange(FLANK_BIN_BP, FLANK_BP + 1, FLANK_BIN_BP)
    for (chrom, strand), group in genes.groupby(["chrom", "strand"]):
        starts = group["start"].to_numpy(dtype=np.int64)[:, None]
        ends = group["end"].to_numpy(dtype=np.int64)[:, None]
        lengths = ends - starts
        # Ceil boundaries assign each base to exactly one body bin, including
        # short genes whose empty bins must not gain artificial observations.
        body = (
            lengths * np.arange(BODY_BINS + 1) + BODY_BINS - 1
        ) // BODY_BINS
        offsets = np.concatenate((
            np.broadcast_to(upstream, (len(group), len(upstream))),
            body,
            lengths + downstream,
        ), axis=1)
        edges = starts + offsets if strand == "+" else ends - offsets
        shape = (len(group), edges.shape[1] - 1)
        groups.setdefault(str(chrom), []).append(GeneBins(
            edges=edges,
            strand=str(strand),
            numerator=np.zeros(shape, dtype=np.float64),
            observations=np.zeros(shape, dtype=np.int64),
        ))
    return groups


def validate_cov(chunk: pd.DataFrame, path: Path) -> None:
    values = chunk[COV_COLUMNS[1:]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Missing or nonfinite numeric COV values: {path}")
    integers = chunk[["start", "end"]].to_numpy()
    if (integers < 0).any() or (integers != np.floor(integers)).any():
        raise ValueError(f"COV positions must be integers >= 0: {path}")
    if (chunk["start"] < 1).any() or not chunk["start"].eq(chunk["end"]).all():
        raise ValueError(f"Expected single-base, 1-based COV positions: {path}")
    if not chunk["percent"].between(0, 100).all():
        raise ValueError(f"COV column 4 must be in [0, 100]: {path}")
    if chunk["chrom"].isna().any():
        raise ValueError(f"Missing chromosome names: {path}")


def accumulate_chunk(chunk: pd.DataFrame, bins: GeneBins) -> None:
    order = np.argsort(chunk["start"].to_numpy(), kind="stable")
    positions = chunk["start"].to_numpy(dtype=np.int64)[order] - 1
    left = np.searchsorted(positions, bins.lower)
    right = np.searchsorted(positions, bins.upper)
    bins.observations += right - left
    values = chunk["percent"]
    prefix = np.concatenate(([0.0], np.cumsum(values.to_numpy()[order])))
    bins.numerator += prefix[right] - prefix[left]


def bin_metadata() -> pd.DataFrame:
    flank_bins = FLANK_BP // FLANK_BIN_BP
    total = 2 * flank_bins + BODY_BINS
    starts = np.concatenate((
        np.arange(-FLANK_BP, 0, FLANK_BIN_BP),
        np.arange(BODY_BINS) / BODY_BINS,
        np.arange(0, FLANK_BP, FLANK_BIN_BP),
    ))
    ends = starts + np.concatenate((
        np.full(flank_bins, FLANK_BIN_BP),
        np.full(BODY_BINS, 1 / BODY_BINS),
        np.full(flank_bins, FLANK_BIN_BP),
    ))
    return pd.DataFrame({
        "bin": np.arange(total),
        "region": (["upstream"] * flank_bins
                   + ["body"] * BODY_BINS
                   + ["downstream"] * flank_bins),
        "relative_start": starts,
        "relative_end": ends,
        "relative_unit": (["bp_from_TSS"] * flank_bins
                          + ["gene_fraction"] * BODY_BINS
                          + ["bp_from_TES"] * flank_bins),
    })


def summarize_profile(
    groups: dict[str, list[GeneBins]],
) -> pd.DataFrame:
    """Average COV percentages per gene/bin, then average covered genes."""
    result = bin_metadata()
    gene_counts = np.zeros(len(result), dtype=np.int64)
    observations = np.zeros(len(result), dtype=np.int64)
    gene_rate_sum = np.zeros(len(result), dtype=np.float64)
    for chromosome_bins in groups.values():
        for bins in chromosome_bins:
            covered = bins.observations > 0
            rates = np.divide(
                bins.numerator, bins.observations,
                out=np.zeros_like(bins.numerator), where=covered,
            )
            gene_rate_sum += rates.sum(axis=0)
            gene_counts += covered.sum(axis=0)
            observations += bins.observations.sum(axis=0)
    result["mean_methylation_percent"] = np.divide(
        gene_rate_sum, gene_counts,
        out=np.full(len(result), np.nan), where=gene_counts > 0,
    )
    result["covered_genes"] = gene_counts
    result["gene_bin_observations"] = observations
    return result


def read_profile(
    path: Path, genes: pd.DataFrame, chunk_size: int,
) -> pd.DataFrame:
    groups = make_gene_bins(genes)
    total_rows = matched_rows = 0
    dtypes = {column: "float64" for column in COV_COLUMNS[1:]}
    dtypes["chrom"] = "str"
    with pd.read_csv(
        path, sep="\t", header=None, usecols=range(4),
        names=COV_COLUMNS, dtype=dtypes, chunksize=chunk_size,
    ) as reader:
        for chunk_number, chunk in enumerate(reader, start=1):
            validate_cov(chunk, path)
            total_rows += len(chunk)
            for chrom, chromosome_chunk in chunk.groupby("chrom", sort=False):
                if chrom not in groups:
                    continue
                matched_rows += len(chromosome_chunk)
                for bins in groups[chrom]:
                    accumulate_chunk(chromosome_chunk, bins)
            if chunk_number % 20 == 0:
                print(f"[tss-tes] {path.name}: {total_rows:,} rows read",
                      flush=True)
    if not matched_rows:
        raise ValueError(
            f"No COV rows match GTF chromosomes: {path}; "
            "check chromosome names and genome assembly"
        )
    profile = summarize_profile(groups)
    if not profile["covered_genes"].any():
        raise ValueError(f"No observations overlap any gene/flank bins: {path}")

    return profile


def plot_profiles(profiles: pd.DataFrame, path: Path) -> None:
    flank_bins = FLANK_BP // FLANK_BIN_BP
    total_bins = 2 * flank_bins + BODY_BINS
    figure, axis = plt.subplots(figsize=(3, 2), layout="constrained")
    for method, color in METHOD_COLORS.items():
        frame = profiles.loc[profiles["method"].eq(method)]
        axis.plot(
            frame["bin"] + 0.5, frame["mean_methylation_percent"],
            color=color, label=method, linewidth=1.5,
        )
    for boundary in (flank_bins, flank_bins + BODY_BINS):
        axis.axvline(boundary, color="#999999", linewidth=0.8)
    axis.set_xlim(0, total_bins)
    axis.set_ylim(bottom=0)
    axis.set_xticks(
        [0, flank_bins, flank_bins + BODY_BINS, total_bins],
        [f"-{FLANK_BP / 1000:g} kb", "TSS", "TES",
         f"+{FLANK_BP / 1000:g} kb"],
    )
    axis.tick_params(axis="both", labelsize=5)
    axis.set_title("TSS–TES methylation", fontsize=7)
    axis.set_ylabel("Mean methylation level (%)", fontsize=6)
    axis.legend(loc="lower right", frameon=False, fontsize=5)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        args.gtf = args.gtf.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        output_path = output_dir / OUTPUT_NAME
        if output_path.exists():
            raise ValueError(f"Output already exists: {output_path}")
        input_paths = {
            "TAPS": args.taps_cov.expanduser().resolve(),
            "TAPS-beta": args.beta_cov.expanduser().resolve(),
        }
        for path in input_paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"COV file not found: {path}")
        genes = read_genes(args.gtf)
        profiles = []
        for method, path in input_paths.items():
            print(f"[tss-tes] method={method} input={path}",
                  flush=True)
            profile = read_profile(path, genes, args.chunk_size)
            profile.insert(0, "method", method)
            profiles.append(profile)

        combined = pd.concat(profiles, ignore_index=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_profiles(combined, output_path)
        print(f"[tss-tes] output-dir={output_dir}")
        return 0
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"[tss-tes] error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
