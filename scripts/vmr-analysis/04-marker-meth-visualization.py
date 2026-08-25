#!/usr/bin/env python3
"""Visualize each marker-gene VMR as a separate spatial methylation heatmap."""

from __future__ import annotations

import argparse
import csv
import gzip
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True,
                        help="MethSCAn methylation_fractions.csv.gz.")
    parser.add_argument("--gene-table", type=Path, required=True,
                        help="VMR-to-gene table from 02-link-vmr-genes.py.")
    parser.add_argument("--gene-file", type=Path, required=True,
                        help="Headerless, one-column TSV containing marker gene names.")
    parser.add_argument("--chip-size", type=int, choices=(50, 100), required=True,
                        help="Number of grid cells along each chip axis: 50 or 100.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row-suffix", default=".CG",
                        help="Suffix stripped from matrix row ids to match spot ids.")
    args = parser.parse_args()
    return args


def read_gene_list(path: Path) -> list[str]:
    """Read marker gene names from a headerless, one-column TSV file."""
    genes: list[str] = []
    with path.open(mode="r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, fields in enumerate(reader, start=1):
            if not fields or not any(field.strip() for field in fields):
                continue
            if len(fields) != 1 or not fields[0].strip():
                raise ValueError(
                    f"Gene file must contain exactly one gene name per line: "
                    f"{path}:{line_number}"
                )
            genes.append(fields[0].strip())
    genes = list(dict.fromkeys(genes))
    if not genes:
        raise ValueError(f"Gene file contains no gene names: {path}")
    return genes


def read_gene_vmrs(path: Path, genes: list[str]) -> pd.DataFrame:
    """Return unique marker-gene/VMR relations in requested gene order."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    required = {"gene_name", "vmr", "relation"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(
            "VMR-to-gene table is missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    gene_order = {gene: index for index, gene in enumerate(genes)}
    pairs = df.loc[
        df["gene_name"].isin(genes), ["gene_name", "vmr", "relation"]
    ].drop_duplicates()
    return (
        pairs.assign(_gene_order=pairs["gene_name"].map(gene_order))
        .sort_values(["_gene_order", "vmr"])
        .drop(columns="_gene_order")
        .reset_index(drop=True)
    )


def matrix_column_map(header: list[str], vmrs: list[str]) -> dict[str, int]:
    """Map each requested VMR to its matrix column index."""
    column_by_vmr = {vmr: index for index, vmr in enumerate(header)}
    missing = [vmr for vmr in vmrs if vmr not in column_by_vmr]
    if missing:
        raise ValueError(
            f"{len(missing)} marker VMR(s) are absent from the matrix: "
            + ", ".join(missing[:5])
        )
    return {vmr: column_by_vmr[vmr] for vmr in vmrs}


def parse_spot(spot: str) -> tuple[int, int]:
    """Parse a DBiT spot label such as 04_35 into integer coordinates."""
    match = re.fullmatch(r"(?P<x>\d+)_(?P<y>\d+)", spot)
    if match is None:
        raise ValueError(f"Cannot parse X/Y coordinates from matrix spot {spot!r}")
    return int(match.group("x")), int(match.group("y"))


def collect_spots(
    path: Path,
    vmr_columns: dict[str, int],
    row_suffix: str,
) -> pd.DataFrame:
    """Read each selected VMR value and derive coordinates from matrix spot ids."""
    rows: list[dict[str, object]] = []
    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for line in reader:
            spot = line[0]
            if row_suffix and spot.endswith(row_suffix):
                spot = spot[: -len(row_suffix)]
            x_index, y_index = parse_spot(spot)
            record: dict[str, object] = {
                "spot": spot,
                "X_index": x_index,
                "Y_index": y_index,
            }
            for vmr, index in vmr_columns.items():
                raw = line[index].strip()
                record[vmr] = float(raw) if raw else np.nan
            rows.append(record)
    return pd.DataFrame(rows)


def plot_vmr_heatmap(
    path: Path,
    data: pd.DataFrame,
    vmr: str,
    genes: list[str],
    relations: list[str],
    chip_size: int,
) -> None:
    figure, axis = plt.subplots(figsize=(5, 5))
    x = data["X_index"].to_numpy()
    y = data["Y_index"].to_numpy()

    # Undetected spots are reported as 0 (low methylation) so the full chip
    # is colored; measured values overwrite the default.
    grid = np.zeros((chip_size, chip_size), dtype=float)
    values = data[vmr].to_numpy(dtype=float)
    for xi, yi, value in zip(x, y, values):
        if not np.isnan(value):
            grid[int(yi), int(xi)] = value
    image = axis.imshow(
        grid,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
    )
    gene_label = ", ".join(genes)
    relation_label = ", ".join(relation_display_name(item) for item in relations)
    title = f"{gene_label}\n{relation_label} | {vmr}"
    axis.set_title(title, fontsize=12, pad=4)
    axis.set_xticks([])
    axis.set_yticks([])
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def relation_display_name(relation: str) -> str:
    """Use a clearer display name for the gene-body relation."""
    return "gene body" if relation == "gene" else relation


def relation_slug(relations: list[str]) -> str:
    """Build a filesystem-safe label for one or more VMR relations."""
    labels = [
        re.sub(r"[^A-Za-z0-9_-]+", "_", relation_display_name(item))
        for item in relations
    ]
    return "_".join(labels)


def heatmap_filename(vmr: str, genes: list[str], relations: list[str]) -> str:
    """Build a readable filesystem-safe name for one VMR heatmap."""
    gene_label = "_".join(genes)
    relation_label = relation_slug(relations)
    region_label = vmr.replace(":", "_")
    return f"{gene_label}_{relation_label}_{region_label}.png"


def validate_spot_coordinates(data: pd.DataFrame, chip_size: int) -> None:
    """Require zero-based spot coordinates within the selected square chip."""
    outside = data[
        (data["X_index"] < 0)
        | (data["X_index"] >= chip_size)
        | (data["Y_index"] < 0)
        | (data["Y_index"] >= chip_size)
    ]
    if not outside.empty:
        examples = ", ".join(outside["spot"].astype(str).head(5))
        raise ValueError(
            f"{len(outside)} spot(s) fall outside the 0-{chip_size - 1} grid "
            f"for --chip-size {chip_size}: {examples}"
        )


def main() -> None:
    args = parse_args()
    gene_table = args.gene_table.expanduser().resolve()
    gene_file = args.gene_file.expanduser().resolve()
    matrix = args.matrix.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path, description in (
        (gene_table, "VMR-to-gene table"),
        (gene_file, "marker gene file"),
        (matrix, "methylation matrix"),
    ):
        if not path.is_file():
            raise SystemExit(f"{description} does not exist: {path}")

    genes = read_gene_list(gene_file)
    gene_vmr_pairs = read_gene_vmrs(gene_table, genes)
    found_genes = set(gene_vmr_pairs["gene_name"])
    missing = [gene for gene in genes if gene not in found_genes]
    if missing:
        raise SystemExit("Marker genes without any VMR: " + ", ".join(missing))

    vmr_genes: dict[str, list[str]] = {}
    vmr_relations: dict[str, list[str]] = {}
    for row in gene_vmr_pairs.itertuples(index=False):
        vmr_genes.setdefault(row.vmr, []).append(row.gene_name)
        relations = vmr_relations.setdefault(row.vmr, [])
        if row.relation not in relations:
            relations.append(row.relation)
    vmrs = list(vmr_genes)

    with gzip.open(matrix, mode="rt", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    vmr_columns = matrix_column_map(header, vmrs)

    spots = collect_spots(matrix, vmr_columns, args.row_suffix)
    if spots.empty:
        raise SystemExit("No spots found in the methylation matrix")
    validate_spot_coordinates(spots, args.chip_size)
    spots = spots.sort_values(["X_index", "Y_index"])

    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = output_dir / "vmr_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "marker_vmr_methylation_by_spot.tsv"
    annotation_path = output_dir / "marker_vmr_annotations.tsv"
    spots[["spot", "X_index", "Y_index"] + vmrs].to_csv(
        table_path, sep="\t", index=False
    )
    annotations = gene_vmr_pairs.copy()
    annotations["relation"] = annotations["relation"].map(relation_display_name)
    annotations.to_csv(annotation_path, sep="\t", index=False)
    for vmr, vmr_gene_names in vmr_genes.items():
        relations = vmr_relations[vmr]
        relation_dir = heatmap_dir / relation_slug(relations)
        relation_dir.mkdir(parents=True, exist_ok=True)
        figure_path = relation_dir / heatmap_filename(
            vmr, vmr_gene_names, relations
        )
        plot_vmr_heatmap(
            figure_path, spots, vmr, vmr_gene_names, relations, args.chip_size
        )

    print(f"Markers: {', '.join(genes)}")
    print(f"Chip: {args.chip_size} x {args.chip_size}")
    print(f"VMRs: {len(vmrs):,}")
    relation_counts = gene_vmr_pairs.groupby("relation")["vmr"].nunique()
    print(
        "VMRs by relation: "
        + ", ".join(
            f"{relation_display_name(relation)}={count:,}"
            for relation, count in relation_counts.items()
        )
    )
    print(f"Spots: {len(spots):,}")
    print(f"Table: {table_path}")
    print(f"Annotations: {annotation_path}")
    print(f"Heatmaps: {heatmap_dir}")


if __name__ == "__main__":
    main()
