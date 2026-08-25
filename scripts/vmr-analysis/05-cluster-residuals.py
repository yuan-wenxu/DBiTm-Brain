#!/usr/bin/env python3
"""Cluster spots from a sparse MethSCAn VMR residual matrix.

This implements the methylation-only branch of the spatial-DMT workflow:
iterative low-rank PCA imputation, PCA, an SNN graph, Leiden clustering, UMAP,
and projection of the clusters back onto the DBiT grid.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
from collections import defaultdict
from pathlib import Path
from typing import TextIO

import igraph as ig
import leidenalg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


SPOT_PATTERN = re.compile(r"^(?P<x>\d+)_(?P<y>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        required=True,
        help="Filtered MethSCAn mean_shrunken_residuals.csv.gz matrix.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--chip-size",
        type=int,
        choices=(50, 100),
        required=True,
        help="Number of zero-based grid positions along each chip axis.",
    )
    parser.add_argument(
        "--row-suffix",
        default=".CG",
        help="Suffix stripped from matrix row identifiers (default: .CG).",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=10,
        help="PCA rank/components for imputation and clustering (default: 10).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum iterative-PCA imputation rounds (default: 100).",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.01,
        help=(
            "Stop when imputation MSE divided by the largest observed MSE is "
            "below this value (default: 0.01)."
        ),
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=20,
        help="Nearest neighbors used for SNN and UMAP (default: 20).",
    )
    parser.add_argument(
        "--snn-prune",
        type=float,
        default=1 / 15,
        help="Discard SNN edges at or below this Jaccard weight (default: 1/15).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Leiden RB-configuration resolution (default: 1).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.3,
        help="UMAP minimum distance (default: 0.3).",
    )
    parser.add_argument(
        "--random-seed", type=int, default=0, help="Random seed (default: 0)."
    )
    args = parser.parse_args()

    if args.n_components < 2:
        parser.error("--n-components must be at least 2")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")
    if not 0 < args.min_gain < 1:
        parser.error("--min-gain must be greater than 0 and less than 1")
    if args.n_neighbors < 2:
        parser.error("--n-neighbors must be at least 2")
    if not 0 <= args.snn_prune < 1:
        parser.error("--snn-prune must be at least 0 and less than 1")
    if args.resolution <= 0:
        parser.error("--resolution must be greater than 0")
    if not 0 <= args.umap_min_dist <= 1:
        parser.error("--umap-min-dist must be between 0 and 1")
    return args


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open(mode="r", encoding="utf-8", newline="")


def canonical_spot(row_id: str, row_suffix: str) -> str:
    if row_suffix and row_id.endswith(row_suffix):
        return row_id[: -len(row_suffix)]
    return row_id


def parse_spot(spot: str) -> tuple[int, int]:
    match = SPOT_PATTERN.fullmatch(spot)
    if match is None:
        raise ValueError(f"Cannot parse zero-based X/Y coordinates from spot {spot!r}")
    return int(match.group("x")), int(match.group("y"))


def inspect_matrix(path: Path, row_suffix: str) -> tuple[list[str], list[str]]:
    """Validate matrix shape and return VMR and canonical spot identifiers."""
    with open_text(path) as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"Matrix is empty: {path}") from error
        if len(header) < 3:
            raise ValueError(f"Matrix must contain at least two VMR columns: {path}")
        vmrs = header[1:]
        if len(vmrs) != len(set(vmrs)):
            raise ValueError(f"Duplicate VMR columns in matrix: {path}")

        spots: list[str] = []
        seen_spots: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"Expected {len(header)} columns in {path} row {row_number}, "
                    f"found {len(row)}"
                )
            spot = canonical_spot(row[0], row_suffix)
            parse_spot(spot)
            if spot in seen_spots:
                raise ValueError(f"Duplicate canonical spot in matrix: {spot}")
            seen_spots.add(spot)
            spots.append(spot)
    if len(spots) < 3:
        raise ValueError(f"Matrix must contain at least three spots: {path}")
    return vmrs, spots


def load_matrix(
    path: Path, vmrs: list[str], spots: list[str], row_suffix: str
) -> np.ndarray:
    """Load the residual matrix as float32 with blanks represented by NaN."""
    values = np.empty((len(spots), len(vmrs)), dtype=np.float32)
    with open_text(path) as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header[1:] != vmrs:
            raise ValueError(f"Matrix header changed while loading: {path}")
        for row_index, row in enumerate(reader):
            spot = canonical_spot(row[0], row_suffix)
            if spot != spots[row_index]:
                raise ValueError(f"Matrix row order changed while loading: {path}")
            try:
                values[row_index] = np.fromiter(
                    (float(raw) if raw else np.nan for raw in row[1:]),
                    dtype=np.float32,
                    count=len(vmrs),
                )
            except ValueError as error:
                raise ValueError(
                    f"Invalid residual value in {path} row {row_index + 2}"
                ) from error
    nonfinite = ~np.isfinite(values) & ~np.isnan(values)
    if nonfinite.any():
        raise ValueError(f"Matrix contains infinite residual values: {path}")
    return values


def validate_coordinates(spots: list[str], chip_size: int) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray([parse_spot(spot) for spot in spots], dtype=int)
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    outside = (x < 0) | (x >= chip_size) | (y < 0) | (y >= chip_size)
    if outside.any():
        examples = ", ".join(np.asarray(spots)[outside][:5])
        raise ValueError(
            f"{int(outside.sum())} spot(s) fall outside the 0-{chip_size - 1} "
            f"grid for --chip-size {chip_size}: {examples}"
        )
    return x, y


def iterative_pca_impute(
    values: np.ndarray,
    n_components: int,
    max_iterations: int,
    min_gain: float,
    random_seed: int,
    block_size: int = 2048,
) -> tuple[np.ndarray, list[tuple[int, float, float]]]:
    """Impute missing residuals using the spatial-DMT iterative-PCA recipe."""
    missing = np.isnan(values)
    matrix = values.copy()
    matrix[missing] = 0.0
    history: list[tuple[int, float, float]] = []
    largest_mse = 0.0

    for iteration in range(1, max_iterations + 1):
        pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=random_seed,
        )
        scores = pca.fit_transform(matrix)
        squared_error = 0.0
        missing_count = 0

        for start in range(0, matrix.shape[1], block_size):
            end = min(start + block_size, matrix.shape[1])
            block_missing = missing[:, start:end]
            if not block_missing.any():
                continue
            # Match pr$x %*% t(pr$rotation): reconstruct the centered signal
            # without adding the PCA column means back.
            reconstruction = scores @ pca.components_[:, start:end]
            current = matrix[:, start:end]
            delta = reconstruction[block_missing] - current[block_missing]
            squared_error += float(np.dot(delta, delta))
            missing_count += int(delta.size)
            current[block_missing] = reconstruction[block_missing]

        mse = squared_error / missing_count if missing_count else 0.0
        largest_mse = max(largest_mse, mse)
        relative_mse = mse / largest_mse if largest_mse else 0.0
        history.append((iteration, mse, relative_mse))
        print(
            f"Iterative PCA {iteration}: relative_mse={relative_mse:.6f}",
            flush=True,
        )
        if iteration > 1 and relative_mse < min_gain:
            break
    return matrix, history


def final_pca(
    matrix: np.ndarray, n_components: int, random_seed: int
) -> np.ndarray:
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_seed,
    )
    return pca.fit_transform(matrix)


def nearest_neighbors(scores: np.ndarray, n_neighbors: int) -> np.ndarray:
    if n_neighbors >= len(scores):
        raise ValueError(
            f"--n-neighbors ({n_neighbors}) must be less than spots ({len(scores)})"
        )
    model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    model.fit(scores)
    indices = model.kneighbors(return_distance=False)
    result = np.empty((len(scores), n_neighbors), dtype=int)
    for index, neighbors in enumerate(indices):
        without_self = neighbors[neighbors != index]
        if len(without_self) < n_neighbors:
            raise ValueError(f"Could not obtain {n_neighbors} neighbors for spot {index}")
        result[index] = without_self[:n_neighbors]
    return result


def build_snn(
    neighbors: np.ndarray, prune: float
) -> ig.Graph:
    """Build a Seurat-like shared-nearest-neighbor graph with Jaccard weights."""
    memberships: list[set[int]] = []
    inverse: dict[int, list[int]] = defaultdict(list)
    for node, row in enumerate(neighbors):
        members = set(int(item) for item in row)
        members.add(node)
        memberships.append(members)
        for member in members:
            inverse[member].append(node)

    shared: dict[tuple[int, int], int] = defaultdict(int)
    for nodes in inverse.values():
        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1 :]:
                pair = (left, right) if left < right else (right, left)
                shared[pair] += 1

    edge_rows: list[tuple[int, int, float]] = []
    for (left, right), intersection in shared.items():
        union = len(memberships[left]) + len(memberships[right]) - intersection
        weight = intersection / union
        if weight > prune:
            edge_rows.append((left, right, weight))
    if not edge_rows:
        raise ValueError(f"SNN graph has no edges after pruning at {prune:g}")

    graph = ig.Graph(
        n=len(neighbors),
        edges=[(left, right) for left, right, _ in edge_rows],
        directed=False,
    )
    graph.es["weight"] = [weight for _, _, weight in edge_rows]
    return graph


def leiden_clusters(
    graph: ig.Graph, resolution: float, random_seed: int
) -> np.ndarray:
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=-1,
        seed=random_seed,
    )
    membership = np.asarray(partition.membership, dtype=int)
    communities = []
    for raw_cluster in sorted(set(membership)):
        members = np.flatnonzero(membership == raw_cluster)
        communities.append((raw_cluster, len(members), int(members.min())))
    communities.sort(key=lambda item: (-item[1], item[2]))
    remap = {raw: new for new, (raw, _, _) in enumerate(communities)}
    stable = np.asarray([remap[item] for item in membership], dtype=int)
    return stable


def run_umap(
    scores: np.ndarray, n_neighbors: int, min_dist: float, random_seed: int
) -> np.ndarray:
    model = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        metric="euclidean",
        min_dist=min_dist,
        random_state=random_seed,
        transform_seed=random_seed,
        n_jobs=1,
    )
    return model.fit_transform(scores)


def cluster_colors(cluster_count: int) -> list[tuple[float, float, float, float]]:
    base = plt.get_cmap("tab20")
    if cluster_count <= 20:
        return [base(index) for index in range(cluster_count)]
    turbo = plt.get_cmap("turbo")
    return [turbo(index / max(1, cluster_count - 1)) for index in range(cluster_count)]


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open(mode="w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_overview(
    path: Path,
    embedding: np.ndarray,
    clusters: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    chip_size: int,
) -> None:
    cluster_count = int(clusters.max()) + 1
    colors = cluster_colors(cluster_count)
    color_values = [colors[item] for item in clusters]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].scatter(
        embedding[:, 0], embedding[:, 1], c=color_values, s=13, linewidths=0
    )
    axes[0].set_title("UMAP of imputed VMR residuals")
    axes[0].set_xlabel("UMAP1")
    axes[0].set_ylabel("UMAP2")

    spatial = axes[1].scatter(
        x, y, c=color_values, s=22, marker="s", linewidths=0
    )
    spatial.set_clip_on(False)
    axes[1].set_xlim(-0.5, chip_size - 0.5)
    axes[1].set_ylim(-0.5, chip_size - 0.5)
    axes[1].set_aspect("equal")
    axes[1].set_title("Leiden clusters on DBiT grid")
    axes[1].set_xlabel("X index")
    axes[1].set_ylabel("Y index")

    figure.suptitle("MethSCAn residual clustering", fontsize=14)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_imputation_history(
    path: Path,
    history: list[tuple[int, float, float]],
    min_gain: float,
) -> None:
    iterations = [row[0] for row in history]
    mse = [row[1] for row in history]
    relative_mse = [row[2] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    axes[0].plot(iterations, mse, marker="o", color="#3C5488")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("MSE on updated missing entries")
    axes[0].set_title("Iterative-PCA imputation")

    axes[1].plot(iterations, relative_mse, marker="o", color="#00A087")
    axes[1].axhline(min_gain, color="#E64B35", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("MSE / maximum MSE")
    axes[1].set_title("Relative convergence")

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    matrix_path = args.matrix.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not matrix_path.is_file():
        raise SystemExit(f"Residual matrix does not exist: {matrix_path}")

    try:
        vmrs, spots = inspect_matrix(matrix_path, args.row_suffix)
        if args.n_components >= min(len(spots), len(vmrs)):
            raise ValueError(
                "--n-components must be smaller than both the spot and VMR counts"
            )
        x, y = validate_coordinates(spots, args.chip_size)
        print(
            f"Loading {len(spots):,} spots x {len(vmrs):,} VMRs...", flush=True
        )
        values = load_matrix(matrix_path, vmrs, spots, args.row_suffix)
        imputed, history = iterative_pca_impute(
            values,
            args.n_components,
            args.max_iterations,
            args.min_gain,
            args.random_seed,
        )
        del values
        scores = final_pca(imputed, args.n_components, args.random_seed)
        del imputed
        neighbors = nearest_neighbors(scores, args.n_neighbors)
        graph = build_snn(neighbors, args.snn_prune)
        clusters = leiden_clusters(graph, args.resolution, args.random_seed)
        embedding = run_umap(
            scores, args.n_neighbors, args.umap_min_dist, args.random_seed
        )
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        raise SystemExit(str(error)) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "residual_cluster_assignments.tsv"
    figure_path = output_dir / "residual_clustering.png"
    convergence_path = output_dir / "iterative_pca_convergence.png"

    assignment_fields = [
        "spot",
        "X_index",
        "Y_index",
        "cluster",
        "UMAP1",
        "UMAP2",
    ]
    assignment_rows = []
    for index, spot in enumerate(spots):
        row: dict[str, object] = {
            "spot": spot,
            "X_index": int(x[index]),
            "Y_index": int(y[index]),
            "cluster": f"D{clusters[index]}",
            "UMAP1": float(embedding[index, 0]),
            "UMAP2": float(embedding[index, 1]),
        }
        assignment_rows.append(row)
    write_tsv(assignments_path, assignment_rows, assignment_fields)
    cluster_count = int(clusters.max()) + 1
    plot_overview(
        figure_path,
        embedding,
        clusters,
        x,
        y,
        args.chip_size,
    )
    plot_imputation_history(convergence_path, history, args.min_gain)

    print(f"Clusters: {cluster_count}")
    print(f"Assignments: {assignments_path}")
    print(f"Figure: {figure_path}")
    print(f"PCA convergence: {convergence_path}")


if __name__ == "__main__":
    main()
