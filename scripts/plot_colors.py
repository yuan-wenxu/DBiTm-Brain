"""Shared colors for numeric cluster IDs."""

from matplotlib import colormaps
from matplotlib.colors import to_hex


CLUSTER_COLORS = (
    "#A73030", "#E64B35", "#4DBBD5", "#CC79A7", "#7E57C2",
    "#C5A3E0", "#D6A500", "#FFE082", "#F28E2B", "#FF80B5",
    "#E7298A", "#9BC9FF", "#C49C94",
)


def cluster_color(cluster_id: int) -> str:
    """Return a cluster's color independently of the clusters plotted."""
    if cluster_id < len(CLUSTER_COLORS):
        return CLUSTER_COLORS[cluster_id]
    palette = colormaps["turbo"]
    # An odd step visits all 256 entries before the fallback colors repeat.
    offset = cluster_id - len(CLUSTER_COLORS)
    return to_hex(palette((64 + 97 * offset) % palette.N))


def cluster_colors(cluster_count: int) -> list[str]:
    return [cluster_color(cluster_id) for cluster_id in range(cluster_count)]
