"""Shared colors for numeric cluster IDs."""

from matplotlib import colormaps
from matplotlib.colors import to_hex


CLUSTER_COLORS = (
    "#A73030", "#E64B35", "#4DBBD5", "#BC8A34", "#F38C47",
    "#91A0E6", "#1696E9", "#FFAF69", "#F4AAA4", "#E489D2",
    "#F0BE38", "#C67F5A", "#BEBEFF", "#ED719A", "#9184F5",
    "#76CAF4", "#A57DCD", "#6A98BE", "#C18DF2", "#CE65C5",
    "#CB7A8B", "#E1A0D2", "#BE8CA5", "#F0918C", "#C9A568",
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
