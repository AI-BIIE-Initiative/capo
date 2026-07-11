"""
Clustering visualizations.

Requires 2D coordinates from the dimensionality-reduction skill
(outputs/dimred/reduced_coordinates.parquet).

Saves both PNG (dpi=150) and PDF copies of every figure.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Optional

# Canonical palette (autoimmunolab — do not substitute other colors)
_BLUE_0    = "#1E5994"
_ORANGE_0  = "#9B3208"
_PURPLE_0  = "#713D8F"
_GREEN_0   = "#0E625C"
_BLUE_50   = "#8DB8E2"
_ORANGE_50 = "#E6905B"
_PURPLE_50 = "#C694E1"
_GREEN_50  = "#78B5B0"
_BLUE_90   = "#BDD9F5"
_ORANGE_90 = "#FAC19E"
_PURPLE_90 = "#EAD5F6"
_GREEN_90  = "#C8DFD9"

_CLUSTER_PALETTE = [
    _BLUE_0, _ORANGE_0, _PURPLE_0, _GREEN_0,
    _BLUE_50, _ORANGE_50, _PURPLE_50, _GREEN_50,
    _BLUE_90, _ORANGE_90, _PURPLE_90, _GREEN_90,
    # second cycle (darker variants via manual override for >12 clusters)
    "#113356", "#5C1D04", "#3E1E52", "#083730",
    "#5A7EA0", "#A85E2E", "#8A559D", "#4A7F7B",
]
_NOISE_COLOR = "#AAAAAA"
_CMAP_STACKED = [
    _BLUE_0, _ORANGE_0, _PURPLE_0, _GREEN_0,
    _BLUE_50, _ORANGE_50, _PURPLE_50, _GREEN_50,
    _BLUE_90, _ORANGE_90, _PURPLE_90, _GREEN_90,
]


def _save(fig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(str(path).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_cluster_map(
    coords: "pd.DataFrame | np.ndarray",
    df: pd.DataFrame,
    cluster_col: str,
    out_path: str,
    color_col: Optional[str] = None,
    alpha: float = 0.4,
    point_size: float = 3.0,
    figsize: tuple = (12, 9),
) -> None:
    """
    Scatter plot coloured by cluster label (or color_col if provided).
    Noise points (cluster == -1) are rendered in grey in the background.

    Args:
        coords:      2D array or parquet DataFrame from dim-red skill.
        df:          DataFrame with cluster_col and metadata, aligned with coords.
        cluster_col: Column with cluster labels.
        out_path:    Output PNG path (PDF saved automatically).
        color_col:   Override colour column (defaults to cluster_col).
    """
    if isinstance(coords, pd.DataFrame):
        dim_cols = [c for c in coords.columns if c.startswith("dim_")][:2]
        xy = coords[dim_cols].values
    else:
        xy = coords[:, :2]

    col = color_col or cluster_col
    labels = df[col].values
    unique_labels = sorted(set(labels) - {-1}, key=str)

    fig, ax = plt.subplots(figsize=figsize)
    rasterized = len(xy) > 50_000

    # Noise points in the background
    noise_mask = df[cluster_col].values == -1
    if noise_mask.any():
        ax.scatter(
            xy[noise_mask, 0], xy[noise_mask, 1],
            c=_NOISE_COLOR, s=point_size, alpha=0.25,
            rasterized=rasterized, label="noise (-1)", linewidths=0, zorder=1,
        )

    for i, cls in enumerate(unique_labels):
        mask = labels == cls
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            c=_CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)],
            s=point_size, alpha=alpha,
            rasterized=rasterized, label=str(cls), linewidths=0, zorder=2,
        )

    handles, label_names = ax.get_legend_handles_labels()
    max_legend = 30
    ax.legend(
        handles[:max_legend], label_names[:max_legend],
        markerscale=3, fontsize=7,
        bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    n_clusters = len(unique_labels)
    n_noise = int(noise_mask.sum())
    ax.set_title(
        f"Coloured by {col} | {n_clusters} clusters | {n_noise:,} noise points",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, out_path)


def plot_cluster_profiles(
    profiles: dict,
    label_col: str,
    out_path: str,
    max_clusters: int = 30,
    figsize: tuple = (14, 6),
) -> None:
    """
    Stacked bar chart showing label composition fraction per cluster.
    Uses enrichment_<label_col> from profiles dict.
    """
    key = f"enrichment_{label_col}"
    if key not in profiles:
        return
    frac = profiles[key].head(max_clusters)
    n_cats = len(frac.columns)
    colors = [_CMAP_STACKED[i % len(_CMAP_STACKED)] for i in range(n_cats)]

    fig, ax = plt.subplots(figsize=figsize)
    frac.plot.bar(ax=ax, stacked=True, color=colors, alpha=0.85, width=0.8)
    ax.set_xlabel("Cluster", fontsize=9)
    ax.set_ylabel(f"Fraction ({label_col})", fontsize=9)
    ax.set_title(f"Cluster composition by {label_col}", fontsize=10)
    ax.legend(
        title=label_col,
        bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False,
    )
    fig.tight_layout()
    _save(fig, out_path)
