"""
Publication-quality 2D scatter plots for dimensionality reduction outputs.

Key design choices:
- Always saves both PNG (dpi=150) and PDF (vector) versions.
- Rasterizes points automatically for large point clouds (>50k points).
- Uses a fixed perceptually-distinct palette for consistent cross-run appearance.
- Supports secondary label via marker shape.
- Compresses legend for datasets with many classes.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# Canonical palette (autoimmunolab — do not substitute other colors)
_PALETTE = [
    "#1E5994", "#9B3208", "#713D8F", "#0E625C",   # dark: blue, orange, purple, green
    "#8DB8E2", "#E6905B", "#C694E1", "#78B5B0",   # mid:  blue, orange, purple, green
    "#BDD9F5", "#FAC19E", "#EAD5F6", "#C8DFD9",   # light: blue, orange, purple, green
]
_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "P"]


def _build_palette(series: pd.Series) -> dict:
    classes = sorted(series.dropna().unique(), key=str)
    return {cls: _PALETTE[i % len(_PALETTE)] for i, cls in enumerate(classes)}


def scatter_2d(
    coords: np.ndarray,
    metadata: pd.DataFrame,
    color_col: str,
    out_path: str,
    title: Optional[str] = None,
    marker_col: Optional[str] = None,
    alpha: float = 0.5,
    point_size: float = 4.0,
    figsize: tuple = (10, 8),
) -> None:
    """
    2D scatter plot coloured by color_col, optionally with marker_col for secondary label.
    Saves PNG and PDF to out_path.

    Args:
        coords:      Array of shape (n, 2).
        metadata:    DataFrame aligned with coords rows.
        color_col:   Column to colour by (e.g. "species", "binding_label").
        out_path:    Output PNG path. PDF saved at same path with .pdf extension.
        marker_col:  Optional secondary label encoded as marker shape.
        alpha:       Point transparency.
        point_size:  Point size in scatter.
        figsize:     Figure dimensions in inches.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rasterized = len(coords) > 50_000

    palette = _build_palette(metadata[color_col])
    marker_classes = (
        sorted(metadata[marker_col].dropna().unique(), key=str)
        if marker_col else [None]
    )

    fig, ax = plt.subplots(figsize=figsize)

    for cls, color in palette.items():
        base_mask = metadata[color_col] == cls
        for mi, mc in enumerate(marker_classes):
            mask = base_mask & (metadata[marker_col] == mc) if mc is not None else base_mask
            pts = coords[mask.values]
            if len(pts) == 0:
                continue
            label = f"{cls} / {mc}" if mc is not None else str(cls)
            ax.scatter(
                pts[:, 0], pts[:, 1],
                c=color, s=point_size, alpha=alpha,
                marker=_MARKERS[mi % len(_MARKERS)],
                label=label, linewidths=0,
                rasterized=rasterized,
            )

    handles, labels = ax.get_legend_handles_labels()
    max_legend = 25
    ax.legend(
        handles[:max_legend], labels[:max_legend],
        markerscale=3, fontsize=7,
        bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False,
    )
    if len(handles) > max_legend:
        ax.text(
            1.01, 0, f"… +{len(handles) - max_legend} more",
            transform=ax.transAxes, fontsize=7, va="bottom",
        )

    ax.set_xlabel("Dim 1", fontsize=10)
    ax.set_ylabel("Dim 2", fontsize=10)
    ax.set_title(title or f"Coloured by {color_col}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(str(out_path).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def scatter_2d_multi(
    coords: np.ndarray,
    metadata: pd.DataFrame,
    color_cols: list[str],
    out_dir: str,
    prefix: str = "scatter",
    **kwargs,
) -> dict[str, str]:
    """Generate one scatter plot per color_col. Returns {col: path} dict."""
    paths = {}
    for col in color_cols:
        if col not in metadata.columns:
            continue
        out_path = f"{out_dir}/{prefix}_{col}.png"
        scatter_2d(coords, metadata, color_col=col, out_path=out_path, **kwargs)
        paths[col] = out_path
    return paths
