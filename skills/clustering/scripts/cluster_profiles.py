"""
Biological profiling of clusters.

Generates composition tables, species/label enrichment, and mutation heatmaps.
This is the main biological validation step — internal metrics alone are insufficient.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Optional

# Sequential colormap: light green → dark green (low entropy → high entropy)
_CMAP_SEQ = LinearSegmentedColormap.from_list(
    "capo_seq", ["#C8DFD9", "#78B5B0", "#0E625C"]
)


def profile_clusters(
    df: pd.DataFrame,
    cluster_col: str,
    label_cols: list[str],
    seq_col: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> dict:
    """
    Generate per-cluster biological profiles.

    Returns dict containing:
        cluster_sizes:          DataFrame (cluster, size, frac)
        composition_<col>:      DataFrame (cluster × label value counts)
        enrichment_<col>:       DataFrame (cluster × label fractions)
        mutation_heatmap_path:  path to PNG (if seq_col provided)
    """
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    results: dict = {}

    # Cluster size table
    size_table = (
        df[cluster_col].value_counts()
        .rename("size")
        .reset_index()
    )
    size_table.columns = [cluster_col, "size"]
    size_table["frac"] = (size_table["size"] / len(df)).round(4)
    results["cluster_sizes"] = size_table
    if out_dir:
        size_table.to_csv(f"{out_dir}/cluster_sizes.csv", index=False)

    # Composition and enrichment per label column
    for col in label_cols:
        if col not in df.columns:
            continue
        comp = df.groupby([cluster_col, col]).size().unstack(fill_value=0)
        comp["total"] = comp.sum(axis=1)
        enrichment = comp.drop(columns="total").div(comp["total"], axis=0).round(4)
        results[f"composition_{col}"] = comp
        results[f"enrichment_{col}"] = enrichment
        if out_dir:
            comp.to_csv(f"{out_dir}/composition_{col}.csv")
            enrichment.to_csv(f"{out_dir}/enrichment_{col}.csv")

    # Per-position mutation / residue diversity heatmap
    if seq_col and seq_col in df.columns:
        heatmap_path = _mutation_heatmap(df, cluster_col, seq_col, out_dir)
        if heatmap_path:
            results["mutation_heatmap_path"] = heatmap_path

    return results


def _mutation_heatmap(
    df: pd.DataFrame,
    cluster_col: str,
    seq_col: str,
    out_dir: Optional[str],
    max_len: int = 50,
    top_clusters: int = 10,
) -> Optional[str]:
    """
    Per-position Shannon entropy heatmap for the top_clusters largest clusters.
    High entropy → variable position. Low entropy → conserved position.
    Plots the first max_len positions only.
    """
    from scipy.stats import entropy

    clusters = df[cluster_col].value_counts().head(top_clusters).index.tolist()
    entropy_matrix = []

    for cls in clusters:
        seqs = df.loc[df[cluster_col] == cls, seq_col].dropna().tolist()
        row = []
        for pos in range(max_len):
            chars = [s[pos] for s in seqs if len(s) > pos]
            if not chars:
                row.append(0.0)
            else:
                counts = pd.Series(chars).value_counts(normalize=True)
                row.append(float(entropy(counts.values, base=2)))
        entropy_matrix.append(row)

    fig, ax = plt.subplots(figsize=(max(8, max_len // 3), max(4, top_clusters)))
    im = ax.imshow(entropy_matrix, aspect="auto", cmap=_CMAP_SEQ, vmin=0)
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"Cluster {c}" for c in clusters], fontsize=8)
    ax.set_xlabel("Sequence position", fontsize=9)
    ax.set_title("Per-position residue entropy (bits) per cluster", fontsize=10)
    plt.colorbar(im, ax=ax, label="Shannon entropy (bits)")
    fig.tight_layout()

    if out_dir:
        path = f"{out_dir}/mutation_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    plt.show()
    plt.close(fig)
    return None
