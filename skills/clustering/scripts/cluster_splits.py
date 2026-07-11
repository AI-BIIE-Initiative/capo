"""
Cluster-aware train/val/test split assignment.

Assigns entire clusters to one split — never splits a cluster across sets.
This prevents homology leakage from sequences in the same embedding cluster
appearing in both train and test.

Use check_cluster_leakage() after assignment to verify.
"""
import numpy as np
import pandas as pd
from typing import Optional


def assign_cluster_splits(
    df: pd.DataFrame,
    cluster_col: str,
    ratios: tuple = (0.8, 0.1, 0.1),
    seed: int = 42,
    stratify_col: Optional[str] = None,
    noise_split: str = "train",
) -> pd.DataFrame:
    """
    Assign each cluster to train, val, or test. Entire clusters stay together.
    Noise points (cluster == -1) go to noise_split.

    Args:
        df:           DataFrame with cluster_col.
        cluster_col:  Name of cluster label column.
        ratios:       (train, val, test) fractions, must sum to 1.0.
        seed:         Random seed.
        stratify_col: If provided, sort clusters by majority class for balanced splits.
        noise_split:  Split to assign noise points (cluster = -1).

    Returns:
        Copy of df with added 'split' column ('train' | 'val' | 'test').
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, f"Split ratios must sum to 1.0, got {sum(ratios)}"
    rng = np.random.default_rng(seed)
    df = df.copy()

    noise_mask = df[cluster_col] == -1
    clusters = sorted(set(df.loc[~noise_mask, cluster_col].unique()))

    if stratify_col and stratify_col in df.columns:
        # Sort clusters by majority label value — helps interleave class distribution
        majority = (
            df[~noise_mask]
            .groupby(cluster_col)[stratify_col]
            .agg(lambda x: x.mode()[0] if len(x) > 0 else "")
        )
        clusters = majority.loc[clusters].sort_values().index.tolist()

    n = len(clusters)
    n_train = round(ratios[0] * n)
    n_val   = round(ratios[1] * n)
    perm = rng.permutation(n)
    train_clusters = set(np.array(clusters)[perm[:n_train]])
    val_clusters   = set(np.array(clusters)[perm[n_train : n_train + n_val]])
    test_clusters  = set(np.array(clusters)[perm[n_train + n_val :]])

    def _split(c):
        if c == -1:
            return noise_split
        if c in train_clusters:
            return "train"
        if c in val_clusters:
            return "val"
        return "test"

    df["split"] = df[cluster_col].map(_split)
    return df


def check_cluster_leakage(
    df: pd.DataFrame,
    seq_col: str,
    cluster_col: str,
    split_col: str = "split",
) -> dict:
    """
    Verify that no sequence appears in multiple splits and
    no non-noise cluster spans multiple splits.

    Returns leakage report dict. Zero for both counts is the target.
    """
    # Sequences in multiple splits
    seq_splits = df.groupby(seq_col)[split_col].nunique()
    seq_leakage = int((seq_splits > 1).sum())

    # Clusters spanning multiple splits (excluding noise)
    valid = df[df[cluster_col] != -1]
    cluster_split_counts = valid.groupby(cluster_col)[split_col].nunique()
    cluster_leakage = int((cluster_split_counts > 1).sum())

    return {
        "sequences_in_multiple_splits": seq_leakage,
        "clusters_spanning_splits":     cluster_leakage,
        "split_counts":                 df[split_col].value_counts().to_dict(),
    }
