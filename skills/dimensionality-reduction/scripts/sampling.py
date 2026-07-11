"""
Sampling utilities for large biological datasets.

Used before nonlinear reduction (t-SNE, UMAP) to keep runtimes manageable
and to ensure balanced representation of rare biological groups.
"""
import numpy as np
import pandas as pd
from typing import Optional


def random_sample(df: pd.DataFrame, n: int, seed: int = 42) -> np.ndarray:
    """Return an index array of n randomly sampled rows (without replacement)."""
    rng = np.random.default_rng(seed)
    n = min(n, len(df))
    return rng.choice(len(df), size=n, replace=False)


def stratified_sample(
    df: pd.DataFrame,
    label_col: str,
    n: int,
    seed: int = 42,
    min_per_class: int = 10,
) -> np.ndarray:
    """
    Return an index array of n rows, stratified by label_col.
    Each class gets at least min_per_class samples (if available).
    Proportional allocation beyond the minimum.

    Args:
        df:            DataFrame to sample from.
        label_col:     Column to stratify by (e.g. "species", "binding_label").
        n:             Target total sample size.
        seed:          Random seed.
        min_per_class: Minimum samples per class regardless of class frequency.
    """
    rng = np.random.default_rng(seed)
    total = len(df)
    indices: list[int] = []

    for cls in df[label_col].dropna().unique():
        cls_idx = df.index[df[label_col] == cls].tolist()
        cls_n = max(min_per_class, int(n * len(cls_idx) / total))
        cls_n = min(cls_n, len(cls_idx))
        sampled = rng.choice(cls_idx, size=cls_n, replace=False)
        indices.extend(sampled.tolist())

    # Deduplicate and trim to target n
    indices = list(set(indices))
    if len(indices) > n:
        indices = rng.choice(indices, size=n, replace=False).tolist()
    return np.array(indices)


def capped_sample(
    df: pd.DataFrame,
    label_col: str,
    cap_per_class: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Return an index array with at most cap_per_class samples per class.
    Useful when class imbalance would otherwise dominate the plot.
    """
    rng = np.random.default_rng(seed)
    indices: list[int] = []
    for cls in df[label_col].dropna().unique():
        cls_idx = df.index[df[label_col] == cls].tolist()
        n = min(cap_per_class, len(cls_idx))
        indices.extend(rng.choice(cls_idx, size=n, replace=False).tolist())
    return np.array(indices)
