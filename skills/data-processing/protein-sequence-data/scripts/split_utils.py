"""
split_utils.py — Train/val/test split creation and leakage auditing.

Used by protein-sequence-data-processing skill (Step 9).

Key rules:
  - Always deduplicate BEFORE calling any split function
  - For homology-aware / clustered splits, refer to skills/clustering/ — do not implement here
  - check_leakage is mandatory — run even when count is expected to be 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Random split
# ---------------------------------------------------------------------------

def random_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    stratify_col: str | None = None,
) -> pd.DataFrame:
    """
    Add a 'split' column ('train' / 'val' / 'test') to df.

    ratios        — (train, val, test) must sum to 1.0
    stratify_col  — if provided, preserves class proportions across splits
                    using sklearn.model_selection.train_test_split

    Returns df with 'split' column added.
    Raises ValueError if 'split' column already exists with values.

    For homology-aware clustered splits, use skills/clustering/ instead.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios)}"

    if "split" in df.columns and df["split"].notna().any():
        raise ValueError(
            "DataFrame already has a 'split' column with values. "
            "Pass a copy with the split column dropped to override."
        )

    df = df.copy()
    rng = np.random.default_rng(seed)

    if stratify_col and stratify_col in df.columns:
        return _stratified_split(df, ratios, stratify_col, rng)

    idx = df.index.tolist()
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])

    split_map: dict = {}
    for i in idx[:n_train]:
        split_map[i] = "train"
    for i in idx[n_train:n_train + n_val]:
        split_map[i] = "val"
    for i in idx[n_train + n_val:]:
        split_map[i] = "test"

    df["split"] = df.index.map(split_map)
    return df


def _stratified_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float],
    stratify_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Internal: stratified split using sklearn if available, else falls back to random."""
    try:
        from sklearn.model_selection import train_test_split  # type: ignore

        labels = df[stratify_col].fillna("__missing__")
        train_val_idx, test_idx = train_test_split(
            df.index, test_size=ratios[2], stratify=labels,
            random_state=int(rng.integers(0, 2**31)),
        )
        train_val_labels = labels.loc[train_val_idx]
        val_ratio_adjusted = ratios[1] / (ratios[0] + ratios[1])
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_ratio_adjusted,
            stratify=train_val_labels,
            random_state=int(rng.integers(0, 2**31)),
        )
    except ImportError:
        import warnings
        warnings.warn("sklearn not available; falling back to non-stratified random_split.")
        return random_split(df, ratios, seed=int(rng.integers(0, 2**31)))

    df = df.copy()
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx,   "split"] = "val"
    df.loc[test_idx,  "split"] = "test"
    return df


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def check_leakage(
    df: pd.DataFrame,
    seq_col: str,
    split_col: str = "split",
) -> dict:
    """
    Detect exact-duplicate sequences appearing in more than one split.

    Always call this after split assignment — report the result even when count is 0.

    Returns:
    {
        "exact_duplicates_across_splits": int,
        "affected_sequences": list[str],   # up to 20 examples
        "per_split_pair": dict             # {"train|test": int, "train|val": int, ...}
    }

    Note: this checks exact-match leakage only. For homology leakage detection,
    refer to skills/clustering/.
    """
    if split_col not in df.columns:
        return {
            "exact_duplicates_across_splits": 0,
            "affected_sequences": [],
            "per_split_pair": {},
            "warning": f"No '{split_col}' column found",
        }

    splits = df[split_col].dropna().unique()
    split_seqs: dict[str, set] = {
        s: set(df.loc[df[split_col] == s, seq_col].dropna())
        for s in splits
    }

    affected: set[str] = set()
    per_pair: dict[str, int] = {}

    split_list = sorted(splits)
    for i, s1 in enumerate(split_list):
        for s2 in split_list[i + 1:]:
            overlap = split_seqs.get(s1, set()) & split_seqs.get(s2, set())
            per_pair[f"{s1}|{s2}"] = len(overlap)
            affected |= overlap

    return {
        "exact_duplicates_across_splits": len(affected),
        "affected_sequences":             sorted(affected)[:20],
        "per_split_pair":                 per_pair,
    }
