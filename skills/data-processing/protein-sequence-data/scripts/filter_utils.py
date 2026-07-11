"""
filter_utils.py — Sequence-level filtering and deduplication.

Used by protein-sequence-data-processing skill (Steps 3, 6–8).

Every function returns (filtered_df, n_removed) or (filtered_df, report_dict)
so the caller can record the delta in the manifest without extra bookkeeping.
"""

from __future__ import annotations

import pandas as pd

from .sequence_utils import (
    has_only_valid_aas,
    has_internal_stop,
    normalize_aa,
    strip_terminal_stop,
)


# ---------------------------------------------------------------------------
# Sequence validity filters
# ---------------------------------------------------------------------------

def filter_invalid_sequences(
    df: pd.DataFrame,
    seq_col: str,
    allow_ambiguous: bool = True,
    allow_terminal_stop: bool = True,
) -> tuple[pd.DataFrame, int]:
    """
    Remove rows where the sequence contains non-amino-acid characters.

    allow_ambiguous      — keep B, X, Z, U, O characters (default True)
    allow_terminal_stop  — strip trailing '*' before validation (default True)

    Returns (filtered_df, n_removed).
    """
    df = df.copy()
    seqs = df[seq_col].copy()
    if allow_terminal_stop:
        seqs = seqs.map(strip_terminal_stop)

    valid_mask = seqs.map(
        lambda s: has_only_valid_aas(s, allow_ambiguous=allow_ambiguous)
        if isinstance(s, str) else False
    )
    n_removed = int((~valid_mask).sum())
    return df[valid_mask].copy(), n_removed


def filter_internal_stops(
    df: pd.DataFrame,
    seq_col: str,
) -> tuple[pd.DataFrame, int]:
    """
    Remove rows containing internal stop codons ('*' not at the terminal position).
    Must be called AFTER strip_terminal_stop has been applied.

    Returns (filtered_df, n_removed).
    """
    mask = df[seq_col].map(lambda s: not has_internal_stop(s) if isinstance(s, str) else True)
    n_removed = int((~mask).sum())
    return df[mask].copy(), n_removed


def filter_null_sequences(
    df: pd.DataFrame,
    seq_col: str,
) -> tuple[pd.DataFrame, int]:
    """
    Remove rows with null or empty sequences.
    Returns (filtered_df, n_removed).
    """
    mask = df[seq_col].notna() & (df[seq_col].str.len() > 0)
    n_removed = int((~mask).sum())
    return df[mask].copy(), n_removed


# ---------------------------------------------------------------------------
# Domain-specific filters
# ---------------------------------------------------------------------------

def filter_wt(
    df: pd.DataFrame,
    seq_col: str,
    wt_sequence: str,
    normalize: bool = True,
) -> tuple[pd.DataFrame, int]:
    """
    Remove rows whose sequence exactly matches the wild-type reference sequence.

    normalize=True — apply normalize_aa + strip_terminal_stop to wt_sequence before
                     comparison (recommended; ensures consistent comparison).

    Returns (filtered_df, n_removed).

    Note: For DMS datasets, do NOT call this — the WT row is often a useful baseline.
    Only use for datasets where the WT is explicitly excluded from the task.
    """
    wt = normalize_aa(wt_sequence) or wt_sequence
    if normalize:
        wt = strip_terminal_stop(wt) or wt

    mask = df[seq_col] != wt
    n_removed = int((~mask).sum())
    return df[mask].copy(), n_removed


def filter_by_length(
    df: pd.DataFrame,
    seq_col: str,
    min_len: int | None = None,
    max_len: int | None = None,
) -> tuple[pd.DataFrame, int]:
    """
    Filter sequences by length range [min_len, max_len] (inclusive on both ends).
    Pass None to skip either bound.

    Common use cases:
    - max_len=1024 for ESM2 compatibility
    - min_len=50 to remove very short fragments

    Returns (filtered_df, n_removed).
    """
    lengths = df[seq_col].str.len()
    mask = pd.Series([True] * len(df), index=df.index)
    if min_len is not None:
        mask &= lengths >= min_len
    if max_len is not None:
        mask &= lengths <= max_len
    n_removed = int((~mask).sum())
    return df[mask].copy(), n_removed


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(
    df: pd.DataFrame,
    seq_col: str,
    id_col: str | None = None,
    keep: str = "first",
) -> tuple[pd.DataFrame, dict]:
    """
    Remove exact-duplicate sequences.

    keep: 'first' | 'last' — which duplicate to retain (default 'first').

    Returns (deduped_df, report) where report contains:
      n_exact_dup_rows   — rows removed
      n_unique_sequences — unique sequences before dedup
      n_dup_sequences    — sequences that appeared more than once
      n_dup_ids          — duplicate IDs (if id_col provided)

    Always run deduplication BEFORE split assignment.
    """
    n_before = len(df)
    n_unique_seqs = df[seq_col].nunique()
    n_dup_seqs = int((df[seq_col].value_counts() > 1).sum())

    n_dup_ids = 0
    if id_col and id_col in df.columns:
        n_dup_ids = int(df[id_col].duplicated().sum())

    df = df.drop_duplicates(subset=[seq_col], keep=keep).copy()
    n_removed = n_before - len(df)

    report = {
        "n_exact_dup_rows":   n_removed,
        "n_unique_sequences": n_unique_seqs,
        "n_dup_sequences":    n_dup_seqs,
        "n_dup_ids":          n_dup_ids,
    }
    return df, report
