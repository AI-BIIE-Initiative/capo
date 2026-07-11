"""
label_utils.py — Label inference, mask creation, and conflict resolution.

Used by protein-sequence-data-processing skill (Steps 4–5).

Key design rules:
  - Never convert NaN/missing to 0/negative — always use label_mask_* columns
  - Conflicts = same sequence with both positive AND negative label in the same column
  - Multi-target datasets resolve conflicts per-column independently
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Label type detection
# ---------------------------------------------------------------------------

_POSITIVE_VALS:  frozenset = frozenset({1, 1.0, True, "1", "1.0", "binder",
                                         "positive", "pos", "+", "yes", "active", "hit"})
_NEGATIVE_VALS:  frozenset = frozenset({0, 0.0, False, "0", "0.0", "non-binder",
                                         "nonbinder", "negative", "neg", "-", "no",
                                         "inactive", "non_binder"})


def is_binder(val) -> bool:
    """Return True if val represents a positive / binding label."""
    if pd.isna(val):
        return False
    return str(val).strip().lower() in {str(v).lower() for v in _POSITIVE_VALS}


def is_nonbinder(val) -> bool:
    """Return True if val represents a negative / non-binding label."""
    if pd.isna(val):
        return False
    return str(val).strip().lower() in {str(v).lower() for v in _NEGATIVE_VALS}


def detect_label_type(series: pd.Series) -> str:
    """
    Infer label type from observed values.
    Returns 'binary' | 'regression' | 'multiclass' | 'unknown'.
    """
    clean = series.dropna()
    if clean.empty:
        return "unknown"

    unique = set(clean.unique())

    # Binary: only 0/1 or binder/non-binder equivalents
    if all(is_binder(v) or is_nonbinder(v) for v in unique):
        return "binary"

    # Regression: numeric with many unique values
    try:
        numeric = pd.to_numeric(clean, errors="raise")
        if numeric.nunique() > 10:
            return "regression"
        else:
            return "multiclass"
    except (ValueError, TypeError):
        pass

    # Multiclass: string categories
    if clean.dtype == object and clean.nunique() <= 50:
        return "multiclass"

    return "unknown"


# ---------------------------------------------------------------------------
# Label column inference
# ---------------------------------------------------------------------------

_LABEL_COL_NAMES = (
    "label", "score", "fitness", "y", "target", "activity", "binding",
    "ddg", "delta_g", "tm", "delta_tm", "kd", "affinity",
    "expression", "solubility", "stability", "localization",
)


def infer_label_columns(df: pd.DataFrame, profile: dict | None = None) -> list[str]:
    """
    Infer label columns.
    Priority: profile.label_info.target_columns > known col names > numeric non-id cols.
    """
    if profile:
        target_cols = (profile.get("label_info") or {}).get("target_columns", [])
        if target_cols:
            return [c for c in target_cols if c in df.columns]

    # Known names
    found = [c for c in _LABEL_COL_NAMES if c in df.columns]
    if found:
        return found

    # Heuristic: numeric columns that aren't length/id-like
    exclude = {"length", "seq_id", "id", "index"}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return [c for c in numeric_cols if c not in exclude and df[c].nunique() > 1]


# ---------------------------------------------------------------------------
# Label masks
# ---------------------------------------------------------------------------

def create_label_mask(series: pd.Series) -> pd.Series:
    """
    Return integer mask: 1 where label is observed (non-null), 0 where missing.
    This preserves missingness without converting NaN to negatives.
    """
    return series.notna().astype(int)


# ---------------------------------------------------------------------------
# Conflict detection and resolution
# ---------------------------------------------------------------------------

def find_conflicts(
    df: pd.DataFrame,
    seq_col: str,
    label_col: str,
) -> set:
    """
    Return set of sequences that have BOTH a positive and negative label
    in the same column (intra-column conflict).
    """
    binders    = set(df.loc[df[label_col].map(is_binder),    seq_col])
    nonbinders = set(df.loc[df[label_col].map(is_nonbinder), seq_col])
    return binders & nonbinders


def check_label_conflicts_multi(
    df: pd.DataFrame,
    seq_col: str,
    label_cols: list[str],
) -> dict[str, set]:
    """
    For multi-target datasets (e.g. per-animal ACE2 columns):
    return {col: set_of_conflicting_sequences} for every label column.
    Call this before resolve_conflicts to understand scope.
    """
    return {col: find_conflicts(df, seq_col, col) for col in label_cols}


def resolve_conflicts(
    df: pd.DataFrame,
    seq_col: str,
    label_cols: list[str],
    drop_globally: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Resolve intra-column label conflicts.

    drop_globally=True  — remove ALL rows where seq appears in any conflict set
    drop_globally=False — set conflicting label cells to NaN (preserve rows; mask handles it)

    Returns (cleaned_df, conflict_report).
    conflict_report = {col: {"n_conflicting_seqs": int, "conflicting_seqs": list}}
    """
    df = df.copy()
    report: dict = {}
    all_conflicting: set = set()

    for col in label_cols:
        conflicts = find_conflicts(df, seq_col, col)
        report[col] = {
            "n_conflicting_seqs": len(conflicts),
            "conflicting_seqs":   sorted(conflicts)[:20],  # cap for report size
        }
        all_conflicting |= conflicts

        if not drop_globally and conflicts:
            # Nullify conflicting labels per-column, keep the row
            mask = df[seq_col].isin(conflicts)
            df.loc[mask, col] = np.nan

    n_before = len(df)
    if drop_globally and all_conflicting:
        df = df[~df[seq_col].isin(all_conflicting)].copy()

    report["_summary"] = {
        "drop_globally":       drop_globally,
        "total_conflicting":   len(all_conflicting),
        "n_rows_removed":      n_before - len(df),
    }
    return df, report
