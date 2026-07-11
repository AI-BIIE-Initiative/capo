"""
sequence_utils.py — Core sequence normalization and validation utilities.

Used by protein-sequence-data-processing skill (Step 1–2).
"""

from __future__ import annotations

import re
import pandas as pd

# Standard 20 AA + common ambiguity/special symbols
STANDARD_AA:   frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")
AMBIGUOUS_AA:  frozenset[str] = frozenset("BXZUO")   # B=Asx, X=unknown, Z=Glx, U=Sec, O=Pyl
STOP_CODON:    str = "*"
VALID_AA:      frozenset[str] = STANDARD_AA | AMBIGUOUS_AA | {STOP_CODON}
GAP_CHARS:     frozenset[str] = frozenset("-.")


# ---------------------------------------------------------------------------
# Sequence cleaning
# ---------------------------------------------------------------------------

def normalize_aa(seq: str | None) -> str | None:
    """Strip surrounding whitespace, remove internal whitespace, uppercase."""
    if seq is None or (isinstance(seq, float)):
        return None
    return re.sub(r"\s+", "", str(seq)).upper()


def strip_terminal_stop(seq: str | None) -> str | None:
    """Remove a single trailing stop codon '*'. Does not touch internal stops."""
    if seq is None:
        return None
    return seq[:-1] if seq.endswith(STOP_CODON) else seq


def remove_gaps(seq: str | None) -> str | None:
    """Remove gap/alignment characters ('-', '.'). Call only when explicitly instructed."""
    if seq is None:
        return None
    return re.sub(r"[-.]", "", seq)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def has_only_valid_aas(seq: str | None, allow_ambiguous: bool = True) -> bool:
    """Return True if every character in seq is a valid amino acid."""
    if not seq:
        return False
    allowed = STANDARD_AA | (AMBIGUOUS_AA if allow_ambiguous else frozenset())
    return all(c in allowed for c in seq)


def detect_non_standard(seq: str | None, allow_ambiguous: bool = True) -> list[str]:
    """Return sorted list of non-standard characters found in seq."""
    if not seq:
        return []
    allowed = STANDARD_AA | (AMBIGUOUS_AA if allow_ambiguous else frozenset())
    return sorted(set(c for c in seq if c not in allowed))


def has_internal_stop(seq: str | None) -> bool:
    """Return True if seq contains '*' that is not the terminal character."""
    if not seq:
        return False
    return STOP_CODON in seq[:-1]


# ---------------------------------------------------------------------------
# Length stats
# ---------------------------------------------------------------------------

def compute_length_stats(lengths: pd.Series) -> dict:
    """Compute length distribution including ESM2 overflow percentages."""
    lengths = lengths.dropna()
    if lengths.empty:
        return {}
    return {
        "min":           int(lengths.min()),
        "median":        float(lengths.median()),
        "mean":          float(lengths.mean()),
        "max":           int(lengths.max()),
        "std":           float(lengths.std()),
        "pct_over_512":  float((lengths > 512).mean() * 100),
        "pct_over_1024": float((lengths > 1024).mean() * 100),
    }


# ---------------------------------------------------------------------------
# Schema normalization
# ---------------------------------------------------------------------------

def normalize_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names, strip whitespace, replace spaces/dashes with underscore."""
    df = df.copy()
    df.columns = [
        re.sub(r"[\s\-]+", "_", c.strip().lower())
        for c in df.columns
    ]
    return df


_SEQ_COL_NAMES = (
    "sequence", "seq", "aa_seq", "protein_sequence", "protein", "prot_seq",
    "peptide", "peptide_sequence", "mut_sequence", "mutant",
)
_ID_COL_NAMES = (
    "seq_id", "id", "sequence_id", "protein_id", "entry", "accession", "name", "uid",
)


def infer_seq_col(df: pd.DataFrame, profile: dict | None = None) -> str:
    """
    Infer sequence column name.
    Priority: profile.schema > known column names > content heuristic.
    Raises ValueError if no candidate or multiple ambiguous candidates.
    """
    # 1. From profile
    if profile:
        schema = profile.get("schema", [])
        for field in schema:
            if field.get("name") in df.columns and field.get("dtype") == "string":
                name = field["name"]
                if name in _SEQ_COL_NAMES or _looks_like_sequences(df[name]):
                    return name

    # 2. Known names
    for name in _SEQ_COL_NAMES:
        if name in df.columns:
            return name

    # 3. Content heuristic
    candidates = [
        col for col in df.columns
        if df[col].dtype == object and _looks_like_sequences(df[col])
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple candidate sequence columns: {candidates}. "
            "Specify the column explicitly."
        )
    raise ValueError("No sequence column could be identified.")


def infer_id_col(df: pd.DataFrame, profile: dict | None = None) -> str | None:
    """Infer identifier column. Returns None if not found (will generate synthetic IDs)."""
    for name in _ID_COL_NAMES:
        if name in df.columns:
            return name
    return None


def _looks_like_sequences(series: pd.Series, sample_n: int = 50) -> bool:
    """Heuristic: does this column look like amino-acid sequences?"""
    sample = series.dropna().head(sample_n).astype(str)
    if sample.empty:
        return False
    pct_aa = sample.map(
        lambda s: sum(c in VALID_AA for c in s.upper()) / max(len(s), 1)
    ).mean()
    return pct_aa > 0.85 and sample.str.len().median() >= 5
