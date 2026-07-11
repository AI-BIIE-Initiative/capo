# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=2.14",
#     "huggingface-hub[hf_transfer]>=0.20",
#     "hf-xet>=1.1.7",
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "scikit-learn>=1.3",
# ]
# ///
"""
Clean and format a protein sequence dataset from HuggingFace Hub.

Pipeline: normalize → filter (null / invalid AA / internal stops / length) →
          deduplicate → train/val/test split → push to Hub.

Example:
    uv run preprocess-protein-dataset.py \\
        owner/my-protein-dataset \\
        owner/my-protein-dataset-clean \\
        --seq-col sequence --label-col label --max-len 1024

    uv run preprocess-protein-dataset.py \\
        owner/raw-antibodies owner/antibodies-clean \\
        --min-len 50 --train 0.8 --val 0.1 --test 0.1
"""

import argparse
import logging
import re
from datetime import datetime

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import DatasetCard, get_token, login

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sequence utilities (inlined from data-processing/protein-sequence-data)
# ---------------------------------------------------------------------------

_STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
_AMBIGUOUS_AA = frozenset("BXZUO")

def _clean(s) -> str | None:
    """Normalize + strip terminal stop codon."""
    if not isinstance(s, str):
        return None
    s = re.sub(r"\s+", "", s).upper()
    return s[:-1] if s.endswith("*") else s

def _valid(s, allow_ambiguous=True) -> bool:
    if not s:
        return False
    ok = _STANDARD_AA | (_AMBIGUOUS_AA if allow_ambiguous else frozenset())
    return all(c in ok for c in s)

def _has_internal_stop(s) -> bool:
    return bool(s) and "*" in s[:-1]

def _infer_seq_col(df: pd.DataFrame) -> str:
    known = ("sequence", "seq", "aa_seq", "protein_sequence", "protein", "prot_seq", "peptide")
    for name in known:
        if name in df.columns:
            return name
    candidates = [
        c for c in df.columns
        if df[c].dtype == object and df[c].dropna().head(20).map(
            lambda s: sum(ch in _STANDARD_AA | _AMBIGUOUS_AA for ch in str(s).upper()) / max(len(str(s)), 1)
        ).mean() > 0.85
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Cannot infer sequence column. Found candidates: {candidates}. Use --seq-col.")

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, seq_col: str, min_len: int | None, max_len: int | None) -> tuple[pd.DataFrame, dict]:
    manifest: dict = {"n_start": len(df)}

    # Normalize
    df[seq_col] = df[seq_col].map(_clean)

    # Filter null / empty
    mask = df[seq_col].notna() & (df[seq_col].str.len() > 0)
    manifest["n_null_removed"] = int((~mask).sum())
    df = df[mask].copy()

    # Filter invalid AA
    mask = df[seq_col].map(lambda s: _valid(s))
    manifest["n_invalid_aa_removed"] = int((~mask).sum())
    df = df[mask].copy()

    # Filter internal stops
    mask = df[seq_col].map(lambda s: not _has_internal_stop(s))
    manifest["n_internal_stop_removed"] = int((~mask).sum())
    df = df[mask].copy()

    # Filter by length
    lengths = df[seq_col].str.len()
    mask = pd.Series(True, index=df.index)
    if min_len is not None:
        mask &= lengths >= min_len
    if max_len is not None:
        mask &= lengths <= max_len
    manifest["n_length_removed"] = int((~mask).sum())
    df = df[mask].copy()

    # Deduplicate (before split)
    n_before = len(df)
    df = df.drop_duplicates(subset=[seq_col], keep="first").copy()
    manifest["n_duplicates_removed"] = n_before - len(df)

    manifest["n_final"] = len(df)
    return df, manifest


def assign_splits(df: pd.DataFrame, ratios: tuple[float, float, float], seed: int, stratify_col: str | None) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(seed)
    idx = df.index.to_numpy().copy()
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    splits = (["train"] * n_train) + (["val"] * n_val) + (["test"] * (n - n_train - n_val))
    df.loc[idx, "split"] = splits

    # Leakage check (exact-match)
    split_seqs = {s: set(df.loc[df["split"] == s, df.columns[0]]) for s in ("train", "val", "test")}
    leakage = len(split_seqs["train"] & split_seqs["test"])
    if leakage:
        logger.warning(f"Exact-sequence leakage: {leakage} sequences appear in both train and test.")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess a protein sequence dataset and push to Hub.")
    parser.add_argument("input_dataset",  help="HF Hub dataset ID to load (e.g. owner/dataset)")
    parser.add_argument("output_repo",    help="HF Hub repo to push cleaned dataset (e.g. owner/dataset-clean)")
    parser.add_argument("--seq-col",  default=None,  help="Sequence column name (auto-detected if omitted)")
    parser.add_argument("--label-col", default=None, help="Label column(s) to keep (comma-separated)")
    parser.add_argument("--min-len",  type=int, default=None, help="Minimum sequence length")
    parser.add_argument("--max-len",  type=int, default=1024, help="Maximum sequence length (default 1024 = ESM2 limit)")
    parser.add_argument("--train",  type=float, default=0.8, help="Train split ratio (default 0.8)")
    parser.add_argument("--val",    type=float, default=0.1, help="Val split ratio (default 0.1)")
    parser.add_argument("--test",   type=float, default=0.1, help="Test split ratio (default 0.1)")
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--split",  default="train", help="Dataset split to load (default: train)")
    args = parser.parse_args()

    login(token=get_token())

    logger.info(f"Loading {args.input_dataset} (split={args.split})")
    ds = load_dataset(args.input_dataset, split=args.split)
    df = ds.to_pandas()
    logger.info(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")

    seq_col = args.seq_col or _infer_seq_col(df)
    logger.info(f"Sequence column: {seq_col!r}")

    # Keep only relevant columns
    keep_cols = [seq_col]
    if args.label_col:
        keep_cols += [c.strip() for c in args.label_col.split(",") if c.strip() in df.columns]
    extra = [c for c in df.columns if c not in keep_cols]
    if extra:
        logger.info(f"Dropping {len(extra)} non-selected columns: {extra}")
    df = df[keep_cols].copy()

    df, manifest = preprocess(df, seq_col, args.min_len, args.max_len)
    logger.info(f"Preprocessing manifest: {manifest}")

    ratios = (args.train, args.val, args.test)
    df = assign_splits(df, ratios, args.seed, stratify_col=args.label_col.split(",")[0] if args.label_col else None)

    split_counts = df["split"].value_counts().to_dict()
    logger.info(f"Split counts: {split_counts}")

    # Push
    dataset_out = Dataset.from_pandas(df, preserve_index=False)
    dataset_out.push_to_hub(args.output_repo)

    card = DatasetCard(f"""---
license: other
---
# {args.output_repo.split('/')[-1]}

Preprocessed from [{args.input_dataset}](https://huggingface.co/datasets/{args.input_dataset})
on {datetime.utcnow().strftime('%Y-%m-%d')}.

## Preprocessing manifest
```
{manifest}
```

## Splits
```
{split_counts}
```
""")
    card.push_to_hub(args.output_repo)
    logger.info(f"Pushed to https://huggingface.co/datasets/{args.output_repo}")


if __name__ == "__main__":
    main()
