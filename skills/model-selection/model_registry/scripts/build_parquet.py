#!/usr/bin/env python3
"""
build_parquet.py — Convert JSONL registry sources to Parquet files in out/.

List fields (arrays) are serialized to JSON strings for Parquet compatibility.
The Parquet files are what gets uploaded to the HF dataset repo.

Usage:
    python build_parquet.py
    python build_parquet.py --registry-dir /path/to/registry_src --out-dir /path/to/out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    raise SystemExit("Missing dependency: pip install pandas pyarrow")

SCRIPT_DIR = Path(__file__).parent
REGISTRY_SRC = SCRIPT_DIR.parent / "registry_src"
OUT_DIR = SCRIPT_DIR.parent / "out"

LIST_FIELDS = [
    "input_modalities", "output_modalities", "conditioning_modalities",
    "primary_tasks", "secondary_tasks",
    "selection_tags", "exclusion_tags",
]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def jsonl_to_parquet(src: Path, dst: Path, list_fields: list[str] | None = None) -> int:
    if not src.exists():
        print(f"  skip {src.name} — not found")
        return 0

    records = load_jsonl(src)
    if not records:
        print(f"  skip {src.name} — empty")
        return 0

    df = pd.DataFrame(records)

    # Serialize list fields to JSON strings for Parquet compatibility
    for col in (list_fields or []):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: json.dumps(v) if isinstance(v, list) else v
            )

    df.to_parquet(dst, index=False, compression="snappy", engine="pyarrow")
    print(f"  ✓ {src.name} → {dst.name}  ({len(df)} rows, {dst.stat().st_size // 1024} KB)")
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-dir", type=Path, default=REGISTRY_SRC)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Building Parquet files ...")
    jsonl_to_parquet(
        args.registry_dir / "models.jsonl",
        args.out_dir / "models.parquet",
        list_fields=LIST_FIELDS,
    )
    jsonl_to_parquet(
        args.registry_dir / "aliases.jsonl",
        args.out_dir / "aliases.parquet",
    )
    jsonl_to_parquet(
        args.registry_dir / "benchmarks.jsonl",
        args.out_dir / "benchmarks.parquet",
    )
    print("Done.")


if __name__ == "__main__":
    main()
