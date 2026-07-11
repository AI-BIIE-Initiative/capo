#!/usr/bin/env python3
"""
validate_registry.py — Validate models.jsonl, aliases.jsonl, and benchmarks.jsonl.

Checks:
  - Each record is valid JSON
  - Each record passes JSON Schema validation
  - registry_id values are unique in models.jsonl
  - hf_repo_id values are unique in models.jsonl
  - All registry_id values in aliases.jsonl and benchmarks.jsonl
    exist in models.jsonl

Usage:
    python validate_registry.py
    python validate_registry.py --registry-dir /path/to/registry_src
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    sys.exit("Missing dependency: pip install jsonschema")

SCRIPT_DIR = Path(__file__).parent
REGISTRY_SRC = SCRIPT_DIR.parent / "registry_src"
SCHEMA_DIR = SCRIPT_DIR.parent / "schema"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ✗ {path.name}:{i} — invalid JSON: {e}")
                sys.exit(1)
    return records


def load_schema(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def validate_against_schema(records: list[dict], schema: dict, filename: str) -> int:
    errors = 0
    validator = jsonschema.Draft202012Validator(schema)
    for i, record in enumerate(records, 1):
        errs = sorted(validator.iter_errors(record), key=lambda e: e.path)
        for err in errs:
            path = ".".join(str(p) for p in err.absolute_path) or "<root>"
            rid = record.get("registry_id", f"row {i}")
            print(f"  ✗ {filename}:{i} ({rid}) [{path}] — {err.message}")
            errors += 1
    return errors


def check_uniqueness(records: list[dict], field: str, filename: str) -> int:
    seen: dict[str, int] = {}
    errors = 0
    for i, record in enumerate(records, 1):
        val = record.get(field)
        if val is None:
            continue
        if val in seen:
            print(f"  ✗ {filename}:{i} — duplicate {field}: '{val}' (first seen at row {seen[val]})")
            errors += 1
        else:
            seen[val] = i
    return errors


def check_foreign_keys(
    child_records: list[dict],
    child_field: str,
    parent_ids: set[str],
    child_filename: str,
) -> int:
    errors = 0
    for i, record in enumerate(child_records, 1):
        val = record.get(child_field)
        if val and val not in parent_ids:
            print(f"  ✗ {child_filename}:{i} — {child_field} '{val}' not found in models.jsonl")
            errors += 1
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-dir", type=Path, default=REGISTRY_SRC)
    args = parser.parse_args()

    src = args.registry_dir
    total_errors = 0

    # --- models.jsonl ---
    models_path = src / "models.jsonl"
    print(f"Validating {models_path.name} ...")
    model_schema = load_schema(SCHEMA_DIR / "model_registry.schema.json")
    models = load_jsonl(models_path)
    total_errors += validate_against_schema(models, model_schema, "models.jsonl")
    total_errors += check_uniqueness(models, "registry_id", "models.jsonl")
    total_errors += check_uniqueness(models, "hf_repo_id", "models.jsonl")
    model_ids = {r["registry_id"] for r in models if "registry_id" in r}
    print(f"  → {len(models)} records")

    # --- aliases.jsonl ---
    aliases_path = src / "aliases.jsonl"
    if aliases_path.exists():
        print(f"Validating {aliases_path.name} ...")
        alias_schema = load_schema(SCHEMA_DIR / "alias_registry.schema.json")
        aliases = load_jsonl(aliases_path)
        total_errors += validate_against_schema(aliases, alias_schema, "aliases.jsonl")
        total_errors += check_foreign_keys(aliases, "registry_id", model_ids, "aliases.jsonl")
        print(f"  → {len(aliases)} records")

    # --- benchmarks.jsonl ---
    benchmarks_path = src / "benchmarks.jsonl"
    if benchmarks_path.exists():
        bench_records = load_jsonl(benchmarks_path)
        if bench_records:
            print(f"Validating {benchmarks_path.name} ...")
            bench_schema = load_schema(SCHEMA_DIR / "benchmark_registry.schema.json")
            total_errors += validate_against_schema(bench_records, bench_schema, "benchmarks.jsonl")
            total_errors += check_foreign_keys(bench_records, "registry_id", model_ids, "benchmarks.jsonl")
            print(f"  → {len(bench_records)} records")

    # --- Summary ---
    print()
    if total_errors == 0:
        print(f"✓ {len(models)} model records valid — no errors found")
    else:
        print(f"✗ {total_errors} error(s) found — fix before publishing")
        sys.exit(1)


if __name__ == "__main__":
    main()
