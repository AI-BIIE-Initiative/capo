#!/usr/bin/env python3
"""
refresh_hf_metadata.py — Sync live metadata from HuggingFace Hub into models.jsonl.

Refreshed fields (do NOT edit these manually — they are overwritten on each run):
  hf_downloads, hf_likes, hf_pipeline_tag, hf_library_name, hf_last_modified

Curated fields are preserved unchanged.

Requires:
    pip install huggingface_hub
    HF_TOKEN env var (optional; needed for gated repos)

Usage:
    python refresh_hf_metadata.py
    python refresh_hf_metadata.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from huggingface_hub import model_info, HfApi
    from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
except ImportError:
    raise SystemExit("Missing dependency: pip install huggingface_hub")

SCRIPT_DIR = Path(__file__).parent
REGISTRY_SRC = SCRIPT_DIR.parent / "registry_src"
REFRESHED_FIELDS = {"hf_downloads", "hf_likes", "hf_pipeline_tag", "hf_library_name", "hf_last_modified", "hf_gated"}


def fetch_hf_metadata(repo_id: str, token: str | None) -> dict:
    try:
        info = model_info(
            repo_id,
            token=token,
            expand=["downloads", "likes", "gated", "tags", "pipeline_tag", "library_name", "lastModified"],
        )
        last_mod = None
        if hasattr(info, "last_modified") and info.last_modified:
            last_mod = info.last_modified.strftime("%Y-%m-%d") if hasattr(info.last_modified, "strftime") else str(info.last_modified)[:10]
        return {
            "hf_downloads": getattr(info, "downloads", None),
            "hf_likes": getattr(info, "likes", None),
            "hf_pipeline_tag": getattr(info, "pipeline_tag", None),
            "hf_library_name": getattr(info, "library_name", None),
            "hf_last_modified": last_mod,
            "hf_gated": getattr(info, "gated", False),
        }
    except GatedRepoError:
        return {"hf_gated": True, "_fetch_error": "gated — token required"}
    except RepositoryNotFoundError:
        return {"_fetch_error": "repo not found on HF"}
    except Exception as e:
        return {"_fetch_error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-dir", type=Path, default=REGISTRY_SRC)
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    models_path = args.registry_dir / "models.jsonl"

    with open(models_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    updated = []
    for record in records:
        rid = record.get("registry_id", "?")
        repo_id = record.get("hf_repo_id")
        if not repo_id:
            updated.append(record)
            continue

        meta = fetch_hf_metadata(repo_id, token)

        if "_fetch_error" in meta:
            print(f"  ⚠  {rid} ({repo_id}): {meta['_fetch_error']}")
        else:
            changes = {k: v for k, v in meta.items() if record.get(k) != v}
            if changes:
                print(f"  ↻  {rid}: {list(changes.keys())}")
            else:
                print(f"  ✓  {rid}: no changes")
            record.update(meta)

        updated.append(record)

    if args.dry_run:
        print("\nDry run — no files written.")
        return

    with open(models_path, "w") as f:
        for record in updated:
            f.write(json.dumps(record) + "\n")

    print(f"\nRefreshed {len(updated)} records → {models_path}")


if __name__ == "__main__":
    main()
