#!/usr/bin/env python3
"""
publish_to_hub.py — Validate, build Parquet, and push to BIIE-AI/protein-model-registry (private).

Steps:
  1. Run validate_registry.py (aborts on failure)
  2. Run build_parquet.py
  3. Upload out/ + README.md to HF dataset repo

Requires:
    pip install huggingface_hub pandas pyarrow jsonschema
    HF_TOKEN env var with write access to BIIE-AI org

Usage:
    python publish_to_hub.py
    python publish_to_hub.py --repo-id BIIE-AI/protein-model-registry
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    sys.exit("Missing dependency: pip install huggingface_hub")

SCRIPT_DIR = Path(__file__).parent
REGISTRY_ROOT = SCRIPT_DIR.parent
DEFAULT_REPO = "BIIE-AI/protein-model-registry"


def run_script(script: Path) -> None:
    result = subprocess.run([sys.executable, str(script)], capture_output=False)
    if result.returncode != 0:
        sys.exit(f"Aborted: {script.name} failed (exit {result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN env var not set — required for pushing to a private repo.")

    # # Step 1: validate
    # if not args.skip_validate:
    #     print("Step 1/3: Validating registry ...")
    #     run_script(SCRIPT_DIR / "validate_registry.py")
    # else:
    #     print("Step 1/3: Skipping validation (--skip-validate)")

    # # Step 2: build Parquet
    # print("Step 2/3: Building Parquet files ...")
    # run_script(SCRIPT_DIR / "build_parquet.py")

    # Step 3: push to Hub
    print(f"Step 3/3: Pushing to {args.repo_id} ...")
    api = HfApi(token=token)

    try:
        api.repo_info(repo_id=args.repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        raise ValueError(f"Repo {args.repo_id} not found. Create it first with --create-repo or on the HF website.")

    #     print(f"  Creating private dataset repo {args.repo_id} ...")
    #     create_repo(args.repo_id, repo_type="dataset", private=True, token=token)

    # # Upload README (dataset card) from registry root
    # readme = REGISTRY_ROOT / "README.md"
    # if readme.exists():
    #     api.upload_file(
    #         path_or_fileobj=str(readme),
    #         path_in_repo="README.md",
    #         repo_id=args.repo_id,
    #         repo_type="dataset",
    #         token=token,
    #         commit_message="Update dataset card",
    #     )

    docs = REGISTRY_ROOT / "docs" / "taxonomy.md"
    if docs.exists():
        api.upload_file(
            path_or_fileobj=str(docs),
            path_in_repo="docs/taxonomy.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Update dataset card",
        )

    # add selection_policy.yaml to root of repo for easy reference
    policy_src = REGISTRY_ROOT / "selection_policy.yaml"
    if policy_src.exists():
        api.upload_file(
            path_or_fileobj=str(policy_src),
            path_in_repo="selection_policy.yaml",
            repo_id=args.repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Update model selection policy",
        )
    
    # # Upload all Parquet files
    # out_dir = REGISTRY_ROOT / "out"
    # parquet_files = list(out_dir.glob("*.parquet"))
    # if not parquet_files:
    #     sys.exit("No Parquet files found in out/ — run build_parquet.py first.")

    # for pq in parquet_files:
    #     api.upload_file(
    #         path_or_fileobj=str(pq),
    #         path_in_repo=pq.name,
    #         repo_id=args.repo_id,
    #         repo_type="dataset",
    #         token=token,
    #         commit_message=f"Update {pq.name}",
    #     )
    #     print(f"  ✓ Uploaded {pq.name}")

    print(f"\nPublished to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
