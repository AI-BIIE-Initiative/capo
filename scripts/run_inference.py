"""
Run protein model inference on Lambda via the CAPO InferenceOrchestrator.

All run options live in a YAML config. Copy scripts/configs/inference.yaml,
edit as needed, and pass its path via --config.

Usage:
    python scripts/run_inference.py --config scripts/configs/inference.yaml

Environment:
    LAMBDA_API_KEY  — required
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from capo.orchestration.orchestration import InferenceOrchestrator
from capo.observability.progress import _format_size

_REPO_ROOT = Path(__file__).parent.parent


def _resolve_task(cfg: dict) -> str:
    task = cfg.get("task")
    if task:
        return str(task).strip()

    task_file = cfg.get("task_file")
    if not task_file:
        raise ValueError("Config must set either `task` or `task_file`.")

    path = Path(task_file)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"task_file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _expand(path: str | None) -> str | None:
    return str(Path(path).expanduser()) if path else None


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (see scripts/configs/inference.yaml).",
    )
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    task_description = _resolve_task(cfg)

    orch = InferenceOrchestrator(
        key_path=_expand(cfg["key_path"]),
        ssh_key_name=cfg["ssh_key_name"],
        instance_type=cfg.get("instance_type"),
        instance_name=cfg.get("instance_name"),
    )

    result = orch.run_sync(
        task_description=task_description,
        run_id=cfg.get("run_id"),
    )

    print(f"\nrun_id     : {result.run_id}")
    print(f"state      : {result.state}")
    print(f"output_dir : {result.local_run_dir}")
    for f in result.output_files:
        size = f.stat().st_size
        print(f"  output   : {f.name}  ({_format_size(size)})")
    if result.cost_usd:
        print(f"agent cost : ${result.cost_usd:.4f}")


if __name__ == "__main__":
    main()
