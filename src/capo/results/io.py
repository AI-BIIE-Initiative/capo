from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from capo.remote.run_manager import RunStatus


@dataclass
class RunOutputSummary:
    run_id: str
    status: RunStatus | None
    metrics: list[dict]
    prediction_files: list[Path]
    checkpoint_files: list[Path]
    local_dir: Path


def has_checkpoint_content(path: Path) -> bool:
    """True if path is a checkpoint worth counting as "saved".

    A checkpoint counts only when it holds real bytes: a non-empty file, or a
    directory containing at least one file (searched recursively). The training
    scaffold routinely pre-creates checkpoints/best and checkpoints/last
    directories that stay empty when a run fails before writing any weights —
    those must not be reported as saved checkpoints.
    """
    if path.is_dir():
        return any(f.is_file() for f in path.rglob("*"))
    return path.is_file() and path.stat().st_size > 0


def list_saved_checkpoints(checkpoints_dir: Path) -> list[Path]:
    """Sorted immediate children of checkpoints_dir that hold content.

    Empty entries (and a missing directory) yield an empty list — see
    :func:`has_checkpoint_content`.
    """
    if not checkpoints_dir.exists():
        return []
    return sorted(p for p in checkpoints_dir.iterdir() if has_checkpoint_content(p))


def load_predictions(local_run_dir: Path) -> list[Path]:
    """Return sorted list of files in local_run_dir/outputs/."""
    outputs_dir = local_run_dir / "outputs"
    if not outputs_dir.exists():
        return []
    return sorted(outputs_dir.iterdir())


def load_metrics(local_run_dir: Path) -> list[dict]:
    """Parse local_run_dir/metrics.jsonl line by line. Return list of dicts."""
    metrics_file = local_run_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return []
    records: list[dict] = []
    for line in metrics_file.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def summarize_outputs(local_run_dir: Path) -> RunOutputSummary:
    """
    Read status.json (if present) → RunStatus.
    load_metrics, load_predictions, glob checkpoints/.
    Return RunOutputSummary.
    """
    local_run_dir = Path(local_run_dir)

    # Read status.json
    status: RunStatus | None = None
    status_file = local_run_dir / "status.json"
    if status_file.exists():
        try:
            data = json.loads(status_file.read_text())
            run_id = data.get("run_id", local_run_dir.name)
            status = RunStatus(
                run_id=run_id,
                state=data.get("state", "unknown"),
                stage=data.get("stage", ""),
                current_step=data.get("current_step", 0),
                total_steps=data.get("total_steps", 0),
                started_at=data.get("started_at", ""),
                updated_at=data.get("updated_at", ""),
                message=data.get("message", ""),
                latest_output=data.get("latest_output"),
                latest_checkpoint=data.get("latest_checkpoint"),
                error=data.get("error"),
            )
        except (json.JSONDecodeError, KeyError):
            pass

    run_id = status.run_id if status else local_run_dir.name
    metrics = load_metrics(local_run_dir)
    predictions = load_predictions(local_run_dir)

    checkpoints_dir = local_run_dir / "checkpoints"
    checkpoint_files = list_saved_checkpoints(checkpoints_dir)

    return RunOutputSummary(
        run_id=run_id,
        status=status,
        metrics=metrics,
        prediction_files=predictions,
        checkpoint_files=checkpoint_files,
        local_dir=local_run_dir,
    )
