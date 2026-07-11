"""
run_manager.py — Remote ML run lifecycle.

Owns: run directory setup, standardized layout, remote command construction,
starting/stopping inference and fine-tuning, and reading run status files.
Never provisions instances. Never decides workflow order.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from capo.observability import progress as _progress
from capo.remote.config import REMOTE_TMUX_SESSION, REMOTE_RUN_ROOT
from capo.remote.tmux_manager import ensure_remote_tmux, send_to_remote_tmux


@dataclass
class RunSpec:
    run_id: str
    task: str           # "inference" | "finetune"
    command: str        # full shell command
    model_name: str
    config: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class RemoteRunPaths:
    run_root: str           # ~/capo_runs/<run_id> (unexpanded, for remote shell)
    spec_json: str
    # canonical operational paths (all under outputs/)
    status_json: str        # outputs/status.json
    pid_file: str           # outputs/train.pid
    stdout_log: str         # outputs/train.log
    stderr_log: str         # outputs/train_err.log
    metrics_jsonl: str      # outputs/metrics.jsonl
    # canonical directories
    outputs_dir: str        # outputs/ (logs, status, pid, streaming metrics)
    checkpoints_dir: str    # checkpoints/ (best/, last/)
    results_dir: str        # results/ (eval csvs, plots, predictions)
    reports_dir: str        # reports/ (manifests, health, summaries)
    src_dir: str            # src/ (data, models, train, eval, utils)


@dataclass
class RunStatus:
    run_id: str
    state: str          # "pending" | "running" | "completed" | "failed" | "stopped"
    stage: str = ""
    current_step: int = 0
    total_steps: int = 0
    started_at: str = ""
    updated_at: str = ""
    message: str = ""
    latest_output: str | None = None
    latest_checkpoint: str | None = None
    error: str | None = None


def get_remote_run_paths(
    run_id: str,
    run_root: str = REMOTE_RUN_ROOT,
) -> RemoteRunPaths:
    """Return all standard paths for run_id. Does not touch the remote."""
    base = f"{run_root}/{run_id}"
    return RemoteRunPaths(
        run_root=base,
        spec_json=f"{base}/reports/spec.json",
        status_json=f"{base}/outputs/status.json",
        pid_file=f"{base}/outputs/train.pid",
        stdout_log=f"{base}/outputs/train.log",
        stderr_log=f"{base}/outputs/train_err.log",
        metrics_jsonl=f"{base}/outputs/metrics.jsonl",
        outputs_dir=f"{base}/outputs",
        checkpoints_dir=f"{base}/checkpoints",
        results_dir=f"{base}/results",
        reports_dir=f"{base}/reports",
        src_dir=f"{base}/src",
    )


def prepare_remote_run_dir(
    ssh_alias: str,
    run_id: str,
    key_path: str | Path | None = None,
    run_root: str = REMOTE_RUN_ROOT,
) -> RemoteRunPaths:
    """
    SSH mkdir -p for the full canonical run directory tree.
    Returns RemoteRunPaths. Raises subprocess.CalledProcessError on failure.
    """
    paths = get_remote_run_paths(run_id, run_root)
    base = paths.run_root
    # Canonical layout — keep this list in sync with capo.utils.checks and the
    # local _setup_run_dir() in fine_tuning_orchestrator.
    dirs = [
        f"{base}/checkpoints/best",
        f"{base}/checkpoints/last",
        f"{base}/compaction",
        f"{base}/configs",
        f"{base}/outputs",
        f"{base}/pricing",
        f"{base}/probe",
        f"{base}/profile/plots",
        f"{base}/reports/health",
        f"{base}/results/plots",
        f"{base}/results/predictions",
        f"{base}/scripts",
        f"{base}/src/data",
        f"{base}/src/models",
        f"{base}/src/train",
        f"{base}/src/eval",
        f"{base}/src/utils",
    ]
    pkg_inits = [
        f"{base}/src/__init__.py",
        f"{base}/src/data/__init__.py",
        f"{base}/src/models/__init__.py",
        f"{base}/src/train/__init__.py",
        f"{base}/src/eval/__init__.py",
        f"{base}/src/utils/__init__.py",
    ]
    mkdir_cmd = f"mkdir -p {' '.join(dirs)}"
    touch_cmd = f"touch {' '.join(pkg_inits)}"
    cmd = ["ssh"] + _ssh_flags(key_path) + [ssh_alias, f"{mkdir_cmd} && {touch_cmd}"]
    subprocess.run(cmd, check=True, capture_output=True)
    return paths


def push_hf_token_to_remote(
    ssh_alias: str,
    hf_token: str,
    key_path: str | Path | None = None,
) -> None:
    """Write ~/.cache/huggingface/token on the remote (mode 600).

    This is the standard location read by huggingface_hub, datasets, trackio,
    and the `hf` CLI. Once set, the training agent does not need to handle
    the token explicitly — push_to_hub() and trackio.init() pick it up.

    Token is written via a single SSH call. The token never appears in shell
    history (sent via stdin) and the resulting file is chmod 600.
    """
    if not hf_token or not hf_token.strip():
        raise ValueError("hf_token is empty; refusing to push an empty token")
    remote_cmd = (
        "mkdir -p ~/.cache/huggingface && "
        "umask 077 && "
        "cat > ~/.cache/huggingface/token && "
        "chmod 600 ~/.cache/huggingface/token"
    )
    cmd = ["ssh"] + _ssh_flags(key_path) + [ssh_alias, remote_cmd]
    subprocess.run(
        cmd,
        input=hf_token.strip(),
        text=True,
        check=True,
        capture_output=True,
    )


def write_remote_spec(
    ssh_alias: str,
    spec: RunSpec,
    key_path: str | Path | None = None,
    run_root: str = REMOTE_RUN_ROOT,
) -> None:
    """
    Write spec.json to the remote run directory via SSH.
    prepare_remote_run_dir must have been called first.
    """
    paths = get_remote_run_paths(spec.run_id, run_root)
    json_str = json.dumps(asdict(spec), indent=2)
    _write_remote_file(ssh_alias, paths.spec_json, json_str, key_path)


def start_remote_inference(
    ssh_alias: str,
    run_id: str,
    command: str,
    key_path: str | Path | None = None,
    remote_session: str = REMOTE_TMUX_SESSION,
    run_root: str = REMOTE_RUN_ROOT,
) -> None:
    """
    Ensure capo_remote exists, write a wrapper script, then send it to capo_remote.
    The wrapper writes status.json on start, completion, and failure.
    Jobs must not be started outside capo_remote.
    """
    ensure_remote_tmux(ssh_alias, remote_session, str(key_path) if key_path else None)
    paths = get_remote_run_paths(run_id, run_root)
    wrapper = _build_run_wrapper(run_id, command, paths)
    _write_remote_file(ssh_alias, f"{paths.run_root}/run.sh", wrapper, key_path)
    _progress.emit(f"Starting remote inference job {run_id}")
    send_to_remote_tmux(
        ssh_alias,
        f"bash {paths.run_root}/run.sh",
        remote_session,
        str(key_path) if key_path else None,
    )


def start_remote_finetune(
    ssh_alias: str,
    run_id: str,
    command: str,
    key_path: str | Path | None = None,
    remote_session: str = REMOTE_TMUX_SESSION,
    run_root: str = REMOTE_RUN_ROOT,
) -> None:
    """Identical to start_remote_inference for now. Separate entry point for later stage."""
    start_remote_inference(ssh_alias, run_id, command, key_path, remote_session, run_root)


def stop_remote_run(
    ssh_alias: str,
    run_id: str,
    key_path: str | Path | None = None,
    remote_session: str = REMOTE_TMUX_SESSION,
) -> None:
    """Send Ctrl-C to capo_remote, then update status.json to state=stopped."""
    send_to_remote_tmux(ssh_alias, "C-c", remote_session, str(key_path) if key_path else None)
    # Best-effort status update
    paths = get_remote_run_paths(run_id)
    now = datetime.now(timezone.utc).isoformat()
    status_json = json.dumps({"run_id": run_id, "state": "stopped", "updated_at": now})
    try:
        _write_remote_file(ssh_alias, paths.status_json, status_json, key_path)
    except subprocess.CalledProcessError:
        pass


def read_remote_run_status(
    ssh_alias: str,
    run_id: str,
    key_path: str | Path | None = None,
    run_root: str = REMOTE_RUN_ROOT,
) -> RunStatus:
    """
    SSH cat status.json, parse JSON, return RunStatus.
    If status.json does not exist: return RunStatus(run_id=run_id, state="pending").
    """
    paths = get_remote_run_paths(run_id, run_root)
    cmd = (
        ["ssh"] + _ssh_flags(key_path)
        + [ssh_alias, f"cat {paths.status_json} 2>/dev/null || echo '{{}}'"]
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout.strip() or "{}")
    if not data:
        return RunStatus(run_id=run_id, state="pending")
    fields = {f: data.get(f, v.default if hasattr(v, "default") else None)
              for f, v in RunStatus.__dataclass_fields__.items()}
    fields["run_id"] = data.get("run_id", run_id)
    fields["state"] = data.get("state", "pending")
    return RunStatus(**{k: v for k, v in fields.items() if v is not None or k in ("run_id", "state")})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ssh_flags(key_path: str | Path | None) -> list[str]:
    flags = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
    if key_path:
        flags += ["-i", str(key_path)]
    return flags


def _write_remote_file(
    ssh_alias: str,
    remote_path: str,
    content: str,
    key_path: str | Path | None = None,
) -> None:
    """Write content to remote_path via SSH + printf."""
    escaped = content.replace("'", "'\\''")
    cmd = ["ssh"] + _ssh_flags(key_path) + [ssh_alias, f"printf '%s' '{escaped}' > {remote_path}"]
    subprocess.run(cmd, check=True, capture_output=True)


def _build_run_wrapper(run_id: str, command: str, paths: RemoteRunPaths) -> str:
    """
    Return a bash script string that:
    1. writes outputs/status.json with state=running + started_at
    2. runs command, redirecting stdout/stderr to outputs/
    3. writes outputs/status.json with state=completed or state=failed
    """
    return f"""#!/bin/bash
set -euo pipefail
RUN_ID="{run_id}"
RUN_DIR="{paths.run_root}"
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
mkdir -p "$RUN_DIR/outputs"

write_status() {{
    printf '%s' "$1" > "$RUN_DIR/outputs/status.json"
}}

write_status "$(printf '{{"run_id":"%s","state":"running","started_at":"%s","updated_at":"%s"}}' "$RUN_ID" "$STARTED" "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"

{command} >> "$RUN_DIR/outputs/train.log" 2>> "$RUN_DIR/outputs/train_err.log"
EXIT_CODE=$?

FINISHED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ $EXIT_CODE -eq 0 ]; then
    write_status "$(printf '{{"run_id":"%s","state":"completed","started_at":"%s","updated_at":"%s"}}' "$RUN_ID" "$STARTED" "$FINISHED")"
else
    write_status "$(printf '{{"run_id":"%s","state":"failed","error":"exit_code_%d","started_at":"%s","updated_at":"%s"}}' "$RUN_ID" "$EXIT_CODE" "$STARTED" "$FINISHED")"
fi
"""
