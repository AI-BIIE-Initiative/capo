"""Lambda Cloud GPU tool implementations.

Pure-Python tool functions — no FastMCP coupling. Each function:

- accepts JSON-serialisable scalar parameters
- returns a dict (not a JSON string — the server wrapper handles encoding)
- never raises through the boundary; errors land in {"ok": False, "error": ...}

Registration with FastMCP happens in capo.mcp.server.lambda_mcp_server, 
which iterates over the canonical tool list and wraps each function with 
logged_tool and JSON serialization.

Owns the in-memory _SESSIONS dict that tracks active :class:LambdaSession
objects.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from capo.utils.logging_utils import init_run, log_trace_event
from capo.remote import instance_guard
from capo.remote import (
    LambdaSession,
    LambdaSessionConfig,
    estimate_cost,
    find_local_ssh_keys,
    get_instance,
    list_instance_types,
    list_instances,
    list_remote_ssh_keys,
    provision_instance,
    run_preflight,
    safe_terminate_instance,
)

RUN_CTX = init_run("lambda_mcp_server")

_SESSIONS: dict[str, LambdaSession] = {}


def _get_session(session_id: str) -> tuple[LambdaSession | None, dict[str, Any] | None]:
    session = _SESSIONS.get(session_id)
    if not session:
        return None, {"ok": False, "error": f"Unknown session_id: {session_id}"}
    return session, None


# ---------------------------------------------------------------------------
# 1. Discovery
# ---------------------------------------------------------------------------

def lambda_find_local_ssh_keys(ssh_dir: str | None = None) -> dict[str, Any]:
    """Scan a local directory (default ~/.ssh) for SSH private keys.

    Returns paths and metadata only — never key material.
    """
    try:
        keys = find_local_ssh_keys(ssh_dir)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "keys": keys, "count": len(keys)}


def lambda_list_ssh_keys(api_key: str | None = None) -> dict[str, Any]:
    """List SSH keys registered on the Lambda Cloud account."""
    try:
        keys = list_remote_ssh_keys(api_key=api_key)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "keys": keys, "count": len(keys)}


def lambda_list_instance_types(
    available_only: bool = True,
    api_key: str | None = None,
) -> dict[str, Any]:
    """List Lambda Cloud instance types with live capacity and pricing."""
    try:
        types = list_instance_types(available_only=available_only, api_key=api_key)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "instance_types": types, "count": len(types)}


def lambda_preflight(
    key_path: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run preflight checks: API key, ssh/rsync/tmux on PATH, key file validity.

    Returns {"ok": bool, "checks": [{"name", "passed", "detail"}, ...]}
    where ok is the AND of every individual check.
    """
    try:
        return run_preflight(key_path=key_path, api_key=api_key)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "checks": []}


# ---------------------------------------------------------------------------
# 2. Provisioning + cost
# ---------------------------------------------------------------------------

def lambda_provision_instance(
    instance_type: str,
    ssh_key_name: str,
    region: str | None = None,
    name: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Launch a new Lambda Cloud GPU instance.

    Returns the instance id, status, region, instance_type, and pricing fields.
    Status will typically be "booting"
    poll lambda_list_instances until it becomes "active".

    A CAPO run may launch only ONE Lambda instance (multi-GPU is allowed via a
    larger instance_type, but never a second distinct instance). If this run has
    already claimed an instance, the call is refused with ok=False — reuse the
    existing instance instead.
    """
    # The one-instance-per-run invariant is only enforced inside a real run
    # (CAPO_RUN_ID set), standalone tool use is unconstrained.
    _guarded = instance_guard.enforcement_enabled()
    if _guarded:
        try:
            instance_guard.assert_can_provision()
        except instance_guard.SingleInstanceViolation as exc:
            return {"ok": False, "error": str(exc), "single_instance_violation": True}
    try:
        inst = provision_instance(
            instance_type=instance_type,
            ssh_key_name=ssh_key_name,
            region=region,
            name=name,
            api_key=api_key,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    # Claim the newly-created instance so any further provision in this run is
    # blocked by the single-instance guard above.
    if _guarded:
        instance_guard.claim(inst.instance_id)

    return {
        "ok": True,
        "instance_id": inst.instance_id,
        "status": inst.status,
        "ip": inst.ip,
        "region": inst.region,
        "instance_type": inst.instance_type,
        "ssh_key_names": inst.ssh_key_names,
        "name": inst.name,
        "price_cents_per_hour": inst.price_cents_per_hour,
        "price_dollars_per_hour": inst.price_dollars_per_hour,
        "launched_at": inst.launched_at,
    }


def _cost_estimate_dict(
    instance_id: str,
    budget_limit_dollars: float | None,
    budget_warning_threshold_dollars: float | None,
    api_key: str | None,
) -> dict[str, Any]:
    inst = get_instance(instance_id, api_key=api_key)
    estimate = estimate_cost(
        instance_id=instance_id,
        price_dollars_per_hour=inst.price_dollars_per_hour,
        started_at=inst.launched_at,
        budget_limit_dollars=budget_limit_dollars,
        budget_warning_threshold_dollars=budget_warning_threshold_dollars,
    )
    return {
        "ok": True,
        "instance_id": instance_id,
        "instance_status": inst.status,
        "instance_type": inst.instance_type,
        "estimate": asdict(estimate),
    }


def lambda_get_first_cost_estimate(
    instance_id: str,
    budget_limit_dollars: float | None = None,
    budget_warning_threshold_dollars: float | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Compute the t0 cost estimate for an instance just after provisioning.

    Equivalent to lambda_get_cost_estimate function, returned separately to
    make the workflow's "baseline at provision time" intent explicit at
    audit time.
    """
    try:
        return _cost_estimate_dict(
            instance_id,
            budget_limit_dollars,
            budget_warning_threshold_dollars,
            api_key,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def lambda_get_cost_estimate(
    instance_id: str,
    budget_limit_dollars: float | None = None,
    budget_warning_threshold_dollars: float | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Compute the elapsed-time × hourly-rate cost for an active instance.

    Pulls launched_at and price_dollars_per_hour from a fresh
    :func:get_instance call — no client-side persistence.
    """
    try:
        return _cost_estimate_dict(
            instance_id,
            budget_limit_dollars,
            budget_warning_threshold_dollars,
            api_key,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# 3. Session lifecycle
# ---------------------------------------------------------------------------

def lambda_start_session(
    host: str,
    user: str,
    remote_workdir: str,
    local_workdir: str,
    key_path: str | None = None,
    rsync_excludes: list[str] | None = None,
    rsync_interval_s: int = 30,
    session_name: str | None = None,
) -> dict[str, Any]:
    """Open a Lambda GPU session: SSH interactive shell + background rsync watch.

    Creates a two-window tmux session (remote + sync).
    """
    config = LambdaSessionConfig(
        host=host,
        user=user,
        remote_workdir=remote_workdir,
        local_workdir=local_workdir,
        key_path=key_path,
        rsync_excludes=rsync_excludes
        or [".git", "__pycache__", "*.pyc", ".DS_Store"],
        rsync_interval_s=rsync_interval_s,
        session_name=session_name,
    )
    try:
        # Construction validates remote_workdir is run-scoped (~/capo_runs/<run_id>)
        # and rejects the bare capo_runs root before any tmux/SSH is created.
        session = LambdaSession(config=config)
        session.setup()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    _SESSIONS[session.session_id] = session
    RUN_CTX.update_session(
        session.session_id,
        {"status": "connected", "started_at": time.time(), "host": host},
    )
    log_trace_event(
        RUN_CTX,
        "session_start",
        session.session_id,
        {"env": "lambda", "host": host, "user": user},
    )

    return {
        "ok": True,
        "session_id": session.session_id,
        "tmux_session": session.tmux_session_name,
        "created_at": time.time(),
        "note": (
            "Window 0 'remote' = SSH shell. Window 1 'sync' = rsync watch loop. "
            "Use lambda_run_command(session_id, command) to run commands. "
            "Use lambda_push_files / lambda_pull_files for file transfer. "
            "Use lambda_tmux_attach_command(session_id) to get the attach command "
            "so you can watch the terminal live. "
            "Use lambda_disconnect(session_id) when done."
        ),
    }


def lambda_run_command(
    session_id: str,
    command: str,
    timeout_s: float = 300,
) -> dict[str, Any]:
    """Run a shell command on the remote Lambda instance.

    Blocks until the command completes (sentinel-based detection) or
    timeout_s elapses. For long jobs, start them with nohup ... & and
    use :func:lambda_get_output to monitor.
    """
    session, err = _get_session(session_id)
    if err:
        return err
    try:
        result = session.execute_code(command, timeout_s=timeout_s)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "stdout": "", "stderr": ""}
    return {
        "ok": True,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_s": round(result.execution_time, 3),
    }


def lambda_push_files(session_id: str) -> dict[str, Any]:
    """One-shot rsync push: local_workdir → remote_workdir."""
    session, err = _get_session(session_id)
    if err:
        return err
    try:
        result = session.sync_push()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": result.success,
        "transferred_bytes": result.transferred_bytes,
        "duration_s": round(result.duration_s, 3),
        "stderr": result.stderr if not result.success else None,
    }


def lambda_pull_files(
    session_id: str,
    remote_subpath: str,
    local_dest: str,
) -> dict[str, Any]:
    """Pull a file or directory from the remote instance to a local path.

    Use this for large result artifacts (checkpoints, logs, outputs) — never
    cat large files through stdout. remote_subpath is relative to
    remote_workdir.
    """
    session, err = _get_session(session_id)
    if err:
        return err
    try:
        result = session.sync_pull(remote_subpath, local_dest)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": result.success,
        "transferred_bytes": result.transferred_bytes,
        "duration_s": round(result.duration_s, 3),
        "local_dest": local_dest,
        "stderr": result.stderr if not result.success else None,
    }


def lambda_get_output(session_id: str, lines: int = 200) -> dict[str, Any]:
    """Capture the last N lines of the SSH pane without blocking."""
    session, err = _get_session(session_id)
    if err:
        return err
    return {"ok": True, "output": session.get_pane_output(lines=lines)}


def lambda_list_sessions() -> dict[str, Any]:
    """List all active Lambda sessions managed by this server."""
    sessions = [
        {
            "session_id": sid,
            "host": s.config.host,
            "user": s.config.user,
            "tmux_session": s.tmux_session_name,
        }
        for sid, s in _SESSIONS.items()
    ]
    return {"ok": True, "sessions": sessions, "count": len(sessions)}


def lambda_list_instances(
    status: str | None = None,
    ssh_key_name: str | None = None,
    region: str | None = None,
    instance_type: str | None = None,
) -> dict[str, Any]:
    """List actual Lambda Cloud instances visible to the current API key."""
    try:
        instances = list_instances(
            status=status,
            ssh_key_name=ssh_key_name,
            region=region,
            instance_type=instance_type,
        )
        return {
            "ok": True,
            "instances": [
                {
                    "instance_id": inst.instance_id,
                    "status": inst.status,
                    "ip": inst.ip,
                    "region": inst.region,
                    "instance_type": inst.instance_type,
                    "ssh_key_names": inst.ssh_key_names,
                    "name": inst.name,
                    "price_dollars_per_hour": inst.price_dollars_per_hour,
                    "launched_at": inst.launched_at,
                }
                for inst in instances
            ],
            "count": len(instances),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def lambda_disconnect(session_id: str) -> dict[str, Any]:
    """Close the Lambda session: stop rsync watch and kill the tmux session."""
    session = _SESSIONS.pop(session_id, None)
    if not session:
        return {"ok": False, "error": f"Unknown session_id: {session_id}"}
    try:
        session.teardown()
    except Exception as exc:
        return {"ok": False, "error": f"Teardown error: {exc}"}
    RUN_CTX.update_session(
        session_id, {"status": "disconnected", "ended_at": time.time()}
    )
    log_trace_event(RUN_CTX, "session_end", session_id, {"env": "lambda"})
    return {"ok": True, "message": f"Disconnected session {session_id}"}


def lambda_tmux_attach_command(session_id: str) -> dict[str, Any]:
    """Return the shell command to attach to the session's tmux window."""
    session, err = _get_session(session_id)
    if err:
        return err
    return {
        "ok": True,
        "command": session.tmux_attach_command(),
        "note": (
            "Run this in your terminal to attach. "
            "Window 0 'remote' is the SSH shell. "
            "Window 1 'sync' is the rsync watch loop. "
            "Detach without killing with Ctrl-b d."
        ),
    }


# ---------------------------------------------------------------------------
# 4. Termination
# ---------------------------------------------------------------------------

def lambda_terminate_safe(
    instance_id: str,
    expected_ssh_key_names: list[str],
    api_key: str | None = None,
) -> dict[str, Any]:
    """Terminate an instance only after verifying ssh_key_names ownership.

    Refuses to terminate when expected_ssh_key_names is not a subset of the
    instance's actual ssh_key_names. Termination is irreversible — a key
    mismatch indicates the instance belongs to a different user.
    """
    try:
        inst = safe_terminate_instance(
            instance_id,
            expected_ssh_key_names=expected_ssh_key_names,
            api_key=api_key,
        )
    except PermissionError as exc:
        return {"ok": False, "error": str(exc), "reason": "ownership_mismatch"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "instance_id": instance_id,
        "verified_ssh_key_names": inst.ssh_key_names,
        "message": f"Terminated {instance_id}",
    }


# ---------------------------------------------------------------------------
# 5. Local + remote tmux helpers
# ---------------------------------------------------------------------------

def lambda_ensure_workspace(
    session_name: str | None = None,
    windows: list[str] | None = None,
) -> dict[str, Any]:
    """Create the local capo tmux workspace (3 windows: remote, sync, local).

    Idempotent — safe to call if it already exists.
    """
    from capo.remote.config import (
        LOCAL_TMUX_SESSION,
        LOCAL_WINDOW_LOCAL,
        LOCAL_WINDOW_REMOTE,
        LOCAL_WINDOW_SYNC,
    )
    from capo.remote.tmux_manager import ensure_local_workspace

    name = session_name or LOCAL_TMUX_SESSION
    wins = windows or [LOCAL_WINDOW_REMOTE, LOCAL_WINDOW_SYNC, LOCAL_WINDOW_LOCAL]
    try:
        ensure_local_workspace(session_name=name, windows=wins)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "session_name": name, "windows": wins}


def lambda_send_to_window(
    session_name: str,
    window_name: str,
    command: str,
) -> dict[str, Any]:
    """Send a command to a named window in a local tmux session."""
    from capo.remote.tmux_manager import send_to_local_window

    try:
        send_to_local_window(session_name, window_name, command)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


def lambda_capture_window(
    session_name: str,
    window_name: str,
    lines: int = 200,
) -> dict[str, Any]:
    """Capture output from a named window in a local tmux session."""
    from capo.remote.tmux_manager import capture_local_window

    try:
        output = capture_local_window(session_name, window_name, lines)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "output": output}


def lambda_ensure_remote_tmux(
    ssh_alias: str,
    remote_session_name: str | None = None,
    key_path: str | None = None,
) -> dict[str, Any]:
    """Ensure capo_remote tmux session exists on the remote Lambda instance."""
    from capo.remote.config import REMOTE_TMUX_SESSION
    from capo.remote.tmux_manager import ensure_remote_tmux

    rsession = remote_session_name or REMOTE_TMUX_SESSION
    try:
        ensure_remote_tmux(ssh_alias, rsession, key_path)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "remote_session_name": rsession}


def lambda_send_to_remote_tmux(
    ssh_alias: str,
    command: str,
    remote_session_name: str | None = None,
    key_path: str | None = None,
) -> dict[str, Any]:
    """Send a command to the remote capo_remote tmux session."""
    from capo.remote.tmux_manager import send_to_remote_tmux

    try:
        send_to_remote_tmux(ssh_alias, command, remote_session_name, key_path)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


# ---------------------------------------------------------------------------
# 6. Run lifecycle
# ---------------------------------------------------------------------------

def lambda_upload_run(
    ssh_target: str,
    local_run_dir: str,
    run_id: str,
    key_path: str | None = None,
) -> dict[str, Any]:
    """Upload run inputs from local capo/<run_id>/ to remote ~/capo_runs/<run_id>/.

    The remote destination is derived from run_id — callers cannot name the
    remote dir directly.
    """
    from capo.remote.rsync_manager import upload_run_inputs

    remote_run_dir = f"~/capo_runs/{run_id}"
    try:
        upload_run_inputs(ssh_target, Path(local_run_dir), remote_run_dir, key_path)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "remote_run_dir": remote_run_dir}


def lambda_push_run_file(
    ssh_target: str,
    local_run_dir: str,
    run_id: str,
    src_rel: str,
    dst_rel: str | None = None,
    key_path: str | None = None,
) -> dict[str, Any]:
    """Push a single file from local capo/<run_id>/<src_rel> → remote ~/capo_runs/<run_id>/<dst_rel>.

    Use this for repair-loop one-file pushes instead of improvising raw rsync.
    The remote destination is derived from run_id + a relative path; absolute
    paths and `..` traversal are rejected, so the destination is structurally
    guaranteed to land under the canonical remote run dir.
    """
    from capo.remote.rsync_manager import push_run_file

    try:
        push_run_file(
            ssh_target=ssh_target,
            local_run_dir=Path(local_run_dir),
            run_id=run_id,
            src_rel=src_rel,
            dst_rel=dst_rel,
            key_path=key_path,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


def lambda_sync_run_status(
    ssh_target: str,
    remote_run_dir: str,
    local_run_dir: str,
    key_path: str | None = None,
) -> dict[str, Any]:
    """Sync status.json, metrics.jsonl, and logs from remote to local."""
    from capo.remote.rsync_manager import sync_run_status

    try:
        sync_run_status(ssh_target, remote_run_dir, Path(local_run_dir), key_path)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


def lambda_start_inference(
    ssh_alias: str,
    run_id: str,
    command: str,
    key_path: str | None = None,
) -> dict[str, Any]:
    """Start a remote inference job inside capo_remote on the Lambda instance."""
    from capo.remote.run_manager import start_remote_inference

    try:
        start_remote_inference(ssh_alias, run_id, command, key_path)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "run_id": run_id}


def lambda_read_run_status(
    ssh_alias: str,
    run_id: str,
    key_path: str | None = None,
) -> dict[str, Any]:
    """Read status.json from the remote run directory."""
    from capo.remote.run_manager import read_remote_run_status

    try:
        status = read_remote_run_status(ssh_alias, run_id, key_path)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "status": asdict(status)}


def lambda_run_workflow(
    instance_type: str,
    ssh_key_name: str,
    key_path: str,
    command: str,
    model_name: str,
    local_workdir: str,
    run_config: dict | None = None,
    run_id: str | None = None,
    region: str | None = None,
    api_key: str | None = None,
    artifacts_dir: str | None = None,
) -> dict[str, Any]:
    """Full end-to-end Lambda inference workflow."""
    from capo.orchestration.orchestration import start_lambda_inference_workflow

    try:
        result = start_lambda_inference_workflow(
            instance_type=instance_type,
            ssh_key_name=ssh_key_name,
            key_path=key_path,
            command=command,
            model_name=model_name,
            local_workdir=local_workdir,
            run_config=run_config,
            run_id=run_id,
            region=region,
            api_key=api_key,
            artifacts_dir=artifacts_dir,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "run_id": result.run_id,
        "instance_id": result.instance_id,
        "ssh_alias": result.ssh_alias,
        "local_run_dir": str(result.local_run_dir),
    }


# ---------------------------------------------------------------------------
# Canonical tool registry. Consumed by the FastMCP server shim
# ---------------------------------------------------------------------------

CANONICAL_TOOLS: list[tuple[str, Any]] = [
    # Discovery
    ("lambda_find_local_ssh_keys", lambda_find_local_ssh_keys),
    ("lambda_list_ssh_keys", lambda_list_ssh_keys),
    ("lambda_list_instance_types", lambda_list_instance_types),
    ("lambda_preflight", lambda_preflight),
    # Provisioning + cost
    ("lambda_provision_instance", lambda_provision_instance),
    ("lambda_get_first_cost_estimate", lambda_get_first_cost_estimate),
    ("lambda_get_cost_estimate", lambda_get_cost_estimate),
    # Session lifecycle
    ("lambda_start_session", lambda_start_session),
    ("lambda_run_command", lambda_run_command),
    ("lambda_push_files", lambda_push_files),
    ("lambda_pull_files", lambda_pull_files),
    ("lambda_get_output", lambda_get_output),
    ("lambda_list_sessions", lambda_list_sessions),
    ("lambda_list_instances", lambda_list_instances),
    ("lambda_disconnect", lambda_disconnect),
    ("lambda_tmux_attach_command", lambda_tmux_attach_command),
    # Termination
    ("lambda_terminate_safe", lambda_terminate_safe),
    # Local + remote tmux helpers
    ("lambda_ensure_workspace", lambda_ensure_workspace),
    ("lambda_send_to_window", lambda_send_to_window),
    ("lambda_capture_window", lambda_capture_window),
    ("lambda_ensure_remote_tmux", lambda_ensure_remote_tmux),
    ("lambda_send_to_remote_tmux", lambda_send_to_remote_tmux),
    # Run lifecycle
    ("lambda_upload_run", lambda_upload_run),
    ("lambda_push_run_file", lambda_push_run_file),
    ("lambda_sync_run_status", lambda_sync_run_status),
    ("lambda_start_inference", lambda_start_inference),
    ("lambda_read_run_status", lambda_read_run_status),
    ("lambda_run_workflow", lambda_run_workflow),
]
