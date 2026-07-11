---
name: lambda-session
description: >
  Manage multi-session parallel execution on Lambda GPU instances via MCP tools.
  Use when running training/inference on a remote machine with parallel terminal contexts
  (remote execution, sync/monitoring, local work). Covers the capo tmux workspace,
  capo_remote remote session, run file-based state (status.json, metrics.jsonl), and
  all associated MCP tools: lambda_ensure_workspace, lambda_send_to_window,
  lambda_ensure_remote_tmux, lambda_start_inference, lambda_sync_run_status,
  lambda_read_run_status.
  Complements lambda-cloud-connection (instance lifecycle); this skill covers
  what to do once the instance IP is known.
---

# Lambda Session Management

## Execution contexts

The CAPO workspace uses **four named contexts** across two machines:

| Context | Location | Window / Session | Owns |
|---------|----------|-----------------|------|
| `remote` | local tmux `capo` | window `remote` | SSH commands, remote control |
| `sync` | local tmux `capo` | window `sync` | rsync transfers, status polling |
| `local` | local tmux `capo` | window `local` | free local work |
| `capo_remote` | Lambda instance | tmux session `capo_remote` | all long-running jobs |

---

## Ownership rules

### Window `remote`
- **MUST**: Run SSH commands against the instance; trigger job start; check remote shell state
- **MUST NOT**: Run rsync; run local scripts; run long-blocking loops

### Window `sync`
- **MUST**: Run `lambda_sync_run_status` / `lambda_upload_run` / `lambda_pull_files` calls
- **MUST NOT**: Start remote jobs; run SSH commands; run local compute

### Window `local`
- **MUST**: Run local scripts, data processing, analysis
- **MUST NOT**: SSH to remote; run rsync to remote

### `capo_remote` (on Lambda)
- **MUST**: Run all long-running jobs (training, inference) via `run.sh` wrapper
- **MUST NOT**: Be bypassed — never start jobs in a bare SSH shell that dies on disconnect

---

## Setup

Create the local workspace with:

```python
lambda_ensure_workspace()
# → {"ok": true, "session_name": "capo", "windows": ["remote", "sync", "local"]}
```

Create `capo_remote` on the Lambda instance:

```python
lambda_ensure_remote_tmux(ssh_alias="lambda-<instance_id>", key_path="/path/to/key")
# → {"ok": true, "remote_session_name": "capo_remote"}
```

Both calls are idempotent — safe to call repeatedly.

---

## Why long-running jobs must live in `capo_remote`

Jobs started in a bare SSH shell die when the SSH connection drops. `capo_remote` is a tmux session on the Lambda instance — it survives SSH disconnects. All training and inference jobs **must** be launched inside `capo_remote` via `lambda_start_inference` or the `lambda_run_workflow` full-workflow tool.

---

## Run files as source of truth

The orchestrator never parses terminal text for run state. Instead, every job writes structured files under `~/capo_runs/<run_id>/`:

```
~/capo_runs/<run_id>/
    spec.json       ← RunSpec: task, command, model, config
    status.json     ← RunStatus: state, step, timestamps, error
    metrics.jsonl   ← append-only structured metrics
    stdout.log
    stderr.log
    outputs/
    checkpoints/
```

`status.json` is the **authoritative run state**. Read it with `lambda_read_run_status`. Sync it locally with `lambda_sync_run_status`.

Possible states: `"pending"` | `"running"` | `"completed"` | `"failed"` | `"stopped"`

---

## Tool reference

### `lambda_ensure_workspace`

```python
lambda_ensure_workspace(session_name=None, windows=None)
# Creates local tmux session "capo" with windows [remote, sync, local]
# → {"ok": true, "session_name": "capo", "windows": ["remote", "sync", "local"]}
```

### `lambda_send_to_window`

```python
lambda_send_to_window(session_name="capo", window_name="remote", command="nvidia-smi")
# → {"ok": true}
```

### `lambda_capture_window`

```python
lambda_capture_window(session_name="capo", window_name="remote", lines=100)
# → {"ok": true, "output": "..."}
```

### `lambda_ensure_remote_tmux`

```python
lambda_ensure_remote_tmux(ssh_alias="lambda-abc123", key_path="/path/to/key")
# → {"ok": true, "remote_session_name": "capo_remote"}
```

### `lambda_send_to_remote_tmux`

```python
lambda_send_to_remote_tmux(ssh_alias="lambda-abc123", command="ls ~/capo_runs/")
# → {"ok": true}
```

### `lambda_upload_run`

```python
lambda_upload_run(
    ssh_target="lambda-abc123",
    local_run_dir="/local/path/run-001",
    remote_run_dir="~/capo_runs/run-001",
    key_path="/path/to/key",
)
# → {"ok": true}
```

### `lambda_start_inference`

```python
lambda_start_inference(
    ssh_alias="lambda-abc123",
    run_id="run-001",
    command="python inference.py --model esm2 --input inputs/seqs.fasta",
    key_path="/path/to/key",
)
# → {"ok": true, "run_id": "run-001"}
```

### `lambda_sync_run_status`

```python
lambda_sync_run_status(
    ssh_target="lambda-abc123",
    remote_run_dir="~/capo_runs/run-001",
    local_run_dir="/local/.capo/artifacts/run-001",
    key_path="/path/to/key",
)
# → {"ok": true}
```

### `lambda_read_run_status`

```python
lambda_read_run_status(ssh_alias="lambda-abc123", run_id="run-001", key_path="/path/to/key")
# → {"ok": true, "status": {"run_id": "run-001", "state": "running", "started_at": "...", ...}}
```

### `lambda_run_workflow`

Full end-to-end launch: provision → SSH ready → workspace → upload → start job.

```python
lambda_run_workflow(
    instance_type="gpu_1x_a10",
    ssh_key_name="my-lambda-key",
    key_path="/Users/me/.ssh/lambda_mykey",
    command="python train.py --epochs 20",
    model_name="esm2_t33_650M_UR50D",
    local_workdir="/Users/me/project",
    run_config={"lr": 1e-4, "batch_size": 32},
)
# → {"ok": true, "run_id": "run-abc12345", "instance_id": "...", "ssh_alias": "lambda-...", "local_run_dir": "..."}
```

---

## Monitoring workflow

```python
# 1. Sync monitoring files from remote
lambda_sync_run_status(ssh_target="lambda-abc123", remote_run_dir="~/capo_runs/run-001",
                       local_run_dir="~/.capo/artifacts/run-001", key_path="...")

# 2. Read authoritative run state
lambda_read_run_status(ssh_alias="lambda-abc123", run_id="run-001", key_path="...")
# state: "running" | "completed" | "failed" | "stopped"

# 3. Capture remote tmux output for debugging
lambda_send_to_window(session_name="capo", window_name="remote",
                      command="ssh lambda-abc123 'tail -20 ~/capo_runs/run-001/stdout.log'")
```

---

## Failure handling

| Symptom | Fix |
|---------|-----|
| `state: "failed"` in status.json | Check `error` field; sync and read `stderr.log` |
| `lambda_read_run_status` → `"pending"` | Job not started yet or run directory missing — call `lambda_start_inference` first |
| `lambda_ensure_remote_tmux` fails | SSH alias not set up; run `lambda_run_workflow` or set up alias manually |
| Job started outside `capo_remote` | Stop it; relaunch with `lambda_start_inference` — bare SSH jobs die on disconnect |

---

## Key rules

1. **All long-running jobs go in `capo_remote`** — never in a bare SSH shell.
2. **Read `status.json` for run state** — never parse terminal output.
3. **Sync window owns all rsync** — never call rsync from the remote window.
4. **`lambda_run_workflow` is the preferred single-call entry point** for new runs.
5. **Terminate via `lambda_terminate_safe`** (ownership-verified) — `sudo shutdown` does not stop billing.
