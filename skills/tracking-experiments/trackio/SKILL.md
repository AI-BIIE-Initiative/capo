---
name: trackio
description: Track and visualize PLM fine-tuning experiments with Trackio. Use when logging metrics during training (Python API), firing alerts for training diagnostics, or retrieving/analyzing logged metrics (CLI). Supports real-time dashboard visualization, alerts with webhooks, HF Space syncing, and JSON output for automation.
---

# Trackio - Experiment Tracking for PLM Fine-tuning

Trackio is an experiment tracking library for logging and visualizing ML training metrics. It syncs to Hugging Face Spaces for real-time monitoring dashboards.

## Three Interfaces

| Task | Interface | Reference |
|------|-----------|-----------|
| **Logging metrics** during training | Python API | [references/logging_metrics.md](references/logging_metrics.md) |
| **Firing alerts** for training diagnostics | Python API | [references/alerts.md](references/alerts.md) |
| **Retrieving metrics & alerts** after/during training | CLI | [references/retrieving_metrics.md](references/retrieving_metrics.md) |

## When to Use Each

### Python API → Logging

Use `import trackio` in your training scripts to log metrics:

- Initialize tracking with `trackio.init(project=..., config=vars(args), name=run_name)`
- Log metrics with `trackio.log()` at step level and per-epoch
- Finalize with `trackio.finish()`

**Key concept**: For remote/cloud training (Lambda GPU, HF Jobs), pass `space_id` — metrics sync to a Space dashboard so they persist after the instance terminates.

→ See [references/logging_metrics.md](references/logging_metrics.md) for setup, PLM training patterns, and configuration options.

### Python API → Alerts

Insert `trackio.alert()` calls in training code to flag important events — like inserting print statements for debugging, but structured and queryable:

- `trackio.alert(title="...", level=trackio.AlertLevel.WARN)` — fire an alert
- Three severity levels: `INFO`, `WARN`, `ERROR`
- Alerts are printed to terminal, stored in the database, shown in the dashboard, and optionally sent to webhooks (Slack/Discord)

**Key concept for autonomous agents**: Alerts are the primary mechanism for autonomous experiment iteration. Insert alerts for PLM-specific diagnostics: loss spikes when unfreezing encoder layers, MCC plateaus, NaN from high `lr_encoder`. Since alerts are printed to the terminal, an agent watching the training output sees them immediately. For background runs, poll via CLI instead.

→ See [references/alerts.md](references/alerts.md) for the full alerts API, webhook setup, and autonomous agent workflows.

### CLI → Retrieving

Use the `trackio` command to query logged metrics and alerts:

- `trackio list projects/runs/metrics` — discover what's available
- `trackio get project/run/metric` — retrieve summaries and values
- `trackio list alerts --project <name> --json` — retrieve alerts
- `trackio show` — launch the dashboard
- `trackio sync` — sync to HF Space

**Key concept**: Add `--json` for programmatic output suitable for automation and agents.

→ See [references/retrieving_metrics.md](references/retrieving_metrics.md) for all commands, workflows, and JSON output formats.

## Minimal Logging Setup

```python
import trackio

trackio.init(
    project="esm2-binding", # create a project for each experiment series (e.g. "esm2-binding") and a run for each individual experiment (e.g. "esm2-650m-unfreeze4-mean")
    name="esm2-650m-unfreeze4-mean",
    space_id="username/trackio",  # required for Lambda/cloud GPU runs
    config={
        "model_name": "facebook/esm2_t33_650M_UR50D",
        "unfreeze_layers": 4,
        "lr_encoder": 1e-5,
        "lr_head": 1e-3,
        "pooling": "mean",
    },
)
trackio.log({"epoch": 1, "loss": 0.42, "validation_loss": 0.45, "macro_f1": 0.61})
trackio.log({"epoch": 2, "loss": 0.35, "validation_loss": 0.38, "macro_f1": 0.68})
trackio.finish()
```

### Minimal Retrieval

```bash
trackio list projects --json
trackio get metric --project esm2-binding --run esm2-650m-unfreeze4-mean --metric macro_f1 --json
```

## Autonomous PLM Experiment Workflow

When running experiments autonomously as an agent, the recommended workflow is:

1. **Set up training with alerts** — insert `trackio.alert()` calls for diagnostic conditions
2. **Launch training** — run the script in the background
3. **Poll for alerts** — use `trackio list alerts --project <name> --json --since <timestamp>`
4. **Read metrics** — use `trackio get metric ...` to inspect values
5. **Iterate** — stop the run, adjust hyperparameters (e.g. `lr_encoder`, `unfreeze_layers`), relaunch

```python
import math, trackio

trackio.init(project="esm2-binding", config={"lr_encoder": 1e-5, "unfreeze_layers": 4})

best_avg_mcc = -1.0
no_improve = 0

for epoch in range(1, epochs + 1):
    loss = train_epoch()
    avg_test_mcc = eval_epoch()
    trackio.log({"epoch": epoch, "loss": loss, "avg_test_mcc": avg_test_mcc})

    if math.isnan(loss):
        trackio.alert(title="NaN loss", text=f"Epoch {epoch}: loss is NaN — check lr_encoder",
                      level=trackio.AlertLevel.ERROR)
        break

    if epoch > 3 and loss > 2.0:
        trackio.alert(title="Loss divergence",
                      text=f"Loss {loss:.4f} still high at epoch {epoch}",
                      level=trackio.AlertLevel.WARN)

    if avg_test_mcc > best_avg_mcc + 1e-4:
        best_avg_mcc = avg_test_mcc
        no_improve = 0
        trackio.alert(title="New best checkpoint",
                      text=f"avg_test_mcc={avg_test_mcc:.4f} at epoch {epoch}",
                      level=trackio.AlertLevel.INFO)
    else:
        no_improve += 1
        if no_improve >= 5:
            trackio.alert(title="MCC plateau",
                          text=f"No improvement for {no_improve} epochs (best={best_avg_mcc:.4f})",
                          level=trackio.AlertLevel.WARN)

trackio.finish()
```

Then poll from a separate terminal/process:

```bash
trackio list alerts --project esm2-binding --json --since "2026-04-01T00:00:00"
```

## Monitoring remote runs

When a training job runs on Lambda inside `capo_remote`, monitor it through files —
not by scraping terminal text.

Every run writes to `~/capo_runs/<run_id>/`:

| File | Purpose |
|------|---------|
| `status.json` | Authoritative state: `pending` / `running` / `completed` / `failed` |
| `metrics.jsonl` | Append-only structured metrics (same values trackio logs) |
| `stdout.log` / `stderr.log` | Human-readable logs |

Sync these with `lambda_sync_run_status` (MCP tool) from the sync window:

```python
lambda_sync_run_status(
    ssh_target="lambda-abc123",
    remote_run_dir="~/capo_runs/run-001",
    local_run_dir="~/.capo/artifacts/run-001",
    key_path="/path/to/key",
)
```

Parse `status.json` with `lambda_read_run_status`:

```python
lambda_read_run_status(ssh_alias="lambda-abc123", run_id="run-001", key_path="...")
# → {"ok": true, "status": {"state": "running", "started_at": "...", ...}}
```

Check `metrics.jsonl` to get the same values trackio logs if the training script writes both.
Use `trackio list alerts` (CLI) to poll for diagnostic alerts without reading the log files directly.

## HF Space Gradio API — verify the dashboard is live

The trackio dashboard is an HF Space (e.g. `theoschiff-biie/capo-trackio`).
The Space exposes a Gradio REST API that can be used to (a) confirm the
dashboard is reachable before launching a run, (b) probe it mid-run to make
sure the iframe will render for the user, and (c) drive any of the Space's
exposed endpoints programmatically.

**URL flattening.** The Space ID `<owner>/<name>` maps to
`https://<owner>-<name>.hf.space` (replace the `/` with `-`).
Example: `theoschiff-biie/capo-trackio` →
`https://theoschiff-biie-capo-trackio.hf.space`.

**Endpoints** (all require `Authorization: Bearer $HF_TOKEN`, get a token at
https://huggingface.co/settings/tokens):

| Action | Method | Path |
|--------|--------|------|
| API schema | `GET` | `/gradio_api/info` |
| Call endpoint | `POST` | `/gradio_api/call/v2/{endpoint}` — body `{"data": [...]}` returns `{"event_id": "<id>"}` |
| Poll result | `GET` | `/gradio_api/call/{endpoint}/{event_id}` — SSE stream, final `data:` line is the JSON result |
| Upload file | `POST` | `/gradio_api/upload` — `-F "files=@file.ext"`, use return path as `{"path": "<p>", "meta": {"_type": "gradio.FileData"}, "orig_name": "file.ext"}` |

**Verifying the dashboard is live (pre-launch check):**

```bash
SPACE_URL="https://theoschiff-biie-capo-trackio.hf.space"
curl -fsSL -H "Authorization: Bearer $HF_TOKEN" "$SPACE_URL/gradio_api/info"
# 200 + JSON  → dashboard is up; the user can open the iframe
# non-200 / timeout → Space is sleeping or failed to build; retry every 15s
# for up to 3 min before treating it as unhealthy.
```

If the check fails on a freshly-created Space, the build is still cold-starting —
keep polling. If it fails on an existing Space, re-run the dashboard seeding
helper (`src/capo/observability/trackio_space.py::ensure_trackio_dashboard`)
which (re)writes README/requirements/app.py and re-mounts the metrics bucket.

**Calling an endpoint programmatically:**

```bash
# 1. List endpoints
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "$SPACE_URL/gradio_api/info" | jq '.named_endpoints | keys'

# 2. Invoke (replace {endpoint} and payload as needed)
EVENT=$(curl -s -X POST \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"data": []}' \
  "$SPACE_URL/gradio_api/call/v2/{endpoint}" | jq -r .event_id)

# 3. Poll until the SSE stream emits a `data:` JSON line
curl -N -H "Authorization: Bearer $HF_TOKEN" \
  "$SPACE_URL/gradio_api/call/{endpoint}/$EVENT"
```

**File inputs** — first upload, then reference the returned path:

```bash
RESP=$(curl -s -X POST -H "Authorization: Bearer $HF_TOKEN" \
  -F "files=@plot.png" "$SPACE_URL/gradio_api/upload")
# RESP is a JSON array of server-side paths; reference like:
#   {"path": "<RESP[0]>", "meta": {"_type": "gradio.FileData"}, "orig_name": "plot.png"}
```

**Operational rule.** Tracking is non-negotiable, and it must be *verified*, not
assumed. The init-time lines ("Found existing space" / "Created new run" /
"View dashboard") print even when the Space is asleep and no metric ever lands,
so they prove nothing. Before handoff the orchestrator must, in order:
(1) confirm `/gradio_api/info` returns 200 (Space RUNNING);
(2) **seed the run** against that awake Space (`init(resume="never") → log →
finish`) so it materialises in the Space's store;
(3) write a truthful `reports/trackio_check.json` whose `reachable`/`run_seeded`
fields reflect the actual command results — never the log text.
train.py then attaches with `resume="allow"` (never `"never"`). Record the
verified dashboard URL in `reports/handoff.json` (`trackio_url`) for the Haiku
monitor and final summary.
