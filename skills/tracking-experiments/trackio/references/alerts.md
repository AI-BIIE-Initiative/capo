# Trackio Alerts

Alerts let you flag important training events directly from code. They are the primary mechanism for agents to diagnose PLM fine-tuning runs and iterate autonomously.

Alerts are printed to the terminal, stored in the database, displayed in the dashboard, and optionally sent to webhooks (Slack/Discord).

## Core API

### trackio.alert()

```python
trackio.alert(
    title="NaN loss",                                # Short title (required)
    text="Loss became NaN at epoch 3 — check lr_encoder",  # Detail (optional)
    level=trackio.AlertLevel.ERROR,                  # INFO, WARN, or ERROR (default: WARN)
    webhook_url="https://hooks.slack.com/...",        # Per-alert webhook override (optional)
)
```

### Alert Levels

| Level | Usage |
|-------|-------|
| `trackio.AlertLevel.INFO` | Milestones: new best checkpoint saved, eval completed |
| `trackio.AlertLevel.WARN` | Potential issues: MCC plateau, high gradient norm, slow convergence |
| `trackio.AlertLevel.ERROR` | Critical failures: NaN loss, loss divergence, CUDA OOM |

### Webhook Support

Set a global webhook URL via `trackio.init()` or the `TRACKIO_WEBHOOK_URL` environment variable. Alerts are auto-formatted for Slack and Discord URLs.

```python
trackio.init(
    project="esm2-binding",
    webhook_url="https://hooks.slack.com/services/...",
    webhook_min_level=trackio.AlertLevel.WARN,  # Only send WARN+ to webhook
)
```

Per-alert override:

```python
trackio.alert(
    title="CUDA OOM",
    level=trackio.AlertLevel.ERROR,
    webhook_url="https://hooks.slack.com/services/...",
)
```

Environment variables:
- `TRACKIO_WEBHOOK_URL` — global webhook URL
- `TRACKIO_WEBHOOK_MIN_LEVEL` — minimum level for webhook delivery (`info`, `warn`, `error`)

## Retrieving Alerts (CLI)

```bash
# List all alerts for a project
trackio list alerts --project esm2-binding --json

# Filter by run or level
trackio list alerts --project esm2-binding --run esm2-650m-unfreeze4 --level error --json

# Poll for new alerts since a timestamp (efficient for agents)
trackio list alerts --project esm2-binding --json --since "6-04-01T12:00:00"
```

### JSON Output Structure

```json
{
  "project": "esm2-binding",
  "run": null,
  "level": null,
  "since": "2026-06-01T12:00:00",
  "alerts": [
    {
      "run": "esm2-650m-unfreeze4-mean",
      "title": "MCC plateau",
      "text": "No improvement for 5 epochs (best avg_test_mcc=0.4821)",
      "level": "warn",
      "step": 5,
      "timestamp": "2026-06-01T12:05:30"
    }
  ]
}
```

## PLM Fine-tuning Alert Patterns

The following conditions cover the most common failure modes when fine-tuning ESM2, ProtT5, or Ankh:

```python
import math, trackio

trackio.init(project="esm2-binding", config={"lr_encoder": 1e-5, "unfreeze_layers": 4})

best_avg_mcc = -1.0
no_improve_epochs = 0
prev_loss = None

for epoch in range(1, epochs + 1):
    loss = train_epoch()
    avg_test_mcc = eval_epoch()

    trackio.log({"epoch": epoch, "loss": loss, "avg_test_mcc": avg_test_mcc})

    # 1. NaN loss — common when lr_encoder is too high for unfrozen PLM layers
    if math.isnan(loss) or math.isinf(loss):
        trackio.alert(
            title="NaN/Inf loss",
            text=f"Epoch {epoch}: loss={loss} — reduce lr_encoder or unfreeze fewer layers",
            level=trackio.AlertLevel.ERROR,
        )
        break

    # 2. Loss divergence — loss increasing or staying very high after warmup
    if epoch > 3 and loss > 2.0:
        trackio.alert(
            title="Loss divergence",
            text=f"Loss {loss:.4f} still high at epoch {epoch} — lr may be too high",
            level=trackio.AlertLevel.WARN,
        )

    # 3. Gradient explosion — large loss jump from previous epoch
    if prev_loss is not None and loss > prev_loss * 3 and epoch > 2:
        trackio.alert(
            title="Gradient explosion suspected",
            text=f"Loss jumped from {prev_loss:.4f} to {loss:.4f} at epoch {epoch}",
            level=trackio.AlertLevel.WARN,
        )
    prev_loss = loss

    # 4. New best checkpoint — INFO so agents know a checkpoint was saved
    if avg_test_mcc > best_avg_mcc + 1e-4:
        best_avg_mcc = avg_test_mcc
        no_improve_epochs = 0
        trackio.alert(
            title="New best checkpoint",
            text=f"avg_test_mcc={avg_test_mcc:.4f} at epoch {epoch} — checkpoint saved",
            level=trackio.AlertLevel.INFO,
        )
    else:
        no_improve_epochs += 1

    # 5. MCC plateau — model not improving; may need different lr or more unfrozen layers
    if no_improve_epochs >= 5:
        trackio.alert(
            title="MCC plateau",
            text=(
                f"avg_test_mcc has not improved for {no_improve_epochs} epochs "
                f"(best={best_avg_mcc:.4f}). Consider adjusting lr_encoder or unfreeze_layers."
            ),
            level=trackio.AlertLevel.WARN,
        )

trackio.finish()
```

## Autonomous Agent Workflow

The recommended pattern for an agent running PLM experiments:

### 1. Insert Alerts Into Training Code

Add the diagnostic calls above for the conditions the agent should react to.

### 2. Monitor Alerts

Alerts are printed to the terminal when fired. For background or detached runs, poll via CLI:

```bash
trackio list alerts --project esm2-binding --json --since "2026-04-01T00:00:00"
```

### 3. Inspect Metrics Around the Alert

When an alert fires at epoch N, get all metrics around that point:

```bash
# All metrics at epoch 5
trackio get snapshot --project esm2-binding --run esm2-650m-unfreeze4-mean --step 5 --json

# avg_test_mcc trajectory around the plateau
trackio get metric --project esm2-binding --run esm2-650m-unfreeze4-mean \
  --metric avg_test_mcc --around 5 --window 5 --json
```

### 4. React and Iterate

Based on alerts:
- **NaN/Inf loss (ERROR)** → stop run; reduce `lr_encoder` (try 10x smaller) or reduce `unfreeze_layers`
- **Loss divergence (WARN)** → inspect loss curve; consider gradient clipping or lower LR
- **MCC plateau (WARN)** → try unfreezing more layers, larger LR for head, or different pooling
- **New best checkpoint (INFO)** → note the epoch and config; continue monitoring
- **Gradient explosion (WARN)** → add gradient clipping (`clip_grad_norm_`) or reduce `lr_encoder`

### 5. Compare Across Runs

```bash
# Check avg_test_mcc from a previous run
trackio get metric --project esm2-binding --run esm2-650m-unfreeze4-mean \
  --metric avg_test_mcc --json

# Launch new run with adjusted config
python train.py --lr_encoder 5e-6 --unfreeze_layers 2
```
