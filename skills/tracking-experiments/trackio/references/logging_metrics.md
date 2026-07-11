# Logging Metrics with Trackio

**Trackio** is a lightweight, free experiment tracking library from Hugging Face. It provides a wandb-compatible API for logging metrics with local-first design.

- **GitHub**: [gradio-app/trackio](https://github.com/gradio-app/trackio)
- **Docs**: [huggingface.co/docs/trackio](https://huggingface.co/docs/trackio/index)

## Installation

```bash
pip install trackio
# or
uv pip install trackio
```

## Core API

### Key Functions

| Function | Purpose |
|----------|---------|
| `trackio.init(...)` | Start a new tracking run |
| `trackio.log(dict)` | Log metrics (called repeatedly during training) |
| `trackio.finish()` | Finalize run and ensure all metrics are saved |
| `trackio.show()` | Launch the local dashboard |
| `trackio.sync(...)` | Sync local project to HF Space |

## trackio.init() Parameters

```python
trackio.init(
    project="esm2-binding",          # Project name (groups runs together)
    name="esm2-650m-unfreeze4-mean", # Run name — encode key hyperparams
    config={...},                    # Full args dict; logged once per run
    space_id="username/trackio",     # Required for Lambda/cloud GPU runs
    resume="allow",                  # Resume the run if it already exists (see below)
    group="unfreeze_4",              # Optional: group related runs
)
```

### CAPO contract: seed-then-resume (cloud/Lambda runs)

A cloud training job pushes metrics to the Space over HTTP. If the Space is
asleep when the first push happens, trackio buffers them locally and only
retries "next time you call `trackio.init()`" — which never happens on a
single-shot training box, so the run never appears on the dashboard even though
init printed "Created new run".

CAPO avoids this by **seeding the run before launch** (an orchestrator step that
waits for the Space to be RUNNING via `GET /gradio_api/info`, then does a tiny
`init(..., resume="never") → log → finish` against the awake Space). The
training script then attaches to that same run:

```python
trackio.init(project="capo-ft", name=run_id, space_id=space_id,
             resume="allow", config=vars(args))   # attaches to the seeded run
```

Rules: on a cloud run always pass `resume="allow"` (never `"never"` — that
spawns a duplicate), pin `trackio==0.29.0` so the storage backend is stable,
and call `trackio.finish()` on every exit path so buffered metrics flush.

## PLM Fine-tuning Config

Log the full training config at init. Include all hyperparameters that vary across runs:

```python
trackio.init(
    project="esm2-binding",
    name="esm2-650m-unfreeze4-mean-lr1e5",
    space_id="username/trackio",
    config={
        "model_name": "facebook/esm2_t33_650M_UR50D",
        "max_length": 256,
        "batch_size": 32,
        "epochs": 10,
        "lr_encoder": 1e-5,       # separate LR for unfrozen encoder layers
        "lr_head": 1e-3,          # LR for classification head
        "weight_decay": 0.01,
        "pooling": "mean",        # "mean" | "bos"
        "unfreeze_layers": 4,     # last N transformer layers unfrozen (0=frozen, -1=all)
        "sampler": "weighted",
        "seed": 42,
    },
)
```

## Pattern 1 — Step-level + per-epoch logging (manual training loop)

This pattern matches the structure from the ESM2 classifier training loop: log train loss
frequently within the epoch, run validation at a fraction of the epoch, log epoch-level
summary at the end.

```python
import trackio

trackio.init(project="esm2-binding", name=args.trackio_run_name, config=vars(args))

best_macro = -1.0
global_step = 0

for epoch in range(1, args.epochs + 1):
    epoch_steps = len(train_loader)
    train_log_every = max(int(epoch_steps * args.log_train_frac), 1)  # e.g. every 10%
    valid_log_every = max(int(epoch_steps * args.log_valid_frac), 1)  # e.g. every 50%

    window_loss, window_steps = 0.0, 0

    for step, batch in enumerate(train_loader, start=1):
        loss = train_step(batch)
        global_step += 1
        window_loss += loss
        window_steps += 1

        # Frequent step-level logging: running train loss
        if step % train_log_every == 0 or step == epoch_steps:
            trackio.log({
                "epoch": epoch,
                "step_in_epoch": step,
                "global_step": global_step,
                "train_loss_running": window_loss / window_steps,
            })
            window_loss, window_steps = 0.0, 0

        # Mid-epoch validation
        if step % valid_log_every == 0 or step == epoch_steps:
            metrics = evaluate(model, valid_loader)
            trackio.log({
                "epoch": epoch,
                "step_in_epoch": step,
                "global_step": global_step,
                "valid_micro_f1": metrics["micro_f1"],
                "valid_macro_f1": metrics["macro_f1"],
            })

    # End-of-epoch summary
    metrics = evaluate(model, valid_loader)
    trackio.log({"epoch": epoch, "loss": avg_epoch_loss, **metrics})

    if metrics["macro_f1"] > best_macro:
        best_macro = metrics["macro_f1"]
        torch.save(model.state_dict(), output_dir / "best_model.pt")

trackio.finish()
```

## Pattern 2 — Per-epoch MCC logging (pair-wise fine-tuning loop)

For tasks evaluated with per-species MCC (e.g. RBD–ACE2 binding), log both individual species
MCCs and the average:

```python
trackio.init(
    project="esm2-ace2-rbd",
    name=f"esm2-pwff-unfreeze{args.unfreeze_last_n}-epoch{args.epochs}",
    config=vars(args),
    space_id="username/trackio",
)

best_avg_mcc = -1.0

for epoch in range(1, epochs + 1):
    train_epoch(...)
    results, avg_test_mcc = eval_on_test(...)  # returns per-species MCCs + average

    log_dict = {"epoch": epoch, "avg_test_mcc": avg_test_mcc}
    for r in results:
        sp = r["species"].lower().replace(" ", "_")
        if not math.isnan(r["test_mcc"]):
            log_dict[f"mcc_{sp}"] = r["test_mcc"]
    trackio.log(log_dict)

    if avg_test_mcc > best_avg_mcc + 1e-12:
        best_avg_mcc = avg_test_mcc
        torch.save(checkpoint, out_dir / "esm2_pwff_best.pt")

trackio.finish()
```

## Local vs Remote Dashboard

### Local (Default)

```python
trackio.init(project="esm2-binding")
# ... training ...
trackio.finish()
trackio.show()          # or: trackio show --project esm2-binding
```

### Remote (HF Space) — required for Lambda GPU runs

```python
trackio.init(
    project="esm2-binding",
    space_id="username/trackio"   # auto-creates Space if it doesn't exist
)
```

⚠️ **Always use `space_id` on Lambda or HF Jobs** — local SQLite is lost when the instance terminates.

### Sync Local to Remote

```python
trackio.sync(project="esm2-binding", space_id="username/plm-experiments")
```

## wandb Compatibility

Trackio is a drop-in replacement for wandb:

```python
import trackio as wandb

wandb.init(project="esm2-binding")
wandb.log({"loss": 0.42, "macro_f1": 0.61})
wandb.finish()
```

## Grouping Runs

Use `group` to organize sweeps in the dashboard sidebar:

```python
# Group by number of unfrozen layers
trackio.init(project="esm2-binding", name="run-unfreeze0", group="frozen_encoder")
trackio.init(project="esm2-binding", name="run-unfreeze4", group="unfreeze_4")

# Group by pooling strategy
trackio.init(project="esm2-binding", name="run-mean", group="pooling_mean")
trackio.init(project="esm2-binding", name="run-bos",  group="pooling_bos")

# Group by encoder LR
trackio.init(project="esm2-binding", name="run-lr1e5", group="lr_encoder_1e-5")
trackio.init(project="esm2-binding", name="run-lr1e4", group="lr_encoder_1e-4")
```

## Embedding Dashboards

Embed Space dashboards in websites with query parameters:

```html
<iframe 
  src="https://username-trackio.hf.space/?project=esm2-binding&metrics=valid_macro_f1,avg_test_mcc&sidebar=hidden" 
  style="width:1600px; height:500px; border:0;">
</iframe>
```

Query parameters:
- `project`: Filter to specific project
- `metrics`: Comma-separated metric names to show
- `sidebar`: `hidden` or `collapsed`
- `smoothing`: 0-20 (smoothing slider value)
- `xmin`, `xmax`: X-axis limits
