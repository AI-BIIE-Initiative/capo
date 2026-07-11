# Retrieving Metrics with Trackio CLI

The `trackio` CLI provides direct terminal access to query Trackio experiment tracking data locally without needing to start the MCP server.

## Quick Command Reference

| Task | Command |
|------|---------|
| List projects | `trackio list projects` |
| List runs | `trackio list runs --project <name>` |
| List metrics | `trackio list metrics --project <name> --run <name>` |
| List system metrics | `trackio list system-metrics --project <name> --run <name>` |
| List alerts | `trackio list alerts --project <name> [--run <name>] [--level <level>] [--since <timestamp>]` |
| Get project summary | `trackio get project --project <name>` |
| Get run summary | `trackio get run --project <name> --run <name>` |
| Get metric values | `trackio get metric --project <name> --run <name> --metric <name>` |
| Get metric at step | `trackio get metric ... --metric <name> --step <N>` |
| Get metric around step | `trackio get metric ... --metric <name> --around <N> --window <W>` |
| Get all metrics snapshot | `trackio get snapshot --project <name> --run <name> --step <N>` |
| Get system metrics | `trackio get system-metric --project <name> --run <name>` |
| Show dashboard | `trackio show [--project <name>]` |
| Sync to Space | `trackio sync --project <name> --space-id <space_id>` |

## Core Commands

### List Commands

```bash
trackio list projects                                    # List all projects
trackio list projects --json                            # JSON output

trackio list runs --project esm2-binding                # List runs in project
trackio list runs --project esm2-binding --json

trackio list metrics --project esm2-binding --run esm2-650m-unfreeze4-mean
trackio list metrics --project esm2-binding --run esm2-650m-unfreeze4-mean --json

trackio list system-metrics --project esm2-binding --run esm2-650m-unfreeze4-mean
trackio list system-metrics --project esm2-binding --run esm2-650m-unfreeze4-mean --json

trackio list alerts --project esm2-binding                          # List alerts
trackio list alerts --project esm2-binding --run esm2-650m-unfreeze4-mean --json
trackio list alerts --project esm2-binding --level error --json
trackio list alerts --project esm2-binding --json --since <ts>      # Poll since timestamp
```

### Get Commands

```bash
trackio get project --project esm2-binding
trackio get project --project esm2-binding --json

trackio get run --project esm2-binding --run esm2-650m-unfreeze4-mean
trackio get run --project esm2-binding --run esm2-650m-unfreeze4-mean --json

trackio get metric --project esm2-binding --run esm2-650m-unfreeze4-mean --metric avg_test_mcc
trackio get metric --project esm2-binding --run esm2-650m-unfreeze4-mean --metric avg_test_mcc --json
trackio get metric ... --metric valid_macro_f1 --step 5                  # At exact epoch/step
trackio get metric ... --metric avg_test_mcc --around 5 --window 3       # ±3 steps
trackio get metric ... --metric train_loss_running --at-time <ts> --window 60

trackio get snapshot --project esm2-binding --run esm2-650m-unfreeze4-mean --step 5 --json
trackio get snapshot --project esm2-binding --run esm2-650m-unfreeze4-mean --around 5 --window 2 --json

trackio get system-metric --project esm2-binding --run esm2-650m-unfreeze4-mean
trackio get system-metric --project esm2-binding --run esm2-650m-unfreeze4-mean --metric gpu_utilization --json
```

### Dashboard Commands

```bash
trackio show                                              # Launch dashboard
trackio show --project esm2-binding                     # Load specific project
trackio show --theme <theme>                            # Custom theme
trackio show --mcp-server                               # Enable MCP server
trackio show --color-palette "#FF0000,#00FF00"         # Custom colors
```

### Sync Commands

```bash
trackio sync --project esm2-binding --space-id username/plm-experiments
trackio sync --project esm2-binding --space-id username/plm-experiments --private
trackio sync --project esm2-binding --space-id username/plm-experiments --force
```

## Output Formats

All `list` and `get` commands support two output formats:

- **Human-readable** (default): Formatted text for terminal viewing
- **JSON** (with `--json` flag): Structured JSON for programmatic use

## Common Patterns

### Discover Projects and Runs

```bash
trackio list projects
trackio list runs --project esm2-binding
trackio get project --project esm2-binding --json
```

### Inspect a Fine-tuning Run

```bash
# Get run summary — confirms config (unfreeze_layers, lr_encoder, pooling, etc.)
trackio get run --project esm2-binding --run esm2-650m-unfreeze4-mean --json

# List available metrics
trackio list metrics --project esm2-binding --run esm2-650m-unfreeze4-mean

# Get per-epoch avg_test_mcc
trackio get metric --project esm2-binding --run esm2-650m-unfreeze4-mean \
  --metric avg_test_mcc --json

# Get mid-epoch validation macro F1
trackio get metric --project esm2-binding --run esm2-650m-unfreeze4-mean \
  --metric valid_macro_f1 --json
```

### Query GPU Utilization

```bash
trackio list system-metrics --project esm2-binding --run esm2-650m-unfreeze4-mean
trackio get system-metric --project esm2-binding --run esm2-650m-unfreeze4-mean --json
trackio get system-metric --project esm2-binding --run esm2-650m-unfreeze4-mean \
  --metric gpu_utilization --json
```

### Automation Scripts

```bash
# Extract best avg_test_mcc for a run
BEST_MCC=$(trackio get metric --project esm2-binding \
  --run esm2-650m-unfreeze4-mean --metric avg_test_mcc --json \
  | jq '[.values[].value] | max')

# Export run summary to file
trackio get run --project esm2-binding --run esm2-650m-unfreeze4-mean \
  --json > run_summary.json

# List all runs containing "unfreeze4"
trackio list runs --project esm2-binding --json \
  | jq '.runs[] | select(contains("unfreeze4"))'
```

### Agent Workflow for PLM Experiments

```bash
# 1. Discover available projects
trackio list projects --json

# 2. Explore project structure
trackio get project --project esm2-binding --json

# 3. Inspect a specific run (check config: unfreeze_layers, lr_encoder, pooling)
trackio get run --project esm2-binding --run esm2-650m-unfreeze4-mean --json

# 4. Track MCC progression across epochs
trackio get metric --project esm2-binding --run esm2-650m-unfreeze4-mean \
  --metric avg_test_mcc --json

# 5. Poll for alerts (use --since for efficient incremental polling)
trackio list alerts --project esm2-binding --json --since "2026-04-04T00:00:00"

# 6. When a plateau alert fires at epoch 5, inspect all metrics around that point
trackio get snapshot --project esm2-binding --run esm2-650m-unfreeze4-mean \
  --around 5 --window 2 --json
```

## Error Handling

Commands validate inputs and return clear errors:

- Missing project: `Error: Project '<name>' not found.`
- Missing run: `Error: Run '<name>' not found in project '<project>'.`
- Missing metric: `Error: Metric '<name>' not found in run '<run>' of project '<project>'.`

All errors exit with non-zero status code and write to stderr.

## Key Options

- `--project`: Project name (required for most commands)
- `--run`: Run name (required for run-specific commands)
- `--metric`: Metric name (required for metric-specific commands)
- `--json`: Output in JSON format instead of human-readable
- `--step`: Exact step filter (for `get metric`, `get snapshot`)
- `--around`: Center step for window filter (for `get metric`, `get snapshot`)
- `--at-time`: Center ISO timestamp for window filter (for `get metric`, `get snapshot`)
- `--window`: Window size: ±steps for `--around`, ±seconds for `--at-time` (default: 10)
- `--level`: Alert level filter (`info`, `warn`, `error`) (for `list alerts`)
- `--since`: ISO timestamp to filter alerts after (for `list alerts`)
- `--theme`: Dashboard theme (for `show` command)
- `--mcp-server`: Enable MCP server mode (for `show` command)
- `--color-palette`: Comma-separated hex colors (for `show` command)
- `--private`: Create private Space (for `sync` command)
- `--force`: Overwrite existing database (for `sync` command)

## JSON Output Structure

### List Projects
```json
{"projects": ["esm2-binding", "esm2-ace2-rbd", "prot-t5-localization"]}
```

### List Runs
```json
{"project": "esm2-binding", "runs": ["esm2-650m-unfreeze4-mean", "esm2-650m-frozen-bos"]}
```

### Run Summary
```json
{
  "project": "esm2-binding",
  "run": "esm2-650m-unfreeze4-mean",
  "num_logs": 120,
  "metrics": ["train_loss_running", "valid_micro_f1", "valid_macro_f1", "loss", "macro_f1", "avg_test_mcc"],
  "config": {"model_name": "facebook/esm2_t33_650M_UR50D", "unfreeze_layers": 4, "lr_encoder": 1e-5},
  "last_step": 10
}
```

### Metric Values
```json
{
  "project": "esm2-binding",
  "run": "esm2-650m-unfreeze4-mean",
  "metric": "avg_test_mcc",
  "values": [
    {"step": 1, "timestamp": "2026-04-04T10:00:00", "value": 0.31},
    {"step": 2, "timestamp": "2026-04-04T10:15:00", "value": 0.45},
    {"step": 3, "timestamp": "2026-04-04T10:30:00", "value": 0.52}
  ]
}
```

## References

- **Complete CLI documentation**: See [docs/source/cli_commands.md](docs/source/cli_commands.md)
- **API and MCP Server**: See [docs/source/api_mcp_server.md](docs/source/api_mcp_server.md)
