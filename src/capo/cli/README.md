# `capo` — the CAPO command-line interface

`capo` is a thin presentation layer over the orchestrator. It reads the **same
config** as `scripts/run_fine_tuning.py` (`scripts/configs/fine_tuning.yaml`) and,
once the task is shaped, builds the **same `FineTuningOrchestrator`** and calls
`run_sync()`. Nothing here changes orchestrator behaviour — it only adds a chat
front door, a live run view, and inspection commands.

Installed as the `capo` entry point (`pyproject.toml → [project.scripts]`).
`python -m capo.cli` is an equivalent alias.

```bash
capo                 # interactive assistant → live run view (default)
capo --auto          # non-interactive launch from config dataset + task
python -m capo.cli   # same as bare `capo`
```

---

## Commands

| Command | What it does |
|---|---|
| `capo` | Open the interactive assistant (chat → run-plan card → live run view). |
| `capo --auto [--dataset REF] [--task TXT]` | Skip the chat; launch straight from config (or the given `--dataset`/`--task`). Equivalent to `cli_mode: auto`. |
| `capo --config PATH` | Use a different config YAML (default: `scripts/configs/fine_tuning.yaml`). |
| `capo resume <run_id> [--answer TXT]` | Resume an interrupted or **paused** run. A pause (e.g. cost-overrun confirmation) re-asks its pending question; `--answer` answers non-interactively (e.g. `--answer accept`). |
| `capo config [--config PATH]` | Arrow-key config editor — edit values, Save writes them back preserving YAML comments. |
| `capo history [-n LIMIT] [--full RUN_ID]` | List recent runs as a table (newest first). `--full` prints untruncated detail for one run. |
| `capo inspect <run_id>` | Full run detail **plus** the artifacts present in the run directory. |
| `capo health <run_id> [-w/--watch]` | Health card: phase, loss, MCC/AUROC, GPU, cost, trackio URL. `--watch` refreshes every 10s. |
| `capo status <run_id>` | One-line phase + terminal state + last-updated timestamp. |
| `capo prune-memory [run_id]` | Forget a run from episodic memory (removes its block from `runs/runs_index.md` only — never touches the run dir or checkpoints). No `run_id` opens a multi-select picker. |

Most read commands accept `--runs-dir PATH` to point at a non-default run root.

---

## The interactive experience

Bare `capo` runs three surfaces in sequence:

1. **Front-door chat** — a Sonnet-backed assistant reads your free-text request,
   asks only the clarifying questions that matter (arrow-key pickers, always with
   a free-text *Other…*), and shapes the enriched task brief that becomes
   `task.md`. Slash commands work at any point.
2. **Run-plan card + budget confirm** — before launching, CAPO shows the resolved
   task, dataset, model, strategy, GPU and budget, and asks you to confirm the
   spend. Declining drops you back to the chat (or points you at `/config`).
3. **Live run view** — a full-screen view owns the terminal for the whole run:
   colour-coded logs stream in a scrollable pane, with a command bar pinned at the
   bottom. Scroll with the mouse wheel / `Shift+↑↓` / `PgUp`·`PgDn`; scrolling back
   to the bottom (or typing a command) resumes live follow. `Ctrl+D` or `/quit`
   detach to plain streaming; `Ctrl+C` stops the run (with confirmation).
   Set `CAPO_RUN_UI=plain` to force the plain streaming path if a terminal
   misbehaves.

When a run finishes, an interactive terminal drops into a **post-run chat** that
can inspect this run's files and, if you ask to fine-tune again, route straight
back through the same pipeline.

### Slash commands

Available inside the chat and the live run view (scope in parentheses). Unique
prefixes resolve — e.g. `/heal` → `/health`.

| Command | Scope | What it does |
|---|---|---|
| `/help` | both | List every command. |
| `/config` | both | Open the config editor. |
| `/history` | both | Show previous runs. |
| `/tune` | both | Refine the task with an instruction. |
| `/retune` | chat | Modify the current task with an instruction. |
| `/prune-memory` | both | Remove a run from memory by run ID. |
| `/health` | run | Show current run status. |
| `/status` | run | One-line run phase + elapsed. |
| `/abort` | run | Stop the active run (confirms, then tears down: stops training, syncs artifacts, terminates the GPU). |
| `/quit` | both | Exit CAPO (`quit` / `exit` / `q` work too). |

---

## Configuration

The CLI is a typed view over `scripts/configs/fine_tuning.yaml` (`CapoConfig`),
so `capo` and `python scripts/run_fine_tuning.py` stay in lockstep. Three ways to
change it, in order of convenience:

- **`capo config` / `/config`** — arrow-key editor for the common fields
  (dataset, model, strategy, GPU, budget, tolerance, SSH key, reuse, HF research,
  memory, trackio, probe retries, max turns, `cli_mode`). Edits are buffered and
  marked `●`; **Save** rewrites only the touched `key:` lines so the file's
  comments survive. **Cancel**/`Esc` discards.
- **Edit the YAML directly** — every field is documented inline in
  `fine_tuning.yaml`.
- **Flags** — `--config`, `--auto`, `--dataset`, `--task` for one invocation.

### CLI-only knob: `cli_mode`

Controls the interactive layer only (ignored by `run_fine_tuning.py`):

- `interactive` *(default)* — chat → run-plan card → live view.
- `auto` — zero prompts; launch from `dataset_ref` + `task`/`task_file`, exactly
  like the script. The `--auto` flag forces this for one run.

### The fields that matter most

| Field | What to set |
|---|---|
| `key_path`, `ssh_key_name` | Private SSH key path in `~/.ssh/` and the key name registered in Lambda Cloud. |
| `dataset_ref` | HF Hub id, local file path, URL, or a bare label (auto-detected). |
| `model_id` | HF/registry id to fine-tune, `custom`, or `null`/`auto` to run model selection. |
| `fine_tune_strategy` | `linear-probe` \| `lora` \| `full`. |
| `gpu_preference` | e.g. `1x A100`, `1x GH200`; `null` lets the infra agent decide. |
| `max_cost_usd`, `tolerance_threshold` | Budget cap and the cost-gate tolerance (α = 1 + tolerance). |
| `trackio_space_id` | `null` (recommended) derives `<hf-user>/capo-trackio` and creates it on first run. |
| `task` / `task_file` | Inline task string, or a path whose contents become the task. |
| `enable_memory`, `enable_hf_research` | Toggle episodic priors (Phase -2) and HF research (Phase -1). |

Auth keys (`ANTHROPIC_API_KEY`, `LAMBDA_API_KEY`, `HF_TOKEN`) are read from the
environment or a `.env` at the repo root; the CLI fails fast with a styled error
listing any that are missing.
