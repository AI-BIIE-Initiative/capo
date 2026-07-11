# CAPO — Claude Code guide

## Mission

**Compute-aware automatic protein language model (PLM) fine-tuning.**
The system connects to a Lambda GPU, selects the right model and strategy for the task, profiles the dataset and generates exploratory plots, writes all training/evaluation scripts, rsyncs everything to the remote instance via a persistent tmux window, launches training, then runs a concurrent health monitor that tracks loss, AUROC, MCC, and other metrics until the run completes. All decisions are budget-aware and backed by JSON artifacts on disk.

---

## End-to-end pipeline (in order)

```
Phase -2 Episodic memory     memory-consultant scans runs/runs_index.md → selects 0–3 prior RUN_REPORT.md → prior_runs.md
Phase -1 HF research         find training / eval datasets, benchmarks, hyperparameters on HF Hub
Phase 0  Secure GPU          attach running instance or provision new (budget-aware)
Phase 1  Profile dataset     4-stage pipeline → profile.json + plots in profile/plots/
Phase 2  Write scripts       probe.py, train.py, eval.py; upload via rsync
Phase 3  Feasibility probe   p99-length batch: forward → forward+backward → probe_result.json
Gate     Cost gate           projected_cost_usd vs max_cost_usd → abort or proceed
Gate     Seed trackio run    wait for Space RUNNING (/gradio_api/info 200) → seed run → truthful trackio_check.json
Phase 4  Launch training     nohup in capo_remote tmux; train.py attaches to the seeded run (resume="allow")
Phase 5  Health monitor      concurrent Haiku loop; 60s → 5min polls; writes history.jsonl
Phase 6  Finalize            sync artifacts, diagnose, write final_summary.json + RUN_REPORT.md, append runs_index.md
```

**Episodic memory (Phases -2 / 6).** Each completed run writes a `RUN_REPORT.md` (YAML frontmatter + scientific body) and appends its frontmatter to the cross-run index at `<repo>/runs/runs_index.md`. Before the next run's pre-launch agents dispatch, the `memory-consultant` subagent scans that index by frontmatter only (cheap fingerprint pass), selectively reads the 0–3 most relevant past report bodies (progressive disclosure), and writes `reports/prior_runs.md`. Those priors are **advisory** — current-run artifacts (probe, profile, cost gate) always override them when they conflict.

---

## Repository layout

```
scripts/                        Entry points (run these directly)
src/capo/                       Core library
  orchestration/                Agent runners and orchestrators
  observability/                Progress emitter + training health monitor
  persistence/                  Disk-backed run state (session store)
  memory/                       Episodic memory: RUN_REPORT.md + runs_index.md helpers
  research/                     Pre-launch HF Hub research helper
  remote/                       Lambda session, rsync, tmux, run lifecycle
  mcp/                          Three MCP servers (lambda, local, docker REPLs)
  preprocessing/                Preprocessor registry
  results/                      Result I/O helpers
skills/                         41 skill SKILL.md files + scripts + references
baselines/                      Experiment baselines (claude/, open-code/, slm-vs-agentskills/)
```

---

## scripts/ — entry points

| Script | What it does |
|--------|-------------|
| `scripts/run_fine_tuning.py` | Main entry point. Reads `scripts/configs/fine_tuning.yaml`, instantiates `FineTuningOrchestrator`, runs the full pipeline end-to-end. |
| `scripts/run_inference.py` | Inference entry point. Reads `scripts/configs/inference.yaml`, runs `InferenceOrchestrator`. |
| `scripts/train.py` | Baseline experiment runner (SLM vs agent comparison). Do not modify. |
| `scripts/configs/fine_tuning.yaml` | All fine-tuning knobs: `model_id`, `fine_tune_strategy`, `dataset_ref`, `gpu_preference`, `max_cost_usd`, `trackio_space_id`, `allow_reuse_existing`, `epochs`, `probe_max_retries`. |
| `scripts/configs/inference.yaml` | Inference config. |

---

## src/capo/orchestration/ — orchestrators and runners

| File | What it does |
|------|-------------|
| `fine_tuning_orchestrator.py` | **Central orchestrator.** `FineTuningOrchestrator` drives the full pipeline via a three-phase agent handoff (Sonnet pre-launch → Haiku monitor → Sonnet finalizer). Contains `_FINE_TUNING_SYSTEM_PROMPT` and `_FINE_TUNING_PROMPT_TEMPLATE` which are the authoritative instructions the Sonnet agent follows step-by-step. |
| `orchestration.py` | `InferenceOrchestrator` and `PhasedOrchestrator` (pre-launch scouting DAG). Also contains `LambdaWorkflowResult` and lower-level Lambda workflow helpers. |
| `agent_runner.py` | `AgentRunner` wraps `claude_agent_sdk.query`. Owns the **`SUBAGENTS` registry** — see table below. |
| `prompt_caching.py` | Claude SDK helpers used only by `agent_runner.py`: `cached_streaming_prompt` (cache-control breakpoints) and `extract_cache_stats` (cache token tallies from raw messages). |
| `slm_runner.py` | `SLMRunner` — local HF Transformers wrapper for small open models (Llama / Qwen / Gemma / Mistral) with chat template handling. |

## src/capo/observability/ — progress emission and run monitoring

| File | What it does |
|------|-------------|
| `progress.py` | `ProgressEmitter` — timestamped, prefixed progress lines to stdout + log files. Parses every tool call and result into human-readable lines. Activity tag configurable (`"inference"` / `"fine-tuning"`). Module-level `emit()`, `error()`, `emit_tool_call()`, `emit_tool_result()` used throughout `remote/` and `orchestration/`. |
| `training_health_monitor.py` | `TrainingHealthMonitor` — concurrent Haiku loop. Waits for `reports/handoff.json`, then polls SSH every 60s (first 15 min) or 5 min (steady state). Parses state, metrics trend, alerts, severity. Appends `HealthReport` to `reports/health/history.jsonl`. Terminates on `completed`, `failed`, or `severe` alert. Accepts `/health` on stdin for on-demand checks. |

## src/capo/persistence/ — disk-backed run state

| File | What it does |
|------|-------------|
| `session_store.py` | `SessionStore`, `SessionState`, `new_session` — atomic, lock-protected JSON manifest under `~/.capo/sessions/<run_id>.json` for resuming interrupted runs. Used by `FineTuningOrchestrator` and `scripts/capo_resume.py`. |

## src/capo/memory/ — episodic cross-run memory

| File | What it does |
|------|-------------|
| `run_report.py` | `RUN_REPORT.md` + `runs_index.md` helpers. `parse_frontmatter`, `read_index_blocks`, `append_index_block` (atomic, `fcntl`-locked, dedup by `run_id`), `validate_frontmatter` (strict 13-field schema). CLI: `python -m capo.memory.run_report append-from-report --run-dir <path>` — invoked by the finalizer to register a finished run. The cross-run index lives at `<repo>/runs/runs_index.md` as a sequence of YAML frontmatter blocks delimited by `---`; it is the cheap discovery layer the `memory-consultant` scans before deciding which full reports to load. |

**Frontmatter schema (13 fields, exactly):** `run_id`, `task_summary`, `modality`, `target`, `organism`, `assay`, `best_metric_name`, `best_metric_value`, `final_val_loss`, `key_decisions`, `key_findings`, `key_pitfalls`, `report_path`. Required: `run_id`, `task_summary`, `report_path`. Per-run reports live at `<repo>/runs/<run_id>/RUN_REPORT.md`.

## src/capo/research/ — pre-launch external research

| File | What it does |
|------|-------------|
| `hf_research.py` | `HFResearcher`, `ResearchFindings` — lightweight Bash-only agent (hf CLI / curl) that gathers training datasets, eval benchmarks, and hyperparameter recommendations from the HuggingFace Hub before the main orchestrator launches. |

### Three-phase fine-tuning handoff

```
(Phase -2 — Haiku                Phase A — Sonnet            Phase B — Haiku                 Phase C — Sonnet
 memory-consultant          ──►  pre-launch + launch   ──►   TrainingHealthMonitor      ──►  finalizer
 reads runs_index.md             writes handoff.json         reads handoff.json               writes final_summary.json
 writes prior_runs.md)           reads prior_runs.md         appends history.jsonl            writes RUN_REPORT.md
                                                                                              appends runs_index.md
```

**Handoff artifacts** (all under `<local_run_dir>/reports/` unless noted):
- `prior_runs.md` — curated 0–3 prior `RUN_REPORT.md` bodies selected by the `memory-consultant` (advisory priors; always written, even as a stub)
- `handoff.json` — `{ssh_alias, remote_run_dir, pid, trackio_url, launched_at_iso}`
- `health/history.jsonl` — one `HealthReport` JSON per line (state, metrics, trend, alerts, severity, trackio_url)
- `final_summary.json` — `{terminal_state, final_metrics, final_model_path, checkpoint_paths, trackio_url, actual_cost_usd}`
- `<local_run_dir>/RUN_REPORT.md` — scientific summary (YAML frontmatter + body); its frontmatter is appended to `<repo>/runs/runs_index.md`

### SUBAGENTS registry (in `agent_runner.py`)

| Key | Model | Skills | Role |
|-----|-------|--------|------|
| `memory-consultant` | Haiku | — | Scans `runs/runs_index.md` frontmatters (progressive disclosure), reads 0–3 most relevant prior `RUN_REPORT.md` bodies, writes advisory `reports/prior_runs.md`. Read-only on the index/reports; the only file it writes is `prior_runs.md`. |
| `model-selector` | Haiku | `model-selection` | Picks best PLM + fine-tune strategy from registry |
| `lambda-session-manager` | Haiku | `lambda-session`, `lambda-cloud-connection` | Full Lambda GPU workflow: launch, SSH, run, retrieve |
| `cloud-provider-connector` | Haiku | `cost-estimation`, `huggingface-jobs` | Budget-constrained hardware + provider selection |
| `data-profiler` | Haiku | `profiling-datasets` | 4-stage dataset profiling → profile.json + plots |
| `experiment-tracker` | Haiku | `tracking-experiments/trackio` | `trackio.init` → returns dashboard URL |
| `training-health-monitor` | Haiku | — | One SSH round-trip per invocation; returns strict JSON health report |

---

## src/capo/remote/ — Lambda session and run lifecycle

| File | What it does |
|------|-------------|
| `lambda_session.py` | `LambdaSession` — SSH + rsync + tmux lifecycle for one instance. `provision_instance()`, `list_instances()` (with status/region/type filters), `_lambda_api_key()`. `LambdaInstance` dataclass. |
| `run_manager.py` | Remote ML run lifecycle: `prepare_remote_run_dir`, `start_remote_inference`, `start_remote_finetune`, `read_remote_run_status`. Owns standardized remote directory layout under `~/capo_runs/<run_id>/`. |
| `rsync_manager.py` | `RsyncManager` — push/pull with progress parsing. `RsyncResult` dataclass. |
| `tmux_manager.py` | `TmuxManager` — local and remote tmux session management (create, send-keys, capture). `ensure_remote_tmux`, `send_to_remote_tmux`. |
| `config.py` | Constants: `REMOTE_TMUX_SESSION="capo_remote"`, `REMOTE_RUN_ROOT`, `LOCAL_ARTIFACTS_ROOT`, SSH timeouts. |

**Remote run directory layout** (on Lambda instance):
```
~/capo_runs/<run_id>/
  outputs/
    stdout.log  stderr.log  metrics.jsonl  status.json  train.pid
    checkpoints/    # on-instance source of truth during training
  reports/          # probe_result.json, cost_report.json, handoff.json, health/
  profile/          # profile.json, probe_batch_recipe.json, plots/
```

---

## src/capo/mcp/ — MCP servers

Three MCP servers auto-loaded via `.mcp.json`. **MCP tools take priority over equivalent skills.**

| Server | Config key | Prefix |
|--------|-----------|--------|
| Lambda GPU sessions | `lambda-repl` | `mcp__lambda-repl__*` |
| Local Python REPL | `local-repl` | `mcp__local-repl__*` |
| Docker REPL | `docker-repl` | `mcp__docker-repl__*` |

**Key lambda-repl tools** (registered in `src/capo/mcp/server/lambda_mcp_server.py`, implemented in `src/capo/mcp/tools/lambda_tools.py`):

| Tool | What it does |
|------|-------------|
| `lambda_find_local_ssh_keys` | Scan `~/.ssh/` for private keys (paths only, never contents) |
| `lambda_list_ssh_keys` | List SSH keys registered on the Lambda account |
| `lambda_list_instance_types` | Catalog of GPU types with live capacity + pricing |
| `lambda_preflight` | Check API key, ssh/rsync/tmux on PATH, key file validity |
| `lambda_provision_instance` | Launch a new Lambda Cloud GPU instance |
| `lambda_get_first_cost_estimate` | Baseline cost estimate at provision time |
| `lambda_get_cost_estimate` | Elapsed-time × hourly-rate cost (vs optional budget) |
| `lambda_start_session` | Open SSH + rsync tmux session to a Lambda instance |
| `lambda_run_command` | Run a shell command on the remote (in capo_remote tmux) |
| `lambda_push_files` / `lambda_pull_files` | Rsync local↔remote |
| `lambda_terminate_safe` | Terminate after verifying ssh_key_names ownership |
| `lambda_get_output` | Tail background job stdout (non-blocking) |
| `lambda_list_sessions` | List in-memory `LambdaSession` objects |
| `lambda_list_instances` | Query Lambda Cloud REST API for real instances (with status/ssh_key/region/type filters) |
| `lambda_ensure_remote_tmux` | Create `capo_remote` tmux session on remote if not alive |
| `lambda_send_to_remote_tmux` | Send command to `capo_remote` tmux window |
| `lambda_upload_run` | Rsync full local run dir to remote |
| `lambda_sync_run_status` | Pull `status.json` and `metrics.jsonl` from remote |
| `lambda_start_inference` / `lambda_read_run_status` | Inference-specific lifecycle |
| `lambda_disconnect` | Close session |

---

## skills/ — 41 skills

Each `SKILL.md` is a self-contained instruction set for a subagent. Scripts live in `scripts/`, references in `references/`.

### Pipeline entry points (invoke these)
| Skill | What it does |
|-------|-------------|
| `profiling-datasets` | **Start here for any dataset.** 4-stage pipeline: detect → load → analyze + plots → preprocessing recommendation. Produces `profile.json` and plots in `profile/plots/`. An empty `plots` dict from Stage 3 is a failure. |
| `model-selection` | Pick best PLM + strategy from `model_registry/`. Returns best-fit / budget / frontier candidates. |
| `cost-estimation` | Estimate GPU cost across Lambda, AWS, GCP, Azure, HF Jobs. References: `references/{lambda,aws,azure,google-cloud,huggingface-jobs}-pricing.md`. |

### Analysis skills (called internally by `profiling-datasets` Stage 3)
| Skill | When used |
|-------|----------|
| `analysis/analyze-protein-sequences` | FASTA / HF datasets with AA sequences |
| `analysis/analyze-tabular` | CSV/Parquet/Excel — instrument-aware (plate reader, Octet, Nanodrop) |
| `analysis/analyze-single-cell` | H5AD / MTX scRNA-seq — scanpy QC metrics |
| `analysis/analyze-fcs` | Flow cytometry FCS — scatter, fluorescence, compensation check |
| `analysis/analyze-fastq-reads` | FASTQ — Q-score distributions, quality gates |
| `analysis/analyze-bam-reads` | BAM/SAM — mapping rate, MAPQ |

### Data processing
| Skill | What it does |
|-------|-------------|
| `data-processing/protein-sequence-data` | Clean, filter, deduplicate, split protein datasets. 5 scripts in `scripts/`. |
| `data-processing/numerical-data` | Normalize, impute, encode, split tabular data. |
| `data-processing/single-cell` | Filter, normalize, HVG selection (scanpy). |

### Model inference scripts (one script each in `scripts/`)
| Skill | Model family |
|-------|-------------|
| `model-inference/esm` | ESM embeddings, variant scoring, structure, generation |
| `model-inference/ankh` | Ankh-base / ankh-large / ankh3 embeddings |
| `model-inference/prottrans` | ProtBert / ProtT5 embeddings |
| `model-inference/boltz` | Structure + affinity prediction (protein, ligand, DNA, RNA) |
| `model-inference/boltzgen` | Generative protein design (binder, antibody, CDR, peptide) |
| `model-inference/chai` | Chai-1 structure prediction (no MSA required) |

### Other skills
| Skill | What it does |
|-------|-------------|
| `clustering` | Cluster sequences or embeddings; homology-safe splits (mmseqs2/CD-HIT). 11 scripts. |
| `dimensionality-reduction` | Reduce ESM embeddings to 2D/3D. 11 scripts. |
| `tracking-experiments/trackio` | `trackio.init` — **only called in Step 9, never during probe or canary**. References: alerts, logging_metrics, retrieving_metrics. |
| `lambda-session` | Reference doc for Lambda SSH + tmux workflow (use MCP tools instead). |
| `cloud-provider-connection/lambda` | Documentation for the lambda-repl MCP tools — workflow ordering, user checklist. No executable code. |
| `huggingface-jobs` | HF compute jobs (3 scripts: finetune, preprocess, batch-inference). |
| `hf-cli` | HF Hub model/dataset upload/download. |
| `uniprot` | Retrieve sequences, annotations, domains from UniProt. |
| `gtars` | Genomic interval analysis (BED, coverage, overlap, tokenization). 6 references. |
| `model-selection/esm` | ESM-specific fine-tuning recipes (linear-probe / LoRA / full). |

---

## Plotting standards — mandatory for every plot

**All plots produced by this system must use only the following colors.** This applies to every agent writing matplotlib/seaborn/plotly code — profiling, evaluation, training curves, clustering, dimensionality reduction, or any other visualization. Text, axis labels, spines, and tick marks stay black (`#000000`).

| Role | Hex | Name |
|------|-----|------|
| Primary bars / histograms / scatter | `#1E5994` | BLUE_0 |
| Secondary / accent bars | `#E6905B` | ORANGE_50 |
| Third category | `#713D8F` | PURPLE_0 |
| Fourth category | `#0E625C` | GREEN_0 |
| First reference line / threshold | `#9B3208` | ORANGE_0 |
| Second reference line / threshold | `#713D8F` | PURPLE_0 |
| Mid-tone blue (5th category) | `#8DB8E2` | BLUE_50 |
| Mid-tone purple (6th category) | `#C694E1` | PURPLE_50 |
| Mid-tone green (7th category) | `#78B5B0` | GREEN_50 |
| Light shades (8–12 categories) | `#BDD9F5` `#FAC19E` `#EAD5F6` `#C8DFD9` | *_90 variants |
| Noise / missing / background | `#AAAAAA` | NOISE |

**Colormaps** — never use `"tab20"`, `"coolwarm"`, `"YlOrRd"`, or `"Blues"`. Use these instead:
- Sequential (entropy, counts): `LinearSegmentedColormap.from_list("capo_seq", ["#C8DFD9", "#78B5B0", "#0E625C"])`
- Diverging (correlations, signed scores): `LinearSegmentedColormap.from_list("capo_div", ["#1E5994", "#FFFFFF", "#9B3208"])`
- Single-hue (confusion matrices): `LinearSegmentedColormap.from_list("capo_blue", ["#BDD9F5", "#8DB8E2", "#1E5994"])`

**In Python scripts within `src/capo/`:** import from `capo.viz.palette` (e.g. `from capo.viz.palette import BLUE_0, ORANGE_50, CMAP_DIV`).

**In standalone skill scripts and generated train/eval code:** define the constants inline at the top of `generate_plots` / the plotting function — do not rely on a package import:
```python
_PRIMARY  = "#1E5994"   # BLUE_0
_ACCENT   = "#E6905B"   # ORANGE_50
_REF1     = "#9B3208"   # ORANGE_0
_REF2     = "#713D8F"   # PURPLE_0
_NOISE    = "#AAAAAA"
```

---

## Key operating rules

1. **GATE BEFORE TRAIN.** Feasibility probe + cost gate must both pass before `nohup train.py` is launched.
2. **trackio run is seeded and VERIFIED before handoff** — in Step 9, after the cost gate: wait for the Space to be RUNNING (`/gradio_api/info` 200), seed the run against the awake Space (`init(resume="never")→log→finish`), and write a truthful `reports/trackio_check.json` (`reachable`/`run_seeded` reflect real command results, never init log lines). train.py then attaches with `resume="allow"` — never creating a second run. Never touch trackio during the probe, probe retries, or any test batch. trackio is pinned to `==0.29.0`; a missing/empty space_id is a silent no-op, never a hard failure.
3. **Checkpoints live on-instance** during training (source of truth). Local sync happens only on terminal state. Optional off-instance durability sync when `projected_hours > 4` or instance is preemptible — at most once per 60 min, gated by time + new checkpoint existence, never by poll count.
4. **Monitoring loop is read-only.** The Haiku health monitor issues exactly one SSH round-trip per check and never writes, kills, or launches remote processes.
5. **MCP tools take precedence over skills.** Use `mcp__lambda-repl__lambda_start_session` instead of the `lambda-session` skill.
6. **All decisions are backed by JSON artifacts.** Missing or unparseable `infra.json`, `profile.json`, `probe_result.json`, or `cost_report.json` is a hard failure.
7. **allow_reuse_existing is budget-aware.** If a running instance is weaker than the requested GPU and `projected_cost <= max_cost_usd`, provision the requested GPU. Only fall back to the weaker instance if the upgrade would exceed budget.
8. **Episodic memory is advisory, never authoritative.** `prior_runs.md` priors inform decisions but current-run artifacts (probe, profile, cost gate) always win on conflict. The finalizer's `RUN_REPORT.md` must be evidence-backed — every metric, decision, finding, and pitfall derives from a specific artifact; missing sources are written as `null`/`unknown`, never fabricated, because future runs consult these reports as priors.

---

