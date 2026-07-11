Run training on Lambda using the CAPO multi-session framework — fine-tuning an
existing protein language model, or pre-training a model from a custom
architecture (the task.md you read in Step 0 says which).

## Run parameters
- run_id:                {run_id}
- task_file:             {local_run_dir}/task.md   (full task specification — read in Step 0)
- model_id:              {model_id}
- fine_tune_strategy:    {fine_tune_strategy}    (linear-probe | lora | full)
- dataset_ref:           {dataset_ref}
- gpu_preference:        {gpu_preference}        (may be "None"; authoritative if set)
- allow_reuse_existing:  {allow_reuse_existing}
- ssh_alias_override:    {ssh_alias_override}    (may be "None"; attach here if given)
- ssh_key_name:          {ssh_key_name}
- key_path:              {key_path}
- local_run_dir:         {local_run_dir}         (runs/<run_id>/)
- remote_run_dir:        {remote_run_dir}        (~/capo_runs/<run_id>/)
- skills_dir:            {skills_dir}
- max_cost_usd:          {max_cost_usd}
- probe_max_retries:     {probe_max_retries}
- trackio_space_id:      {trackio_space_id}

## Dataset source (read before Step 1)
`dataset_ref` above is the EFFECTIVE ref. The orchestrator already resolved the
user's dataset into one of four kinds and wrote `reports/dataset_source.json`
(`kind`, `original_ref`, `effective_ref`, `local_path`, `staged_rel_path`,
`file_format`). Read that file once at Step 0 and act on `kind`:
  - **hf** — `dataset_ref` is a HuggingFace Hub id. Nothing special; the flow is
    unchanged (the instance downloads it with HF_TOKEN).
  - **local** — `dataset_ref` is a relative `<file>` and the file is ALREADY
    staged in the local run dir. The Step 3 `lambda_upload_run` carries it to
    `~/capo_runs/<run_id>/data/` automatically — do nothing extra. probe.py and
    train.py `cd {remote_run_dir}` first, so the relative `<file>` resolves;
    the code-writing loader contract loads it as a local file (§Dataset load).
  - **uri / named** — the dataset is NOT on the instance yet. AFTER the Step 3
    upload and BEFORE the Step 5 probe, fetch it into `<file>` on the
    REMOTE via `lambda_send_to_remote_tmux` (never local rsync):
      • uri  — `cd {remote_run_dir} && mkdir -p data && curl -fL "<original_ref>"
        -o "<staged_rel_path>"` (or `gsutil cp` / `aws s3 cp` for gs://, s3://).
      • named — read `{local_run_dir}/task.md` for the fetch instructions, then
        fetch into `data/` the same way; if `dataset_ref` is still the bare
        label, update your `--dataset` argument to the `<file>` you created.
    Verify the file exists on the remote (`ls -la data/`) before the probe. The
    `--dataset {dataset_ref}` argument already points at the staged relative path
    for uri; for named, substitute the path you fetched to.

    RE-PROFILE ON THE INSTANCE (deferred datasets only — this is where the profile
    plots come from). When `kind` is uri/named, the pre-launch data-profiler could
    not see the data yet, so `profile/profile.json` is a DEFERRED STUB
    (`"profiled_on": "deferred"`, empty `plots`, null `length_percentiles`). Once
    the data is fetched on the remote (above) and BEFORE the Step 5 probe, run a
    profiling pass ON THE INSTANCE so the plots and percentiles actually get
    produced — do NOT leave the stub in place:
      1. Write a small self-contained profiling script that loads the fetched data
         the SAME way train.py's loader will (so the bytes profiled are the bytes
         trained on — for a multi-file/custom pipeline, profile the constructed
         training rows, not a raw dump), computes sequence-length p50/p90/p95/p99,
         label/class balance, and n_samples, and writes profile.json + PNG plots to
         `~/capo_runs/<run_id>/profile/` (plots under `profile/plots/`). Use ONLY
         the mandatory plot colors from your system prompt.
      2. Run it on the instance via `lambda_send_to_remote_tmux`
         (`cd {remote_run_dir} && python <script>`), printing progress lines.
      3. Pull the results back with `lambda_pull_files` into
         `{local_run_dir}/profile/` (both profile.json and profile/plots/*.png).
      This profiling pass is exploratory and MUST be non-fatal: if the custom
      loader makes a full profile impossible, still compute and write
      `length_percentiles` (the probe recipe needs them) and emit a warning rather
      than aborting the run. A crash in plotting alone never blocks training.

## Research context
{research_context}

## Prior-run memory
{memory_context}

## Pre-launch results (completed before this agent started)
{pre_launch_context}

## Transfer rules (HARD CONTRACT — read before any step)

NEVER run `rsync`, `scp`, or any other file-transfer command via Bash or
`lambda_send_to_remote_tmux`. Past runs leaked entire `/Users/...` subtrees
onto the remote because the agent improvised raw rsync with `--relative`,
and other runs dumped files into the shared `~/capo_runs/` root because the
`<run_id>` segment was omitted from the destination. The rsync helper now
rejects both patterns at the wire, so a raw rsync attempt produces a hard
failure rather than silent corruption.

Use only these tools to move bytes:
  - `mcp__lambda-repl__lambda_upload_run(ssh_target, local_run_dir, run_id, key_path)`
    → rsyncs the full local `capo/<run_id>/` to remote `~/capo_runs/<run_id>/`.
    The destination is DERIVED from `run_id` — you no longer pass `remote_run_dir`.
  - `mcp__lambda-repl__lambda_push_run_file(ssh_target, local_run_dir, run_id, src_rel, dst_rel, key_path)`
    → pushes ONE file. `src_rel` / `dst_rel` are relative to the run dir; absolute
    paths and `..` are rejected. Use this in the repair ladder for single-file
    fixes instead of inventing rsync commands.
  - `mcp__lambda-repl__lambda_pull_files(session_id, remote_subpath, local_dest)`
    → remote → local pull.
  - `mcp__lambda-repl__lambda_sync_run_status(ssh_target, remote_run_dir, local_run_dir, key_path)`
    → status.json + metrics.jsonl + logs only.
  - `mcp__lambda-repl__lambda_send_to_remote_tmux(...)` runs SHELL commands —
    never rsync/scp/cp -r through it.

`{remote_run_dir}` is provided below as a READ-ONLY string for `cd <dir>` in
remote shell commands. Never pass it to a transfer tool; the tools derive it
from `run_id`.

## Steps — execute strictly in this order

### 0. Verify pre-launch artifacts. HARD GATE.
mcp__lambda-repl__lambda_ensure_workspace()

Read and verify:
  {local_run_dir}/task.md                          — full task specification; read before writing any code
{prior_runs_artifact_line}{research_artifact_line}  {local_run_dir}/infra.json                       — confirm state=="ready"; extract ssh_alias, hourly_rate_usd
  {local_run_dir}/profile/profile.json             — confirm no errors; extract n_samples, length_percentiles
  {local_run_dir}/reports/model_selection.json             — confirm present; extract driver_script, min_vram_gb

If infra state != "ready" OR profile has critical errors: write reports/failure.json and stop.

Run the structure validator (preflight stage) to confirm canonical dirs are present:
  import subprocess
  subprocess.run(["python", "-m", "capo.utils.checks", "--run-dir", "{local_run_dir}",
                  "--stage", "preflight", "--repair"], check=False)

ip.emit("[fine-tuning] Phase 0 verified: instance=<type> p99=<len> model=<id>")

### 1. Phase 1 — probe recipe (data-profiler result already written)
Read {local_run_dir}/profile/profile.json (written by the data-profiler subagent
in Step 0, or by the on-instance re-profile above for uri/named datasets). Parse
length_percentiles. If it still reads `"profiled_on": "deferred"` or
length_percentiles is null, the on-instance re-profile did not run — do it now (see
"Re-profile on the instance" above) or, at minimum, compute p50/p90/p95/p99
yourself from the fetched data before proceeding. Then write
{local_run_dir}/profile/probe_batch_recipe.json with:
  {{ p50_length, p90_length, p95_length, p99_length,
     recommended_max_seq_length, probe_batch_size,
     probe_effective_batch_size, gradient_accumulation_steps,
     checkpoint_every_n_steps, eval_every_n_steps }}

ip.emit("[fine-tuning] Phase 1 profiling complete: n_samples=... n_labels=... p99=...")

### 2. Stage scripts and canonical structure

**Run-directory contract (non-negotiable, enforced by gate Step 1 + finalizer):**

  Entry points at run root:    train.py, probe.py, requirements.txt, infra.json
  Code package (flat):         src/{{data,models,train,eval,utils}}/  (no deeper nesting)
  Configs:                     configs/{{experiment,training,evaluation}}.yaml
  Operational outputs:         outputs/   (every *.log, status.json, metrics.jsonl, train.pid)
  Scientific outputs:          results/   (eval_metrics.csv, train_metrics.csv, eval_per_class.csv,
                                           metrics.json, results/plots/*.png, results/predictions/*)
  Probe artifacts:             probe/     (probe_result.json, probe.log, probe_batch_recipe.json)
  Pricing artifacts:           pricing/   (lambda-<gpu>.json, cost_report.json)
  Reports & manifests:         reports/   (prior_runs.md, evaluation_report.md, plot_manifest.json,
                                           handoff.json, final_summary.json, trackio_url.txt,
                                           research_findings.json, health/history.jsonl)
  Checkpoints:                 checkpoints/best/, checkpoints/last/
  Scripts:                     scripts/launch_command.sh
  Top-of-run files:            manifest.json, state.json, task.md, RUN_REPORT.md (written by finalizer)

  FORBIDDEN: any directory named fine-tuning/, finetuning/, training/, ft/, logs/, probes/,
             data/, archive/, repairs/, environment/, figures/.
  FORBIDDEN: any *.log, *.csv, *.json, or *.md at run root other than the four listed top-of-run files.
  FORBIDDEN: any Python file at run root other than train.py and probe.py.
  FORBIDDEN: README.md (use RUN_REPORT.md, written by the finalizer).
  If you violate this, gate Step 1 will fail; the finalizer will physically move your files.

Read {skills_dir}/model-selection/SKILL.md for the recipe matching {fine_tune_strategy}.
Write the training entry point at:
  {local_run_dir}/train.py
It is a thin launcher — argument parsing + a call to a `main()` defined in `src/train/`.
Heavy lifting (training loop, optimizer setup, callbacks, loss) lives in
src/train/*.py modules. Data loading + tokenization + collators live in src/data/.
PLM wrappers + LoRA/PEFT setup + heads live in src/models/. Logging / seeding /
config helpers live in src/utils/.

**Both train.py and probe.py MUST call `load_and_validate_dataset(...)` before
any tokenize/filter/map step** — see `{skills_dir}/code-writing/SKILL.md`
§Dataset load + schema validation. The helper detects HF Hub fallback to a
stale cache, validates that `required_columns` (sequence + every label column
from configs/training.yaml + any filter column) are present, and on mismatch
writes `outputs/dataset_load_error.json` + exits 5. This is a hard contract;
generating tokenize calls that bypass the validator is a Phase 1 failure.

**train.py MUST additionally invoke
`load_and_validate_dataset(split="all", deep_check=True, label_columns=...)`
once, before the training loop**, to deep-check the actual splits on the
instance (non-empty val/test, label/sequence alignment, no dupes, and
per-class `pos_weight` sanity — see SKILL.md §Deep validation checks).
On failure this writes `outputs/dataset_load_error.json` with
`failure_category="data_validation_failed"` and exits 5; on pass it writes
`outputs/data_validation.json`. probe.py does NOT run deep_check (it is a
smoke).

**Both train.py and probe.py MUST import the shared `make_train_step` factory
from `src/train/step.py`** (see SKILL.md §Shared train-step + precision
contract). The autocast / mixed-precision call lives in exactly one file —
re-implementing autocast in probe.py or in any Trainer callback is a Phase 1
failure. This is what makes the probe exercise the same backward pass
train.py crashes on.

**train.py MUST run a canary block (200 steps) before entering the main
training loop** — see SKILL.md §Canary block. It reuses `make_train_step`
on the training DataLoader, tracks loss-trend + grad-norm + finiteness, and
writes `outputs/canary_summary.json`. On a degenerate signal (NaN/Inf loss,
grad-norm explosion, loss-slope >= 0 over 200 steps) it writes
`outputs/canary_failure.json` with `failure_category="canary_failed"` and
exits 6. This is the in-training-loop counterpart to the probe gate: the
probe proves the backward pass runs once, the canary proves it learns. It
also gates the optimizer / learning-rate before a full multi-hour run
commits to broken hyperparameters.

**train.py MUST initialise trackio with `resume="allow"` at startup — BEFORE
the canary — attaching to the run the orchestrator seeded in Step 9** (same
`project="capo-ft"`, same `name` from `--trackio-run`, same
`--trackio-space-id`). `resume="allow"` resumes that already-created run
instead of spawning a duplicate; initialising before the canary means the
canary and the full run both log into it and the Space stays awake throughout.
NEVER pass `resume="never"` in train.py. Call `trackio.finish()` on every exit
path (success, canary-fail, exception). If `--trackio-space-id` is empty, the
tracker is a silent no-op — never a hard failure.

Write the evaluation entry point at:
  {local_run_dir}/src/eval/evaluate.py
It must: load a checkpoint from configs/evaluation.yaml, run inference on the
test set, compute all metrics, write results/metrics.json + results/eval_metrics.csv,
and write reports/evaluation_report.md. train.py imports run_final_evaluation()
from src/eval/evaluate.py and invokes it at end of training. It must also accept
a CLI flag `--eval-only --checkpoint <path>` so the finalizer can re-run eval
against an existing checkpoint if results are missing.

Write the probe entry point at:
  {local_run_dir}/probe.py
It must accept CLI args:
  --model-id --seq-length --batch-size --out-json
and implement the probe contract (forward-only, forward+backward, write
probe_result.json, exit 0 on success). probe.py is also a thin launcher —
shared logic lives under src/. The probe script must NOT import or initialize
trackio — it is a pure compute validation script.

Write experiment config files:
  {local_run_dir}/configs/experiment.yaml   — run_id, task, model, method, seed, paths
  {local_run_dir}/configs/training.yaml     — epochs, batch_size, learning_rate, etc.
  {local_run_dir}/configs/evaluation.yaml   — checkpoint_path, metrics, output_dir

Write requirements.txt at the run root pinning every Python dep used by train.py /
probe.py / src/. The remote install step (Step 4) does `pip install -r requirements.txt`.
Any run that loads a HuggingFace/parquet dataset MUST pin the data stack explicitly
— `datasets`, `pyarrow`, `fastparquet` — even though `datasets` "should" pull them:
a base image without pyarrow makes load_dataset fail at Stage 1 with "Unable to find
a usable engine", which is the most common wasted-launch cause after the GPU kernel.

Update {local_run_dir}/manifest.json to record the scripts and configs just written.
Do NOT create README.md — RUN_REPORT.md is the user-facing summary and is written
by the finalizer.

Write the canonical launch shell entry point at {local_run_dir}/scripts/launch_command.sh.
This is the contract consumed by the resume path: it must contain only the
`python train.py ...` portion (no nohup, no redirect, no pid-capture), so the
resume flow can append `--resume-from-checkpoint <path>` cleanly. It is invoked
by the launcher block in Step 10.

### 3. Upload run directory
mcp__lambda-repl__lambda_upload_run(
  ssh_target=<resolved ssh_alias>,
  local_run_dir="{local_run_dir}",
  run_id="{run_id}",
  key_path="{key_path}")

For single-file repair pushes (Step 6), use
`mcp__lambda-repl__lambda_push_run_file(ssh_target, local_run_dir, run_id,
src_rel, dst_rel, key_path)` — never `rsync` via Bash.

### 4. Install dependencies + verify the environment (HARD GATE)

mcp__lambda-repl__lambda_send_to_remote_tmux(
  ssh_alias=<resolved ssh_alias>,
  command="cd {remote_run_dir} && pip install -q -r requirements.txt 'trackio==0.29.0'",
  key_path="{key_path}")
Poll lambda_get_output until pip returns.

trackio is PINNED to `==0.29.0` (both here and in requirements.txt). Do not
leave it unpinned — an unpinned `trackio` silently installs a different storage
backend across runs (older dataset-backed vs newer bucket-backed), which is a
known cause of runs never materialising in the Space. 0.29.0 supports
`resume="allow"` (Step 9/10 seed-then-resume) and the persistent metrics bucket.

requirements.txt must already pin EVERY dep used by train.py / probe.py / src/,
including any GPU-kernel package the model needs at RUNTIME. Read the model's
SKILL.md dependency section before writing it — e.g. boltz needs
`cuequivariance-torch cuequivariance cuequivariance-ops-cu12`, which
`pip install boltz` does NOT pull and which only fails at the first GPU forward
(see skills/model-inference/boltz/SKILL.md §GPU kernel dependency). A dep that
is missing here is the single most common cause of a wasted GPU launch.

Then VERIFY every critical module actually imports on THIS instance — before
spending any GPU money. Missing native-dep packages (pyarrow, datasets, rdkit,
the GPU kernel) fail here in seconds instead of mid-training for dollars:

mcp__lambda-repl__lambda_send_to_remote_tmux(
  ssh_alias=<resolved ssh_alias>,
  command=(
    "cd {remote_run_dir} && python - <<'PY'\n"
    "import importlib, json, pathlib\n"
    "# Fill mods from the model SKILL.md import names; ALWAYS include the data\n"
    "# stack and any GPU-kernel package. boltz example shown:\n"
    "mods = ['torch', 'datasets', 'pyarrow', 'rdkit', 'boltz', 'cuequivariance_torch']\n"
    "out = {{}}\n"
    "for m in mods:\n"
    "    try:\n"
    "        importlib.import_module(m)\n"
    "        out[m] = 'ok'\n"
    "    except Exception as e:\n"
    "        out[m] = 'FAIL: ' + type(e).__name__ + ': ' + str(e)\n"
    "try:\n"
    "    import torch\n"
    "    out['cuda'] = bool(torch.cuda.is_available())\n"
    "except Exception as e:\n"
    "    out['cuda'] = 'FAIL: ' + str(e)\n"
    "pathlib.Path('reports').mkdir(exist_ok=True)\n"
    "pathlib.Path('reports/env_check.json').write_text(json.dumps(out, indent=2))\n"
    "bad = [k for k, v in out.items() if isinstance(v, str) and v.startswith('FAIL')]\n"
    "if out.get('cuda') is not True:\n"
    "    bad.append('cuda')\n"
    "print('ENV_CHECK_FAIL=' + ','.join(sorted(set(bad))) if bad else 'ENV_CHECK_OK')\n"
    "PY"
  ),
  key_path="{key_path}")
Poll lambda_get_output, then lambda_pull_files reports/env_check.json.

If `cuda` is not True: the instance has no usable GPU (driver/hardware — NOT a
pip-fixable problem). Write LOCAL {local_run_dir}/reports/aborted_env_check.json,
emit ip.emit("[fine-tuning] no usable CUDA device — aborting"),
state=aborted_env_check, STOP.

If the output contains `ENV_CHECK_FAIL` on one or more MODULES (cuda is True):
attempt ONE in-place repair before aborting — a missing pip package is the most
common and the most cheaply fixable cause, and the goal is to fine-tune the model,
not to bail on a one-line install. Reinstall requirements plus the named failing
packages, then re-run the SAME import gate exactly once:
  mcp__lambda-repl__lambda_send_to_remote_tmux(
    ssh_alias=<resolved ssh_alias>,
    command=(
      "cd {remote_run_dir} && "
      "pip install -q --extra-index-url https://pypi.nvidia.com "
      "-r requirements.txt <failing-package-names>"
    ),
    key_path="{key_path}")
  # Map each failing IMPORT name to its pip package: datasets→datasets,
  # pyarrow→pyarrow, fastparquet→fastparquet, rdkit→rdkit, and any GPU-kernel
  # import (e.g. cuequivariance_torch) → the full package set named in that
  # model's SKILL.md (boltz: see skills/model-inference/boltz/references/
  # boltz2-cuda.lock). The --extra-index-url is harmless for pure-PyPI packages.
Re-run the env-check heredoc. If it now prints ENV_CHECK_OK → proceed to the probe.
If it STILL prints ENV_CHECK_FAIL: write LOCAL
{local_run_dir}/reports/aborted_env_check.json with the still-failing modules, emit
  ip.emit("[fine-tuning] env verification failed after repair: <modules> — aborting before GPU spend")
state=aborted_env_check. STOP. Do NOT run the probe — a missing dependency caught
here costs seconds; the same one caught at the first GPU forward wastes the whole
probe/launch. This is the cheap analogue of the cost gate.

### 5. Feasibility probe (attempt 1 of {probe_max_retries})
mcp__lambda-repl__lambda_send_to_remote_tmux(
  ssh_alias=<resolved ssh_alias>,
  command="cd {remote_run_dir} && python probe.py "
          "--model-id {model_id} --seq-length <p99> "
          "--batch-size <probe_batch_size> "
          "--dataset {dataset_ref} "
          "--out-json probe/probe_result.json "
          "2>&1 | tee probe/probe.log",
  key_path="{key_path}")
Poll every 10s (expected <= 5 min). Then lambda_pull_files probe/ outputs/.
Parse probe/probe_result.json.

The `--dataset` flag (NEW) makes probe.py exercise `load_and_validate_dataset
+ tokenize_dataset` on the first 64 rows of `dataset_ref` before the
forward+backward pass — see `skills/code-writing/SKILL.md` §Dataset load +
schema validation and §Probe script contract. This catches schema drift
(missing `sequence` column, silent HF Hub fallback to a stale cache) at the
gate instead of 30 minutes into the nohup training run.

After pulling, if `outputs/dataset_load_error.json` exists on the remote,
read its `failure_category`. Route as:
  - `data_schema_mismatch` → existing repair branch (re-invoke data-profiler
    with stricter flags, see Step 6 below).
  - `data_validation_failed` → SAME repair branch as `data_schema_mismatch`
    (override `probe_result.failure_category` regardless of what the probe
    wrote; the JSON wins). This category surfaces from train.py's
    `deep_check=True` invocation in Step 10's pre-launch line — if you see
    it after the probe, it means an earlier failed launch left a stale file;
    treat it the same way.
If the probe reports `hub_lookup_failed=true` but `dataset_load_ok=true`,
emit a `[fine-tuning]` warning line with the cache_mtime_iso and continue —
the schema validated against the cached snapshot.

### 6. Probe failure routing — the 3-step gate repair ladder

Routing rules are now centralised in `src/capo/orchestration/gate.py` (the
`ThreeStepGate` state machine) and the compact-packet contract in
`src/capo/orchestration/probe_failure_packet.py`. Follow them exactly — do
NOT improvise.

**Mechanical failures (no LLM repair, no subagent):**
| failure_category    | action                                                  |
|---------------------|---------------------------------------------------------|
| oom                 | Halve probe_batch_size; retry. Second OOM → abort_too_large |
| nan_inf             | Enable bf16; lower lr 10×; retry                        |
| resource_mismatch   | Halve probe_batch_size; double grad_accum_steps; retry  |

Use `gate.mechanical_fix_oom`, `gate.mechanical_fix_nan_inf`,
`gate.mechanical_fix_resource_mismatch` — they patch probe_batch_recipe.json
deterministically. Never call the code-repair-critic for these.

**`script_bug` — the repair ladder (bounded, no full-prompt reloads):**

For Attempts 1 and 2 (same orchestrator, you):
  - Build a compact packet via `probe_failure_packet.build_compact_packet(
    failing_file, failure_category='script_bug', traceback, expected_schema,
    budget, history)`. Write it to
    `reports/repairs/attempt_<N>_packet.json`.
  - Read ONLY the packet (≤8 KB), the failing file, and
    `skills/code-writing/SKILL.md`. Do NOT reload the full prompt template or
    profile.json — the packet has what you need.
  - Edit the failing file in place. Re-upload only the patched file via
    `mcp__lambda-repl__lambda_push_files`. Re-run the probe.

For Attempt 3 (code-repair-critic subagent):
  - Build the packet again with `history` listing both prior attempts.
  - Invoke the subagent via the Agent tool: `subagent_type='code-repair-critic'`,
    `prompt='compact_packet_path={local_run_dir}/reports/repairs/attempt_3_packet.json
    local_run_dir={local_run_dir}'`.
  - The critic returns a fenced diff. Save it to
    `reports/repairs/attempt_3.diff`. Apply with `git apply --check` then
    `git apply` (run via Bash from {local_run_dir}). If the diff fails to
    apply OR the critic returns `INSUFFICIENT_INFO:`, treat the ladder as
    EXHAUSTED.

If the ladder exhausts on a `script_bug`:
  - Read `model_selection.json`. If `candidates[candidate_index + 1]` exists,
    advance the candidate, regenerate scripts for the new model (Step 2), and
    re-enter the gate from Step 5. Increment `candidate_index` in
    `reports/gate_state.json`.
  - If no more candidates: write `reports/probe_failure_packet.json` with
    `probe_attempts`, the last `probe_result`, the sequence of fixes tried,
    and `exhausted_step=2`; state=probe_failed; STOP.

**`data_schema_mismatch` — never to the critic.**
  - First attempt: re-invoke the `data-profiler` subagent with stricter flags
    (e.g. force task_type / label_column re-detection). Patch profile.json,
    re-run the probe.
  - If still mismatched: the gap is user-only knowledge. Build the question
    payload (header='Schema gap', single free-text option, answer_target=
    'profile.label_semantics'), write it to
    `reports/pending_question.json`, update state.json with
    `paused=True, pause_reason='probe_data_schema_user_only',
    pending_question_path='reports/pending_question.json'`, emit:
    `ip.emit("[fine-tuning] probe paused for user input — run capo_resume after answering")`
    and exit. capo_resume picks up here.

**Hard rule:** the bound is `probe_max_retries={probe_max_retries}` total
attempts for a single candidate. Attempts 1+2 (orchestrator self-repair) and
Attempt 3 (critic) all count against this bound.

### 7. Cost report + checkpoint cadence
Read {local_run_dir}/pricing/*.json. Compute:
  step_latency_s  = backward_latency_s
  total_steps     = epochs * ceil(n_samples / effective_batch_size)
  projected_hours = total_steps * step_latency_s / 3600
  projected_cost  = projected_hours * hourly_rate_usd
Write {local_run_dir}/pricing/cost_report.json with all inputs + projections.

Compute checkpoint cadence (optimizer steps, not micro-batches):
  target_checkpoint_interval_s = 900   # worst-case lost work ≤ ~15 min
  time_based_n = math.ceil(
      target_checkpoint_interval_s / max(step_latency_s, 1e-6)
  )
  checkpoint_every_n_steps = max(50, time_based_n)   # 50-step efficiency floor

  # Long-run cap: only applied when the run is long enough to make it
  # meaningful (>= 250 steps ensures the floor itself does not win).
  if total_steps >= 250:
      checkpoint_every_n_steps = min(
          checkpoint_every_n_steps,
          max(1, total_steps // 5),
      )

Update probe_batch_recipe.json with checkpoint_every_n_steps = <computed value>.
The training command in Step 10 passes --checkpoint-every checkpoint_every_n_steps.
The training script must write a final checkpoint unconditionally at end of
training even if the cadence would not otherwise trigger.

Determine durability sync eligibility and record in cost_report.json:
  instance_is_preemptible = read from infra.json (field: preemptible, if absent assume False)
  enable_durability_sync  = (projected_hours > 4) or instance_is_preemptible
Record checkpoint_every_n_steps and enable_durability_sync in cost_report.json.

### 8. Cost gate (automatic, no user prompt)
If projected_cost_usd > {max_cost_usd}:
  ip.emit("[fine-tuning] projected ${{p:.2f}} > max ${{m:.2f}} — aborting")
  Write {local_run_dir}/reports/abort_over_budget.json.
  state=aborted_over_budget. STOP.
Else:
  ip.emit("[fine-tuning] cost gate passed: projected ${{p:.2f}} / max ${{m:.2f}}")

### 9. Seed the trackio run and VERIFY it (after cost gate, before launch)

The HF Space "{trackio_space_id}" was already created, seeded
(README/requirements/app.py) and wired to a metrics bucket by the orchestrator
before this prompt ran; {local_run_dir}/reports/trackio_url.txt is already
written. Your job HERE is to make the run ACTUALLY appear on the dashboard
before you hand off — not to trust that init "printed success". trackio prints
"Found existing space" / "Created new run" / "View dashboard" even when the
Space is asleep and every metric silently buffers locally on this ephemeral
instance and is lost when it dies. So you must seed the run against a
confirmed-awake Space and verify it, then let train.py attach to it.

If "{trackio_space_id}" is the empty string, the orchestrator could not set up
a dashboard (no HF_TOKEN or seeding failed). Write
{local_run_dir}/reports/trackio_check.json =
{{"space_id": "", "reachable": false, "run_seeded": false, "run_verified": false, "run_name": "{run_id}"}},
then skip to Step 10 — training still runs, just with no live curve (non-fatal).

Otherwise do ALL of 9a–9d, in order. The flat dashboard URL replaces the `/`
in the space id with `-` (e.g. "owner/capo-trackio" → "owner-capo-trackio.hf.space").

9a. WAIT for the Space to be RUNNING (init-time log lines do NOT prove this).
    The remote box already has HF_TOKEN exported — poll its Gradio API:
      mcp__lambda-repl__lambda_send_to_remote_tmux(
        ssh_alias=<resolved ssh_alias>,
        command=("for i in $(seq 1 16); do "
                 "code=$(curl -s -o /dev/null -w '%{{http_code}}' "
                 "-H \"Authorization: Bearer $HF_TOKEN\" "
                 "https://<flat-space-url>/gradio_api/info); "
                 "echo attempt $i http=$code; "
                 "[ \"$code\" = 200 ] && echo SPACE_UP && break; "
                 "sleep 15; done"),
        key_path="{key_path}")
    Poll lambda_get_output until it returns. If SPACE_UP never prints (~4 min),
    the Space failed to boot: write trackio_check.json with reachable/run_seeded
    /run_verified all false, emit ip.emit("[trackio] Space not reachable —
    proceeding without dashboard"), and go to Step 10 (non-fatal).

9b. SEED the run on the remote, against the now-awake Space, using the SAME
    trackio version train.py will use. resume='never' creates it fresh:
      mcp__lambda-repl__lambda_send_to_remote_tmux(
        ssh_alias=<resolved ssh_alias>,
        command=("cd {remote_run_dir} && python -c \""
                 "import trackio; "
                 "trackio.init(project='capo-ft', name='{run_id}', "
                 "space_id='{trackio_space_id}', resume='never', "
                 "config={{'seeded_by':'orchestrator','run_id':'{run_id}'}}); "
                 "trackio.log({{'seed_step': 0}}); "
                 "trackio.finish(); print('SEED_OK')\""),
        key_path="{key_path}")
    Poll lambda_get_output until it prints SEED_OK. This creates the run in the
    Space's store and confirms end-to-end connectivity. train.py (Step 10)
    ATTACHES to this same run with resume="allow" — it must NOT create a second.

9c. VERIFY the run now exists (best-effort; do not infer from log text):
      mcp__lambda-repl__lambda_send_to_remote_tmux(
        ssh_alias=<resolved ssh_alias>,
        command=("TRACKIO_SPACE_ID={trackio_space_id} "
                 "trackio list runs --project capo-ft --json 2>/dev/null "
                 "| python -c \"import sys,json; "
                 "d=json.load(sys.stdin); "
                 "runs=d if isinstance(d,list) else d.get('runs',[]); "
                 "print('RUN_PRESENT' if any('{run_id}' in str(r) for r in runs) else 'RUN_ABSENT')\""),
        key_path="{key_path}")
    Poll lambda_get_output. RUN_PRESENT → confirmed. If RUN_ABSENT or the CLI
    errors (its shape varies across trackio builds) but 9b printed SEED_OK,
    treat the run as seeded — SEED_OK against a 200 Space is authoritative.

9d. Write a TRUTHFUL {local_run_dir}/reports/trackio_check.json:
      {{
        "space_id":       "{trackio_space_id}",
        "dashboard_url":  "https://huggingface.co/spaces/{trackio_space_id}",
        "flat_url":       "https://<flat-space-url>/",
        "reachable":      <true ONLY if 9a printed SPACE_UP>,
        "run_seeded":     <true ONLY if 9b printed SEED_OK>,
        "run_verified":   <true ONLY if 9c printed RUN_PRESENT>,
        "run_name":       "{run_id}",
        "verified_via":   "gradio_api/info (9a) + remote trackio seed (9b) + trackio list runs (9c)",
        "checked_at_iso": "<ISO8601 UTC>"
      }}
    reachable / run_seeded / run_verified MUST reflect the ACTUAL command
    results above — never set them true from init log lines. Then move to Step 10.

### 10. Launch full training (nohup in capo_remote), confirm UP, hand off, exit

Before launching, persist the exact command to scripts/launch_command.sh on
the REMOTE. This file is the contract consumed by the resume path — it must
contain only the `python train.py ...` portion (no nohup, no redirect, no
pid-capture), so the resume flow can append `--resume-from-checkpoint <path>`
cleanly.

**Pre-launch on-instance data validation (HARD GATE).** Before the nohup
launch, run a one-shot deep-check on the instance. This catches degenerate
splits / pos_weights / alignment bugs in the *actual* data train.py will
read, on the *actual* machine training will run on. If it fails, the
instance shell exits non-zero; outputs/dataset_load_error.json is pulled
locally and routed via the same `data_validation_failed` branch as Step 6.

mcp__lambda-repl__lambda_send_to_remote_tmux(
  ssh_alias=<resolved ssh_alias>,
  command=(
    "cd {remote_run_dir} && "
    "python -c \"import json, pathlib, sys; "
    "from src.data.dataset import load_and_validate_dataset; "
    "import logging; lg=logging.getLogger('validate'); lg.setLevel(logging.INFO); "
    "cfg=json.loads(pathlib.Path('configs/training.yaml').read_text() if False else '{{}}'); "
    "import yaml; cfg=yaml.safe_load(open('configs/training.yaml')); "
    "load_and_validate_dataset(cfg['dataset_ref'], split='all', "
    "required_columns=cfg['required_columns'], "
    "label_columns=cfg['label_columns'], "
    "deep_check=True, "
    "allow_zero_positive_classes=bool(cfg.get('allow_zero_positive_classes', False)), "
    "logger=lg)\" || exit 7"
  ),
  key_path="{key_path}")
Poll lambda_get_output until the validator returns. On exit 7, lambda_pull_files
`outputs/dataset_load_error.json` + `outputs/data_validation.json`, emit
`[fine-tuning] data validation failed — see outputs/dataset_load_error.json`,
write `reports/failure.json` with `failure_category="data_validation_failed"`,
and exit. The Haiku monitor is not started.
On success (exit 0), proceed to the nohup launch.

mcp__lambda-repl__lambda_send_to_remote_tmux(
  ssh_alias=<resolved ssh_alias>,
  command=(
    "cd {remote_run_dir} && "
    "mkdir -p outputs results/plots results/predictions reports checkpoints/best checkpoints/last scripts && "
    "cat > scripts/launch_command.sh <<'LAUNCH_EOF'\n"
    "export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"\n"
    "export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"\n"
    "python train.py "
    "--model-id {model_id} --strategy {fine_tune_strategy} "
    "--dataset {dataset_ref} --epochs <selected_epochs> "
    "--batch-size <eff_bs> --grad-accum <ga> "
    "--checkpoint-every <N> --eval-every <M> "
    "--checkpoints checkpoints/ "
    "--results results/ "
    "--outputs outputs/ "
    "--trackio-project capo-ft --trackio-run {run_id}{trackio_cli_arg}\n"
    "LAUNCH_EOF\n"
    "chmod +x scripts/launch_command.sh && "
    "nohup bash -c '"
    "bash scripts/launch_command.sh; rc=$?; "
    "if [ $rc -ne 0 ]; then "
    "python3 - \"$rc\" <<PYEOF\n"
    "import json, sys, time, pathlib\n"
    "p = pathlib.Path(\"outputs/status.json\")\n"
    "try:\n"
    "    st = json.loads(p.read_text())\n"
    "except Exception:\n"
    "    st = {{}}\n"
    "if st.get(\"state\") not in (\"completed\", \"failed\"):\n"
    "    st.update({{\"state\": \"failed\", \"returncode\": int(sys.argv[1]), \"updated_at\": time.strftime(\"%Y-%m-%dT%H:%M:%SZ\", time.gmtime())}})\n"
    "    p.write_text(json.dumps(st))\n"
    "PYEOF\n"
    "fi; exit $rc' "
    "> outputs/train.log 2>&1 & echo $! > outputs/train.pid"
  ),
  key_path="{key_path}")

EXIT-TRAP RATIONALE (do not omit the wrapper). train.py is also required to
record state="failed" on any uncaught exception, but a SIGKILL/OOM or a crash
in a child subprocess (e.g. `boltz predict`) can still kill it before it
writes anything — leaving status.json frozen at "running" and the monitor
blind. The `bash -c` wrapper guarantees a terminal "failed" status the moment
the launch command exits non-zero. scripts/launch_command.sh itself stays PURE
(only `python train.py ...`, the resume contract). The recorded pid is the
wrapper, whose process args still contain "launch_command", so the Haiku
monitor's command-verified liveness check still matches it.
NOTE: HF_TOKEN and HUGGING_FACE_HUB_TOKEN are already exported in the
capo_remote tmux session by the orchestrator (Step -1). train.py can call
`huggingface_hub.HfApi()` and trackio.init() directly without re-loading
credentials. The token also lives at ~/.cache/huggingface/token (mode 600).
train.py's trackio.init() uses `resume="allow"` so it attaches to the run you
seeded in Step 9 (`--trackio-run {run_id}`) rather than creating a new one.

Replace <selected_epochs> with the number from epoch_plan.json, and <eff_bs>
/ <ga> / <N> / <M> with the values you computed in Step 7.

#### 10a. Confirm training is UP (bounded, max ~2 min)

Issue up to THREE SSH snapshots, 40s apart, reading:
  ssh -i {key_path} -o StrictHostKeyChecking=no <host> "
    R={remote_run_dir}
    P=\$(cat \$R/outputs/train.pid 2>/dev/null || echo '')
    echo =PID=\$P
    [ -n \$P ] && ps -p \$P >/dev/null 2>&1 && echo =ALIVE=1 || echo =ALIVE=0
    echo =STATUS=
    cat \$R/outputs/status.json 2>/dev/null
    echo =METRICS_ROWS=
    wc -l < \$R/outputs/metrics.jsonl 2>/dev/null || echo 0
    echo =LOG_TAIL=
    tail -20 \$R/outputs/train.log 2>/dev/null
  "

Training is UP when ALIVE=1 AND at least one of:
  • METRICS_ROWS ≥ 1, OR
  • STATUS has "state": "running" | "training".

If all three snapshots report ALIVE=0 with no metrics written: this is a
startup crash. Sync outputs/train.log to {local_run_dir}/outputs/ (one
lambda_pull_files), write {local_run_dir}/reports/failure.json with the
tail of train.log as evidence, emit:
  ip.emit("[fine-tuning] startup crash — training did not come up")
and exit.

#### 10b. Write the handoff contract and EXIT

Once training is UP:
  1. Read the pid from remote outputs/train.pid.
  2. Read the trackio URL from {local_run_dir}/reports/trackio_url.txt
     (written in Step 9; may be empty string if trackio init did not produce
     a URL).
  3. Compute the GPU-active deadline. The Haiku monitor + a deterministic
     orchestrator backstop use this to catch a silent crash that leaves the GPU
     idle behind a stale status.json — the costliest failure mode (an idle GPU
     billing for hours). Set it to the LATEST time by which the GPU must be
     doing real work:
       expected_gpu_active_by_iso = launched_at
         + (time to load/download model weights)
         + (any CPU-only prep before the first GPU op)
         + a 10-minute safety margin.
     Use the strategy's REAL startup profile, not a generic guess:
       • External-inference strategies (e.g. boltz custom-affinity-head): weights
         load and the first `boltz predict` forward happen within ~10 min of
         launch once download finishes → budget ≈ 20–30 min total.
       • HF tokenization-heavy full/LoRA runs: tokenization can be CPU-bound for
         10–60 min → budget = your tokenization estimate + 10 min.
     Also set heartbeat_timeout_sec = max(600, 3 × the interval at which train.py
     refreshes status.json's updated_at).
  4. Write LOCAL {local_run_dir}/reports/handoff.json:
     {{
       "ssh_alias":                  "<resolved alias>",
       "remote_run_dir":             "{remote_run_dir}",
       "pid":                        <int>,
       "trackio_url":                "<url>" | null,
       "launched_at_iso":            "<ISO8601 UTC>",
       "expected_gpu_active_by_iso": "<ISO8601 UTC>",
       "heartbeat_timeout_sec":      <int>
     }}

  ip.emit("[fine-tuning] training launched — handing off to Haiku monitor")

EXIT IMMEDIATELY after writing handoff.json. Do NOT monitor, do NOT poll,
do NOT sync artifacts. A separate Haiku monitoring agent watches training
health every 1–5 minutes, and a Sonnet finalizer runs after the monitor
signals a terminal state and handles all remote→local transfers and the
final summary.

Return a short structured answer noting:
  run_id, instance_type, ssh_alias, instance_reused, projected_cost_usd,
  probe_attempts, trackio_url, and the literal string "training_launched".
