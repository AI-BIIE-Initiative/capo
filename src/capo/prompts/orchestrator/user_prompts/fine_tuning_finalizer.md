## Inputs (use verbatim)

- handoff_kind:        {handoff_kind}       ("completed" | "failed" | "escalation")
- escalation_reason:   {escalation_reason}  (empty unless escalation)
- run_id:              {run_id}
- ssh_alias:           {ssh_alias}
- key_path:            {key_path}
- remote_run_dir:      {remote_run_dir}
- local_run_dir:       {local_run_dir}
- trackio_url:         {trackio_url}
- hub_push_config:     {hub_push_config}
- auto_terminate_on_failure: {auto_terminate_on_failure}  ("True" | "False")
- ssh_key_name:        {ssh_key_name}

- last_health_report (JSON from the Haiku monitor, or "null"):
{last_health_report}

- handoff.json (as Sonnet pre-launch wrote it):
{handoff_json}

## Procedure

### 1. Pull artifacts (required)

Use mcp__lambda-repl__lambda_pull_files to copy these remote subdirs into the
local run directory. One call per subdir.

  {remote_run_dir}/outputs/      ->  {local_run_dir}/outputs/
  {remote_run_dir}/results/      ->  {local_run_dir}/results/
  {remote_run_dir}/checkpoints/  ->  {local_run_dir}/checkpoints/
  {remote_run_dir}/reports/      ->  {local_run_dir}/reports/
  {remote_run_dir}/src/          ->  {local_run_dir}/src/
  {remote_run_dir}/configs/      ->  {local_run_dir}/configs/
  {remote_run_dir}/pricing/      ->  {local_run_dir}/pricing/
  {remote_run_dir}/profile/      ->  {local_run_dir}/profile/
  {remote_run_dir}/probe/        ->  {local_run_dir}/probe/
  {remote_run_dir}/scripts/      ->  {local_run_dir}/scripts/

Pull `outputs/` FIRST and always — it holds the remote `train.log`/`stdout.log`,
the authoritative training log the user does not otherwise have locally (the local
`outputs/run.log` is only the orchestrator log). If every other pull fails, still
try `outputs/` alone so the training log lands locally.

On repeated sync failure, set failure.cause = "sync_failed" in the summary and
proceed with whatever artifacts are already local (do NOT fabricate metrics, and
per §4 case 3 the terminal_state is "unknown", not "failed").

### 2. Enforce structure (--repair)

Run the structure validator with --repair. This PHYSICALLY MOVES misplaced
files into their canonical subdir (e.g. a stray train.log at run root gets
moved into outputs/, eval_metrics.csv → results/, evaluation_report.md →
reports/). It also deletes any forbidden subdir (fine-tuning/, logs/, etc)
after migrating its contents, and removes the deprecated README.md.

  import subprocess, json as _json
  result = subprocess.run(
      ["python", "-m", "capo.utils.checks", "--run-dir", "{local_run_dir}",
       "--stage", "postrun", "--repair", "--json"],
      capture_output=True, text=True, check=False
  )
  Write `result.stdout` to {local_run_dir}/reports/structure_validation.json.
  Parse it; capture `repaired_files` and `repaired_dirs` for final_summary.

After --repair: every artifact mentioned below lives at its canonical path.

### 3. Read local files

After sync + repair, read from the LOCAL copies only:
  - {local_run_dir}/outputs/status.json                     (if present)
  - last 50 rows of {local_run_dir}/outputs/metrics.jsonl
  - last 200 lines of {local_run_dir}/outputs/train.log     (authoritative)
  - list {local_run_dir}/checkpoints/                    (if present)
  - {local_run_dir}/checkpoints/best/                    (does it exist and have files?)
  - {local_run_dir}/checkpoints/last/                    (does it exist and have files?)
  - {local_run_dir}/pricing/cost_report.json             (if present)
  - {local_run_dir}/results/metrics.json                 (if present — final eval metrics)

### 4. Determine the TRUE terminal state from evidence (do this BEFORE branching)

`handoff_kind` is a HINT, not the verdict (see the system prompt "GROUND TRUTH
OVER HANDOFF"). Compute the real terminal state from what you can observe:

  1. COMPLETION EVIDENCE — the run completed if ANY of these are true:
       - {local_run_dir}/results/metrics.json exists and has real eval numbers, OR
       - {local_run_dir}/results/eval_metrics.csv has a split="test" (or "val") row, OR
       - {local_run_dir}/checkpoints/best/ exists AND contains config.json +
         model.safetensors* , OR
       - the remote/local outputs/status.json has "state": "completed".
     If completion evidence exists → **terminal_state = "completed"**, EVEN IF
     handoff_kind == "failed". (A recovery relaunch or a post-monitor finish is
     exactly why the hint goes stale.) Delete/ignore any stale
     reports/post_launch_failure.json and finalize as a completed run.

  2. FAILURE EVIDENCE — the run failed if there is NO completion evidence AND
     either a crash signature is present in outputs/train.log / train_err.tail.log
     OR the remote status.json has "state": "failed". Proceed with the failure
     branch below.

  3. UNVERIFIABLE — the artifact pull failed AND there is neither completion nor
     failure evidence locally. You cannot see the remote, so you do not know what
     happened. **terminal_state = "unknown"** (the single could-not-verify value —
     NOT "failed", and there is no "indeterminate"/"partial" value);
     failure.cause = "sync_failed" (record the exact pull error). Do NOT claim the
     run failed, "never started", or produced "zero checkpoints" — you have no
     basis for any of those claims.

Only after fixing the true terminal state, use the branches below to fill in the
details.

### 4b. Branch on the resolved terminal state

If terminal_state == "completed":
  - Read the last row of outputs/metrics.jsonl. That row is the best terminal metric.
  - Determine final_metrics: pick whichever of val_mcc / val_auc / val_f1 /
    val_accuracy / val_loss / train_loss are present. Include all non-null ones.
  - If src/eval/evaluate.py wrote results/metrics.json, include its top-line scores.
  - final_model_path = checkpoints/best/ if it exists and is non-empty, else null.
  - failure = null.

If terminal_state == "failed" (confirmed by §4 failure evidence — NOT merely by
a stale handoff_kind hint):
  - **First check {local_run_dir}/reports/post_launch_failure.json.** If it
    exists AND §4 did not find completion evidence, the orchestrator's diagnostics
    phase already classified the failure into a precise category. PREFER that
    classification over your own log scanning:
      - failure.cause = post_launch_failure.failure_category
        (one of data_schema_mismatch, script_bug, oom, nan_inf,
         hub_fallback_stale_cache, unknown)
      - failure.evidence = an object containing:
          summary, failing_file, missing_columns, required_columns,
          observed_columns, hub_lookup_failed, cache_path, cache_mtime_iso,
          remediation
        plus the last 20 lines of outputs/train_err.tail.log (if present)
        or outputs/train.log otherwise.
      - Set failure.recoverable = post_launch_failure.recoverable (bool).
  - If post_launch_failure.json is absent (e.g. SSH was unreachable at
    diagnosis time), fall back to scanning the last 200 lines of
    outputs/train.log for known failure signatures:
      * "cuda_oom"          : "CUDA out of memory"
      * "nan_loss"          : "nan" / "inf" in loss or NaN guard messages
      * "traceback"         : Python traceback present but no specific cause
      * "sigkill_oom"       : "Killed" / OOM-killer / exit code 137
      * "disk_full"         : "No space left on device"
      * "unknown_failure"   : none of the above
    Emit failure.cause = one label from the list above and
    failure.evidence = the last 20 lines verbatim (as a single string).
  - If any checkpoints exist under checkpoints/, the most recent is the
    recoverable state for a future --resume-from-checkpoint.

If terminal_state == "unknown" (§4 case 3 — remote unverifiable):
  - failure.cause = "sync_failed"; failure.evidence = the exact pull error(s) plus
    whatever local artifacts DO exist (list them).
  - Write RUN_REPORT.md honestly: state that the remote could not be reached at
    finalize time, so completion could not be confirmed or denied. Do NOT assert
    the run failed or never trained. Suggest the user re-run the finalizer once the
    instance is reachable, or inspect the remote run dir directly.
  - Skip the Hub push (nothing verified to push).

If handoff_kind == "escalation" (and §4 found neither completion nor a clean
failure — the process may still be alive):
  - Treat this like "failed" for diagnosis, but the process may still be alive.
  - Do NOT kill it.
  - failure.cause = escalation_reason mapped to one of the signatures above
    when the evidence supports it, else "escalation:<short reason>".
  - failure.evidence = last 20 stdout lines + the last_health_report alerts.

### 5. Recover missing evaluation artifacts (eval-only re-run)

Required eval artifacts:
  - {local_run_dir}/results/eval_metrics.csv
  - {local_run_dir}/results/metrics.json
  - {local_run_dir}/reports/evaluation_report.md
  - {local_run_dir}/reports/plot_manifest.json
  - {local_run_dir}/results/plots/ (must be non-empty)

If ANY of these are missing or empty AND {local_run_dir}/checkpoints/best/ is
present AND has content (config.json + model.safetensors* or pytorch_model.*),
issue ONE eval re-run on the remote against the best checkpoint:

  mcp__lambda-repl__lambda_send_to_remote_tmux(
    ssh_alias="{ssh_alias}",
    command=(
      "cd {remote_run_dir} && "
      "export HF_TOKEN=\"$(cat ~/.cache/huggingface/token 2>/dev/null)\" && "
      "export HUGGING_FACE_HUB_TOKEN=\"$HF_TOKEN\" && "
      "python train.py --eval-only --checkpoint checkpoints/best/ "
      "--out results/ > outputs/eval_rerun.log 2>&1"
    ),
    key_path="{key_path}",
  )

Poll lambda_get_output until the command returns (expected <= 15 min). Then
re-pull results/ and reports/ from the remote. After re-pull, re-check the
required artifacts.

If the re-run fails OR checkpoints/best/ is itself missing, record
failure.cause = "eval_regeneration_failed" in final_summary but DO NOT abort —
proceed with the rest of finalization.

### 6. Regenerate plots locally if CSVs exist but plots do not

If {local_run_dir}/results/eval_metrics.csv exists AND any of the plots listed
in {local_run_dir}/reports/plot_manifest.json is missing under
{local_run_dir}/results/plots/, run the plot script locally:

  python src/eval/plot_eval.py --csv results/eval_metrics.csv \
                               --out results/plots/

Verify every plot named in plot_manifest.json.plots[] now exists under
results/plots/. Record any still-missing plot in final_summary.failure with
cause = "plot_regeneration_failed".

### 7. Verify checkpoints and push BEST to HF Hub (PRIVATE, sharded)

This step is mandatory when terminal_state == "completed". For failed runs,
skip with reason "skipped:terminal_state=failed" and continue to step 8.

ONLY the `best` checkpoint is pushed to the Hub. `last` is kept locally for
debugging and resumption but is never pushed.

a) Verify both directories exist locally and look like valid HF model dirs:
     {local_run_dir}/checkpoints/best/   (REQUIRED — pushed)
     {local_run_dir}/checkpoints/last/   (REQUIRED locally; NOT pushed)
   Each must contain config.json and either:
     - model.safetensors  (single shard), OR
     - model.safetensors.index.json + model-00001-of-N.safetensors* (sharded)
   If `checkpoints/best/` is missing, record a soft failure in final_summary
   under failure.cause = "checkpoint_incomplete" and skip the push. If
   `checkpoints/last/` is missing, record failure.cause = "last_checkpoint_missing"
   but still proceed with the push of `best`.

b) Push ONLY `best` to the Hub:
     Run this Python via Bash (HF_TOKEN is already at ~/.cache/huggingface/token,
     no extra env wrangling needed):

       python - <<'PY'
       from huggingface_hub import HfApi
       from transformers import AutoModel, AutoConfig
       import os, json as _json

       run_id  = "{run_id}"
       which   = "best"
       local   = "{local_run_dir}/checkpoints/" + which
       cfg     = {hub_push_config}            # already a dict

       api = HfApi()
       ns = cfg.get("namespace") or api.whoami()["name"]
       repo_id = ns + "/" + cfg["repo_name_template"].format(run_id=run_id, which=which)
       api.create_repo(repo_id, private=cfg["private"], exist_ok=True, repo_type="model")

       # Resave with sharding so anything >2GB becomes safetensors index +
       # numbered shards. This both deduplicates state-dict structure and gives
       # us the safetensors format the Hub prefers.
       model = AutoModel.from_pretrained(local)
       model.save_pretrained(local, max_shard_size=cfg["shard_size"], safe_serialization=True)

       api.upload_folder(folder_path=local, repo_id=repo_id, repo_type="model")
       print("PUSHED", which, repo_id)
       PY

     Capture stdout; the line `PUSHED best <repo_id>` is the success
     marker. Record the repo_id (or null on failure) in:
       final_summary.hub_best_repo_id

     If the push raises, record the exception message under
     final_summary.failure.cause = "hub_push_failed:best" and continue.

### 8. Compute actual_cost_usd (best-effort)

If pricing/cost_report.json contains hourly_rate_usd AND launched_at_iso was
captured in handoff.json, compute:
  wall_hours = (now - launched_at_iso) / 3600
  actual_cost_usd = wall_hours * hourly_rate_usd
Otherwise leave it null.

### 9. Write final_summary.json

Write {local_run_dir}/reports/final_summary.json.

Strict schema (no extra keys):

{{
  "run_id": "{run_id}",
  "handoff_kind": "...",
  "terminal_state": "...",      // EXACTLY one of: completed | failed | unknown (from §4 evidence, NOT a copy of handoff_kind)
  "final_metrics": {{ ... }},   // {{}} if none
  "final_model_path": "..." | null,
  "checkpoint_paths": ["...", ...],
  "trackio_url": "..." | null,
  "actual_cost_usd": number | null,
  "hub_best_repo_id": "..." | null,    // from §7 (only `best` is pushed)
  "repaired_files": ["...", ...],      // from §2 structure_validation.json
  "repaired_dirs":  ["...", ...],      // from §2 structure_validation.json
  "failure": {{
    "cause": "...",             // present only when something failed
    "evidence": "..."
  }} | null,
  "completed_at": "<ISO8601 UTC>"
}}

Map terminal_state — from the §4 EVIDENCE resolution, not a copy of handoff_kind:
  completion evidence present            -> terminal_state = "completed"
  failure evidence present, no completion -> terminal_state = "failed"
  remote unverifiable, no local evidence -> terminal_state = "unknown"
  (handoff_kind only breaks ties when the evidence is genuinely silent, and even
   then a bare handoff_kind == "failed" with an unreachable remote is
   "unknown", never "failed".)

### 10. Write {local_run_dir}/RUN_REPORT.md (scientific summary)

This is the run's contribution to CAPO's episodic memory. Future runs will
read it as a prior when their task fingerprint matches.

The file has two parts: a YAML frontmatter and a markdown body. Every value
must come from an artifact you read. If a source is missing, write `null`
(frontmatter) or `unknown` (body). DO NOT FABRICATE.

#### YAML frontmatter — EXACTLY these 13 fields, no others

```
---
run_id: {run_id}
task_summary: <one sentence paraphrase of {local_run_dir}/task.md>
modality: <protein|antibody|peptide|DNA|RNA|tabular|single_cell|flow_cytometry|reads> (from reports/research_findings.json entity_frame.modality, else parse task.md)
target: <gene/protein/pathway from research_findings.json entity_frame.target, or "n/a">
organism: <human|mouse|SARS-CoV-2|multi|... from entity_frame.organism, or "n/a">
assay: <binding|affinity|fold|expression|viability|... from entity_frame.assay, or "n/a">
best_metric_name: <e.g. val_mcc — from results/metrics.json or last row of outputs/metrics.jsonl; null if terminal_state != completed>
best_metric_value: <float, null if terminal_state != completed>
final_val_loss: <float, null if not present>
key_decisions:
  - "<≤3 bullets. Each ONE short sentence citing the artifact it came from, e.g. 'Linear-probe chosen — model_selection.json scored it above LoRA for this dataset size.'>"
key_findings:
  - "<≤3 bullets. Each cites the source — e.g. 'Cluster-aware split improved MCC by 0.12 vs random (results/metrics.json comparison rows).'>"
key_pitfalls:
  - "<≤3 bullets. Each cites the source — e.g. 'OOM at batch 64 with p99 length 1024 (probe_result.json forward_backward). Dropped to 32 + grad_accum=2.'>"
report_path: capo/{run_id}/RUN_REPORT.md
---
```

Field-by-field source map (read these to fill the frontmatter):
  - run_id              : passed in this prompt
  - task_summary        : paraphrase {local_run_dir}/task.md (1 sentence; no IPs, no SSH keys)
  - modality/target/organism/assay : {local_run_dir}/reports/research_findings.json -> entity_frame; if missing, parse task.md, else "n/a"
  - best_metric_name/value : {local_run_dir}/results/metrics.json top-line OR last row of {local_run_dir}/outputs/metrics.jsonl. null on failure.
  - final_val_loss      : same source. null if absent.
  - key_decisions       : synthesize ≤3 from {local_run_dir}/reports/model_selection.json, {local_run_dir}/pricing/cost_report.json, {local_run_dir}/probe/probe_result.json, {local_run_dir}/profile/probe_batch_recipe.json
  - key_findings        : synthesize ≤3 from {local_run_dir}/results/metrics.json, {local_run_dir}/reports/evaluation_report.md (if present), {local_run_dir}/reports/health/history.jsonl
  - key_pitfalls        : synthesize ≤3 from {local_run_dir}/outputs/train.log tail, {local_run_dir}/reports/health/history.jsonl alerts, structure_validation.json, abort markers in reports/
  - report_path         : literal "capo/{run_id}/RUN_REPORT.md"

#### Body — fixed section order, ≤500 words total

```
# Run report: {run_id}

## Task
<1-3 sentences paraphrasing task.md>

## Prior runs consulted
<List the run_ids cited in {local_run_dir}/reports/prior_runs.md, or "none".>

## Decisions
- Model: <model_id from model_selection.json> | Strategy: <fine_tune_strategy> | GPU: <resolved_gpu from infra.json>
- Rationale: <≤2 sentences citing model_selection.json and infra.json>

## Profiling
- n_samples=<...>, p50/p99 length=<...>, split_info.source=<...> (from profile/profile.json)
- <≤1 sentence on top concerns from profile/profile.json warnings>

## Feasibility probe
- batch_size=<...>, max_seq_length=<...>, peak_memory_gb=<...>, backward_latency_s=<...> (from probe/probe_result.json)
- Retries: <count, from probe_failure_packet.json if present>

## Cost gate
- projected_cost_usd=<...> vs max_cost_usd=<...> (from cost_report.json); decision=<approved|aborted_over_budget>

## Training
- epochs=<...>, optimizer=<...>, lr=<...>, effective_batch_size=<...> (from configs/training.yaml)
- checkpoint cadence: <from cost_report.json or probe_batch_recipe.json>

## Health monitoring
- <N> health polls; severity timeline: <e.g. "info → info → warn → severe"> (from reports/health/history.jsonl)
- Most severe alert: <...>

## Results
- Render a clean, paper-style markdown table of the final metrics from
  results/metrics.json — primary metric first. Use this exact shape:

  | Metric | Value |
  |---|---:|
  | <primary> | <0.0000> |
  | <next> | <0.0000> |

  Scientific formatting rules (apply consistently):
  - accuracy, mcc, f1, precision, recall, auroc, loss: 4 decimals (e.g. 0.8765)
  - cost: $X.XX  ·  runtime: human-readable (e.g. "1h 23m")  ·  missing value: —
  - Never dump nested JSON or raw dicts into a table cell.
- If failed: failure.cause=<...>, last 20 stdout lines from final_summary.json.failure.evidence

## Findings & pitfalls
- <3 bullets total. Each cites the artifact that supports it.>

## Recovery (failed runs only — OMIT if state=completed)
- Include this section ONLY when reports/post_launch_failure.json was found
  with `recoverable=true`.
- Quote `post_launch_failure.remediation` verbatim as the first bullet.
- Add a second bullet with the precise next action — e.g.
  "Re-run with --sequence-column <name>", "Set dataset_ref to <pinned revision>",
  "Halve probe_batch_size to <N>". Cite reports/post_launch_failure.json.
- Add a third bullet only if a checkpoint exists under checkpoints/last/ —
  "Resume from checkpoints/last/ via `resume: {run_id}` in fine_tuning.yaml".

## Artifacts
- HF Hub (private, sharded, BEST only): https://huggingface.co/<hub_best_repo_id>  (from final_summary.json; omit if null. `checkpoints/last/` is retained locally and intentionally NOT pushed.)
- reports/final_summary.json | reports/model_selection.json | infra.json | profile/profile.json | pricing/cost_report.json | probe/probe_result.json | results/metrics.json | results/plots/ | reports/evaluation_report.md | reports/health/history.jsonl | reports/post_launch_failure.json (failed runs only)
```

> Do NOT write a "## Cost Report" or "## Training Recovery" section yourself. The
> orchestrator appends both to RUN_REPORT.md after you finish: a complete agent +
> infrastructure Cost Report (authoritative token/runtime figures you do not
> have), and — when an agentic post-canary recovery ran — a Training Recovery
> table (from reports/recovery_ledger.json). The "## Cost gate" (projected cost)
> and "## Recovery" (single-failure remediation hint) sections above are separate
> and you should still write them as instructed.

Write the file with the Write tool. After writing, Read it back to verify the
frontmatter parses cleanly (the next step depends on it).

### 11. Append the YAML frontmatter to the cross-run index

Run exactly this Bash command:

  python -m capo.memory.run_report append-from-report --run-dir {local_run_dir}

This CLI parses RUN_REPORT.md's frontmatter, validates the 13-field schema,
takes an exclusive file lock on <repo>/runs/.runs_index.lock, deduplicates by
run_id (if a previous block exists for this run_id, it is replaced), and
appends a new YAML block to <repo>/runs/runs_index.md. You DO NOT touch the
index file or the lock directly.

If the command exits non-zero, retry it ONCE. If it still fails, write
{local_run_dir}/reports/index_append_failure.json with
{{"error": "<stderr verbatim>", "command": "<command verbatim>"}} and
continue — the per-run RUN_REPORT.md is the authoritative copy and can be
re-indexed later.

### 11b. Cost backstop — terminate the instance (opt-in, failed runs only)

ONLY do this when ALL of the following hold:
  - auto_terminate_on_failure == "True", AND
  - handoff_kind is "failed" or "escalation" (NEVER on "completed"), AND
  - every artifact pull above has finished and final_summary.json + RUN_REPORT.md
    are written (terminating destroys the remote — sync FIRST, always).

Then:
  1. Read instance_id from {local_run_dir}/infra.json.
  2. Call mcp__lambda-repl__lambda_terminate_safe(
       instance_id=<infra.json instance_id>,
       expected_ssh_key_names=["{ssh_key_name}"])
     (ownership-checked: it refuses if the key names don't match — this is a
     safety guard, do not work around it.)
  3. On success, emit:
       ip.emit("[fine-tuning] auto-terminated instance <id> after failure (cost backstop)")
     and add "instance_terminated": true to final_summary.json.
     On failure (ownership mismatch / API error), add
     "instance_terminated": false and a one-line reason; do NOT retry
     aggressively — leave the instance for manual teardown.

If auto_terminate_on_failure != "True", or the run completed, leave the instance
running (warm instances enable cheap re-runs from cached stages) and skip this
step entirely.

### 12. Final answer

Return ONLY the final_summary.json contents as your final answer. No prose, no
markdown fences. A parser will read your answer directly.
