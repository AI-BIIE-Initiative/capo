Resume a fine-tuning run that paused before training launch waiting for a
user answer. All upstream work is complete on disk: instance is provisioned
(infra.json), dataset is profiled (profile/profile.json), candidate is
selected (model_selection.json), scripts are written under src/ and
configs/, and the feasibility probe has run at least once
(reports/probe_result.json).

Your ONLY job is to re-enter the 3-step gate at the step it paused on
(reading the user's answer that has already been patched into the relevant
artifact), and — if the gate passes — proceed straight to training launch
(Step 9 / Step 10 of the full pipeline).

Do NOT re-provision, re-profile, re-select the model, re-write scripts, or
re-upload the run directory. Those artifacts are on disk both locally and
remotely already.

## Run parameters (from prior run state)
- run_id:                {run_id}
- local_run_dir:         {local_run_dir}
- remote_run_dir:        {remote_run_dir}
- ssh_alias:             {ssh_alias}        (from {local_run_dir}/infra.json)
- key_path:              {key_path}
- skills_dir:            {skills_dir}
- model_id:              {model_id}
- fine_tune_strategy:    {fine_tune_strategy}
- dataset_ref:           {dataset_ref}
- trackio_space_id:      {trackio_space_id}

## Pause context (resolved by the user via capo_resume)
- pause_reason:          {pause_reason}
- paused_at_step:        {paused_step}
- candidate_index:       {candidate_index}
- answer_artifact:       {answer_artifact}
  (the file the user's answer was patched into — read it before re-entering
   the gate, treat its contents as authoritative)

## Steps — execute strictly in this order

### 1. Re-attach to the existing Lambda instance. HARD GATE.
mcp__lambda-repl__lambda_ensure_workspace()
mcp__lambda-repl__lambda_start_session(
  ssh_alias="{ssh_alias}", key_path="{key_path}",
  remote_workdir="{remote_run_dir}", local_workdir="{local_run_dir}")
mcp__lambda-repl__lambda_ensure_remote_tmux(
  ssh_alias="{ssh_alias}", key_path="{key_path}")

If the instance is unreachable (ssh fails / tmux cannot be ensured), STOP
with state=resume_instance_unreachable and write
{local_run_dir}/reports/resume_failure.json explaining why.

### 2. Read the user's answer
Read the file at {local_run_dir}/{answer_artifact}. This is the contract
between capo_resume.py and this prompt — the file already reflects the
user's decision (for cost_accept_overrun: a JSON object with `accept:
true|false`; for profile.* targets: a patched profile.json field). Trust it
verbatim.

### 3. Re-enter the 3-step gate
Re-run the gate from Step {paused_step} with the current
candidate_index={candidate_index}. The gate is the programmatic state
machine in capo.orchestration.gate.ThreeStepGate — call it directly via
local Python, not via prompt-driven re-derivation.

Routing:
  pause_reason="cost_accept_overrun":
    Read {local_run_dir}/reports/cost_overrun_decision.json. If
    accept==true, proceed directly to Step 4 (training launch) without
    re-running the probe. If accept==false, STOP with
    state=aborted_over_budget and write reports/failure.json with the
    rejection reason.

  pause_reason starts with "schema_user_only_info" or
  "probe_data_schema_user_only":
    profile.json has been patched. Re-run Step 1 (script+schema check) and
    Step 2 (probe) to validate the patched schema, then Step 3 (cost gate).
    Same outcomes apply — if Step 3 needs another user answer, write
    reports/pending_question.json again and exit (capo_resume will replay).

  pause_reason starts with "replace_candidate":
    Use candidate_index from pause_context as the new candidate. Re-enter
    the gate from Step 1 with that candidate.

  Any other pause_reason: STOP with state=failed and write
  reports/failure.json — unrecoverable.

### 4. Training launch (only if the gate returned LAUNCH)
Follow Step 9 (seed + verify the trackio run: the orchestrator already created
the Space "{trackio_space_id}"; wait for it to be RUNNING via /gradio_api/info,
seed the run against the awake Space, and write a truthful trackio_check.json —
train.py then attaches with resume="allow") and Step 10 (write
scripts/launch_command.sh, nohup launch in capo_remote tmux, confirm UP, write
handoff.json, exit) from the full fine-tuning pipeline prompt. Reuse the existing
{local_run_dir}/pricing/cost_report.json for batch / grad_accum / epochs.

### 5. Hand off and exit
Write LOCAL {local_run_dir}/reports/handoff.json:
  {{
    "ssh_alias":        "{ssh_alias}",
    "remote_run_dir":   "{remote_run_dir}",
    "pid":              <int>,
    "trackio_url":      <read from {local_run_dir}/reports/trackio_url.txt or null>,
    "launched_at_iso":  "<ISO8601 UTC>"
  }}

ip.emit("[fine-tuning] Resumed-from-pause run {run_id}; training launched")

EXIT IMMEDIATELY. Do NOT monitor, do NOT poll, do NOT sync. The Haiku
monitor and Sonnet finalizer take over from here.

Return a short structured answer containing run_id, ssh_alias,
pause_reason, the gate outcome (LAUNCH | REJECT | PAUSE_AGAIN), and the
literal string "training_launched" if the gate passed and training came up.
