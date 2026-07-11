Resume a fine-tuning run that was interrupted. The prior run's infrastructure,
profiling, probe, and cost gate all completed; your job is to re-attach to the
existing Lambda instance, verify state, resume from wherever the run ACTUALLY
stopped, monitor, and sync.

There are TWO resume modes — pick the right one from the on-disk state:

  * CHECKPOINT resume (interrupted mid-training): a training checkpoint exists
    under checkpoints/last/. Re-launch training from the latest checkpoint.
  * ARTIFACT resume (an expensive PRE-training stage finished but training never
    started — e.g. Boltz embedding generation): NO checkpoint exists, yet valid
    artifacts sit on disk. Re-launch the idempotent pipeline, which REUSES
    completed artifacts and computes only what is missing. NEVER recompute an
    artifact that already exists and is valid.

A read-only plan is in {local_run_dir}/run_state.json (next_resume_point, the
reusable count, and the missing set). Read it first to choose the mode.

Do NOT re-run profiling, the feasibility probe, or the cost gate. Do NOT
provision a new instance (unless the recorded one is gone). Do NOT re-initialize
trackio (it already has a run id; resuming appends metrics to the existing run).

## Run parameters (from prior run state on disk)
- run_id:           {run_id}
- local_run_dir:    {local_run_dir}
- remote_run_dir:   {remote_run_dir}
- ssh_alias:        {ssh_alias}     (from {local_run_dir}/infra.json)
- key_path:         {key_path}
- skills_dir:       {skills_dir}
- trackio_space_id: {trackio_space_id}

## Steps — execute strictly in this order

### 1. Attach to the existing Lambda instance. HARD GATE.
mcp__lambda-repl__lambda_ensure_workspace()
mcp__lambda-repl__lambda_start_session(
  ssh_alias="{ssh_alias}", key_path="{key_path}",
  remote_workdir="{remote_run_dir}", local_workdir="{local_run_dir}")
mcp__lambda-repl__lambda_ensure_remote_tmux(
  ssh_alias="{ssh_alias}", key_path="{key_path}")
mcp__lambda-repl__lambda_list_sessions(...)

If the instance is unreachable (ssh fails / tmux cannot be ensured), STOP
with state=resume_instance_unreachable and write
{local_run_dir}/reports/resume_failure.json explaining why.

### 2. Verify the previous training process is dead
Issue ONE compound SSH snapshot and parse its output:
  ssh -i {key_path} -o StrictHostKeyChecking=no <host> "
    RUN={remote_run_dir}
    pid=\$(cat \$RUN/outputs/train.pid 2>/dev/null || echo '')
    echo '=PID=' \$pid
    echo '=ALIVE=' \$(ps -p \$pid > /dev/null 2>&1 && echo 1 || echo 0)
    echo '=STATUS_JSON='
    cat \$RUN/outputs/status.json 2>/dev/null
    echo '=CKPT_LIST='
    ls -1t \$RUN/checkpoints/last/ 2>/dev/null | head -10
    echo '=LAUNCH_CMD_EXISTS='
    test -f \$RUN/scripts/launch_command.sh && echo 1 || echo 0
  "
Take THREE snapshots separated by >=30s before concluding DEAD. The
PROCESS LIVENESS RULE from the system prompt applies absolutely — a
process is DEAD only after three consecutive DEAD snapshots.

If ALIVE on any snapshot: the previous training is still running. Do NOT
restart. Skip steps 3–5 and go directly to step 6 (monitor) — this is a
pure re-attach, not a checkpoint restart.

### 3. Choose the resume mode (checkpoint vs artifact)
Parse =CKPT_LIST= from the snapshot in step 2.

If a checkpoint EXISTS → CHECKPOINT resume. Pick the most recent checkpoint (by
mtime from `ls -1t`, or by numerical step-suffix if the directory uses
checkpoint-<step>/ naming — prefer the higher step when consistent). Record it as
RESUME_CKPT (absolute path on remote) and continue to Step 4.

If the checkpoints directory is EMPTY or missing, do NOT immediately give up.
Read {local_run_dir}/run_state.json: when next_resume_point is an artifact stage
(resume_embeddings / start_training_from_embeddings / skip_embeddings) and valid
artifacts are present (done > 0) → ARTIFACT resume. Set RESUME_CKPT to none and
continue to Step 4. The expensive completed artifacts (e.g. Boltz
embeddings_*.npz) will be reused by the idempotent pipeline, not recomputed.

ONLY if there is NO checkpoint AND no reusable artifacts (run_state.json shows
done == 0 and has_checkpoint false): STOP with state=resume_no_checkpoint and
write {local_run_dir}/reports/resume_failure.json — there is no saved state to
resume from; a full re-run is required.

### 4. Reconstruct the launch command
If =LAUNCH_CMD_EXISTS= was 1, read the original command via:
  ssh -i {key_path} <host> "cat {remote_run_dir}/scripts/launch_command.sh"
Otherwise, reconstruct it by reading {local_run_dir}/pricing/cost_report.json
(batch size, grad accum, epochs, eval_every, checkpoint_every) plus the
model_id and dataset from {local_run_dir}/infra.json / the original task,
and assemble a command matching the structure of Step 10 from the full
fine-tuning pipeline prompt.

CHECKPOINT resume (RESUME_CKPT is a path): append
`--resume-from-checkpoint <RESUME_CKPT>` to the reconstructed command.

ARTIFACT resume (RESUME_CKPT is none): use the launch command UNCHANGED — do
NOT append --resume-from-checkpoint. The generated pipeline is idempotent: it
recomputes only the complexes whose valid embedding is missing, and must NOT
pass `--override` (which would discard completed embeddings). Inspect the
command and REMOVE any `--override` on the main boltz predict call before
re-launching.

In both modes, write the resumed command to scripts/launch_command.sh on the
remote so future resumes see the latest form. Archive the previous
launch_command.sh as launch_command.sh.bak on the remote first.

### 4b. Reinstall dependencies + verify the environment on THIS instance (HARD GATE)

A resume may land on a DIFFERENT instance than the original run — the recorded one
was terminated and re-provisioned, or its environment was never fully built. NEVER
relaunch without first proving every required module imports on the instance you
are about to spend GPU money on. This is the single most common resume failure: a
fresh instance missing pyarrow/datasets/the GPU kernel crashes Stage 1 minutes in,
for dollars, when the same gap is caught here in seconds. The main pipeline runs
this gate at its Step 4; resume MUST run it too.

1. Reinstall (idempotent — pip skips already-satisfied pins):
   mcp__lambda-repl__lambda_send_to_remote_tmux(
     ssh_alias="{ssh_alias}",
     command="cd {remote_run_dir} && pip install -q -r requirements.txt trackio",
     key_path="{key_path}")
   Poll lambda_get_output until pip returns.

2. Verify every critical module imports on THIS instance — torch, the data stack
   (datasets, pyarrow, fastparquet, rdkit) and every model/GPU-kernel module named
   in the model's SKILL.md import section. Run the SAME import-gate heredoc as
   Step 4 of the full fine-tuning pipeline prompt: it writes reports/env_check.json
   and prints `ENV_CHECK_OK` or `ENV_CHECK_FAIL=<modules>`.

3. On `ENV_CHECK_FAIL` (modules; cuda True): attempt ONE repair — reinstall
   requirements plus the named failing packages (GPU-kernel imports → the full
   package set in that model's SKILL.md, with
   `--extra-index-url https://pypi.nvidia.com`), then re-run the gate once. Only
   when it prints `ENV_CHECK_OK` may you proceed to Step 5 and relaunch. If it
   still fails (or cuda is not True), STOP with state=resume_env_check_failed and
   write {local_run_dir}/reports/resume_failure.json naming the still-failing
   modules — do NOT relaunch into a guaranteed crash.

### 5. Re-launch training, confirm UP, hand off, exit

Stale-process safety: even though the liveness rule says the pid is DEAD,
issue one more sanity kill before re-launching to guarantee no stale
Python process is holding GPU memory:
  ssh <host> "pkill -f 'python.*src/train\\.py' || true"
  sleep 5

GPU-kernel verify (Boltz / any cuequivariance-dependent run): if the prior
failure was a CUDA/cuequivariance kernel error (check
{local_run_dir}/reports/post_launch_failure.json for failure_category ==
"cuda_kernel"), the env MUST be repaired before re-launch or it re-crashes
identically. Run the runtime-path import gate on the remote:
  ssh <host> "python - <<'PY'
import cuequivariance_torch, cuequivariance_ops_torch
from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
print('KERNELS_OK', cuequivariance_ops_torch.__version__)
PY"
If it does not print KERNELS_OK, install the full six-package set from the NVIDIA
index (see skills/model-inference/boltz/SKILL.md and references/boltz2-cuda.lock),
then re-verify. If the kernels still will not load, add `--no_kernels` to the
boltz predict call in launch_command.sh (slower pure-PyTorch path, guaranteed
progress) before re-launching.

Then re-launch via tmux:
  mcp__lambda-repl__lambda_send_to_remote_tmux(
    ssh_alias="{ssh_alias}",
    command=(
      "cd {remote_run_dir} && "
      "nohup bash scripts/launch_command.sh "
      ">> outputs/train.log 2>&1 & echo \$! > outputs/train.pid"
    ),
    key_path="{key_path}")

Note the `>>` append — the original outputs/train.log is preserved; new output
is concatenated. outputs/metrics.jsonl and outputs/status.json continue
appending per train.py's resumption contract.

#### 5a. Confirm training is UP (bounded, max ~2 min)

Issue up to THREE SSH snapshots, 40s apart, reading pid liveness +
outputs/metrics.jsonl row count + outputs/status.json (same pattern as Step 10a
of the full pipeline). Training is UP when pid is alive AND
(outputs/metrics.jsonl has ≥1 row OR outputs/status.json shows state in
{{running, training}}).

If all three snapshots show the process DEAD: sync outputs/train.log locally,
write {local_run_dir}/reports/failure.json with a brief explanation, and
exit without writing handoff.json.

#### 5b. Write handoff.json and EXIT

Once UP, write LOCAL {local_run_dir}/reports/handoff.json:
  {{
    "ssh_alias":        "{ssh_alias}",
    "remote_run_dir":   "{remote_run_dir}",
    "pid":              <int>,
    "trackio_url":      <read from {local_run_dir}/reports/trackio_url.txt or null>,
    "launched_at_iso":  "<ISO8601 UTC>",
    "resumed_from_checkpoint": "<RESUME_CKPT>"
  }}

ip.emit("[fine-tuning] Resumed run {run_id} from checkpoint <RESUME_CKPT>; handing off")

EXIT IMMEDIATELY. Do NOT monitor, do NOT poll, do NOT sync. The Haiku
monitor and Sonnet finalizer take over from here.

Return a short structured answer containing run_id, ssh_alias,
instance_type (from infra.json), the RESUME_CKPT path, and the literal
string "training_launched". Include the exact phrase
"Resumed run {run_id} from checkpoint <RESUME_CKPT>" somewhere in the
answer — the orchestrator extracts the resume path from it.
