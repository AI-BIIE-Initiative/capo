Run protein model inference on Lambda using the CAPO multi-session framework.

## Run parameters
- run_id:           {run_id}
- task_description: {task_description}
- ssh_key_name:     {ssh_key_name}
- key_path:         {key_path}
- local_run_dir:    {local_run_dir}
- skills_dir:       {skills_dir}
- instance_type:    {instance_type}   ("auto" = select cheapest available)
- instance_name:    {instance_name}   ("auto" = infer a short name from task_description, e.g. "esm2_embeddings")

## Steps — execute in order

### 1. Local tmux workspace
mcp__lambda-repl__lambda_ensure_workspace()

### 2. Identify the inference script
Read the task_description to determine the model family (esm2, boltz, chai, ankh, prottrans, etc.).
Find the matching script under {skills_dir}/model-inference/<family>/scripts/.
Use Read to inspect it briefly and confirm the CLI interface and required dependencies.

### 3. Get or provision Lambda instance
Use the lambda-repl MCP tools as documented in `skills/cloud-provider-connection/lambda/SKILL.md`.

instance_name = "{instance_name}"

If instance_name != "auto":
  - mcp__lambda-repl__lambda_list_instances()
  - If a running instance named instance_name exists → use it (record ip, skip launch)
  - Otherwise → mcp__lambda-repl__lambda_provision_instance(
        instance_type="{instance_type}", ssh_key_name="{ssh_key_name}", name=instance_name)
If instance_name == "auto":
  - mcp__lambda-repl__lambda_list_instances()
  - If instances exist, show them and ask the user which to connect to. Wait for reply.
  - If user names one → use it. If "new" or none running → infer a short snake_case name
    from task_description (e.g. "esm2_embeddings") and provision with that name.
Record the selected instance ip and id for subsequent steps.

### 4. Wait for SSH and set alias
  Poll mcp__lambda-repl__lambda_list_instances until status="active" and an IP is assigned.
  The MCP runtime calls ensure_ssh_alias() automatically before any session opens.

### 5. Ensure capo_remote
  mcp__lambda-repl__lambda_ensure_remote_tmux(ssh_alias="lambda-{run_id}", key_path="{key_path}")

### 6. Prepare remote run directory
  Bash: ssh -o StrictHostKeyChecking=no -i {key_path} lambda-{run_id} "mkdir -p ~/capo_runs/{run_id}/outputs ~/capo_runs/{run_id}/checkpoints"

### 7. Stage local run directory
Copy the inference script identified in step 2 into {local_run_dir}.
Write any required input files (sequences, FASTA, config YAML) into {local_run_dir} based on task_description.
Then upload:
  mcp__lambda-repl__lambda_upload_run(
    ssh_target="lambda-{run_id}",
    local_run_dir="{local_run_dir}",
    remote_run_dir="~/capo_runs/{run_id}",
    key_path="{key_path}"
  )

### 8. Install dependencies
Read the SKILL.md for the chosen model family to find the pip requirements.
  mcp__lambda-repl__lambda_send_to_remote_tmux(
    ssh_alias="lambda-{run_id}",
    command="pip install -q <deps>",
    key_path="{key_path}"
  )
Wait for completion (capture remote tmux output after 90s to verify pip finished).

### 9. Start inference job in capo_remote
Construct the CLI command from the inference script's interface (--sequences, --model, --output, etc.).
  mcp__lambda-repl__lambda_start_inference(
    ssh_alias="lambda-{run_id}",
    run_id="{run_id}",
    command="<full command pointing to ~/capo_runs/{run_id}/outputs/ for output>",
    key_path="{key_path}"
  )

### 10. Poll status until terminal
Every 15 seconds:
  mcp__lambda-repl__lambda_sync_run_status(
    ssh_target="lambda-{run_id}",
    remote_run_dir="~/capo_runs/{run_id}",
    local_run_dir="{local_run_dir}",
    key_path="{key_path}"
  )
  mcp__lambda-repl__lambda_read_run_status(
    ssh_alias="lambda-{run_id}",
    run_id="{run_id}",
    key_path="{key_path}"
  )
Stop when state is "completed", "failed", or "stopped".
On failure: read {local_run_dir}/outputs/run_err.log and report the error.

### 11. Download outputs
  Bash: rsync -avz --partial -e "ssh -i {key_path} -o StrictHostKeyChecking=no" \
    lambda-{run_id}:~/capo_runs/{run_id}/outputs/ {local_run_dir}/outputs/

### 12. Report
Inspect outputs (shape for .npy, file list for .pdb, etc.) and report:
  run_id, model used, final state, output files with shapes/sizes.

### 13. Calculate infrastructure cost
Read skills/cost-estimation/references/lambda-pricing.md to find the hourly rate for the
instance type you used (multiply per-GPU rate × number of GPUs for the full instance rate).
Measure elapsed time from when the instance first became SSH-ready to when rsync of outputs
completed.  Report:  instance_type, per-GPU rate, total instance rate ($/h),
elapsed time (HH:MM:SS), and total_cost = elapsed_hours × total_rate.
