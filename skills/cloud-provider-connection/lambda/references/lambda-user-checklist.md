---
name: lambda-user-checklist
description: Three Lambda checklists for the human user — pre-launch, pre-workload, pre-termination.
---

# Lambda User Checklist

Confirm every item before each phase. Items that fail must be resolved
before proceeding.

## Pre-launch

- [ ] `LAMBDA_API_KEY` is exported in this shell (`echo $LAMBDA_API_KEY`).
- [ ] A private SSH key exists locally (commonly `~/.ssh/lambda_*` or
      `~/.ssh/id_ed25519`) with permissions `0600`.
- [ ] The matching key name is registered in the Lambda Cloud console
      (Account → SSH Keys).
- [ ] Instance type chosen — confirmed against `lambda_list_instance_types`
      with **live capacity** in at least one region.
- [ ] Region selected (or accept `auto` and let the runtime pick).
- [ ] Budget chosen — `budget_limit_dollars` and optional
      `budget_warning_threshold_dollars` for the cost estimate calls.
- [ ] Persistent storage decision made — Lambda filesystems mount at
      `/lambda/nfs/<NAME>` and survive termination; instance-local storage
      does not.

## Pre-workload

- [ ] Instance status is `active` and an IP has been assigned
      (`lambda_list_instances`).
- [ ] SSH alias works — `ssh lambda-<name>` returns a shell.
- [ ] GPU verified — `lambda_run_command(session_id, "nvidia-smi")` returns
      the expected device(s).
- [ ] Code synced via `lambda_push_files` (or via the background rsync
      watch loop the session opened).
- [ ] Dataset path verified on the remote — confirm before launching the
      workload, not after.
- [ ] A venv (or container) is active — never replace system Python.

## Pre-termination

- [ ] Outputs pulled (`lambda_pull_files` for `outputs/`).
- [ ] Evaluation results and plots pulled (`lambda_pull_files` for `results/`)
- [ ] Checkpoints pulled (`lambda_pull_files` for `checkpoints/`).
- [ ] Logs pulled (`lambda_pull_files` for `logs/`).
- [ ] Final cost estimate recorded — call `lambda_get_cost_estimate` and
      log the result.
- [ ] Ownership check — `lambda_terminate_safe(instance_id,
      expected_ssh_key_names=[your_key_name])` will refuse to terminate
      if the instance's registered keys do not include yours. If it
      refuses, **stop**: the instance belongs to someone else.
