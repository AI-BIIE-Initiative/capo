---
name: lambda-workflow
description: Conceptual flow for the lambda-repl MCP tools — what to call when and why.
---

# Lambda Workflow

The agent should call tools in roughly this order. Each stage has a *why*;
deviate when the situation calls for it, but do not skip the gates.

## 1. Discovery — `lambda_find_local_ssh_keys` → `lambda_list_ssh_keys`

Match a local private key to a Lambda-account-registered key. Show the user
**file paths only** — never key contents. If multiple candidates exist, ask
which to use; recommend by filename hint (`lambda_*`, `id_ed25519`).

## 2. Catalog — `lambda_list_instance_types`

Pick a GPU type with **live regional capacity** and a price the user can
afford. Default `available_only=True`. Save the chosen type's
`price_dollars_per_hour` for the cost-tracking phase.

## 3. Preflight — `lambda_preflight`

Hard gate. Run with the chosen `key_path`. If any check fails (missing
`LAMBDA_API_KEY`, ssh/rsync/tmux not on PATH, key permissions not 0600,
unrecognised key header), surface the failure to the user and stop.

## 4. Provision + baseline — `lambda_provision_instance` → `lambda_get_first_cost_estimate`

Launch the instance, then immediately compute the t0 cost estimate with the
chosen budget thresholds. The baseline (`elapsed_hours ≈ 0`,
`estimated_cost ≈ 0`, `price_dollars_per_hour` populated) becomes the audit
record for "what we promised the user before any compute ran."

Wait for `status="active"` and an IP via `lambda_list_instances` polling
(the runtime calls `wait_for_instance_ip` and `wait_for_ssh_ready` for you
when `lambda_run_workflow` is used end-to-end).

## 5. Session — `lambda_start_session`

Opens the two-window tmux layout (`remote` SSH + `sync` rsync watch) for
the agent's interactive use. The `remote` window owns SSH commands; the
`sync` window owns rsync. Cross-window commands are an error.

## 6. Workload loop — `lambda_push_files` → `lambda_run_command` → `lambda_pull_files`

- Push code/data, run the workload, pull results.
- Use `lambda_get_output` to monitor long jobs without blocking.
- For very large transfers, use `lambda_pull_files` in chunks rather than
  catting through stdout.

## 7. Cost recheck — `lambda_get_cost_estimate`

After meaningful elapsed time (and before any decision to keep running),
recheck the cost against the user's budget. The tool returns the same
`LambdaCostEstimate` shape as the t0 baseline — diff it to see actual spend.

## 8. Safe termination — `lambda_terminate_safe`

Pass `expected_ssh_key_names=[<the user's key name>]`. The tool refuses to
terminate if the live instance's `ssh_key_names` does not include every
expected name — termination is irreversible and a key mismatch indicates the
instance belongs to a different user.

`sudo shutdown`, `poweroff`, and `halt` do **not** stop billing on Lambda.
The only correct termination path is `lambda_terminate_safe` (or the Lambda
Cloud console).
