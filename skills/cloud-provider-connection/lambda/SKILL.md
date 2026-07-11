---
name: lambda-cloud-connection
description: >
  Manage Lambda On-Demand GPU instances end-to-end via the lambda-repl MCP
  tools: SSH key discovery, preflight, instance provisioning, session
  management, command execution, file transfer, cost tracking, and safe
  termination. Documentation only — all functional behaviour lives in
  mcp__lambda-repl__* tools (implemented in src/capo/mcp/tools/lambda_tools.py
  and src/capo/remote/).
user-invokable: true
compatibility: >
  Requires LAMBDA_API_KEY env var, ssh + rsync + tmux on PATH, and an SSH key
  registered in the Lambda Cloud console. Targets Lambda On-Demand instances
  only (not Kubernetes / 1-Click Clusters / Slurm).
---

# Lambda Cloud Connection

This skill is reference documentation. All actions are MCP tools.

## Tool inventory (recommended order)

1. `lambda_find_local_ssh_keys` — scan `~/.ssh/` for private-key candidates (paths only)
2. `lambda_list_ssh_keys` — list SSH keys registered on the Lambda account
3. `lambda_list_instance_types` — catalog with live capacity + pricing
4. `lambda_preflight` — gate: API key, ssh/rsync/tmux on PATH, key file validity
5. `lambda_provision_instance` — launch a new GPU instance
6. `lambda_get_first_cost_estimate` — t0 baseline immediately after provision
7. `lambda_start_session` — open SSH + rsync tmux session
8. `lambda_push_files` — sync local → remote
9. `lambda_run_command` — run a shell command on the remote
10. `lambda_pull_files` — pull artifacts remote → local
11. `lambda_get_cost_estimate` — recheck elapsed cost vs budget
12. `lambda_terminate_safe` — terminate after verifying `ssh_key_names` ownership

## See

- `references/lambda-workflow.md` — conceptual flow and decision points
- `references/lambda-user-checklist.md` — pre-launch / pre-workload / pre-termination

## Critical agent rules

- **Never display SSH key contents** — only file paths.
- **Never use `sudo shutdown` / `poweroff` / `halt`** — billing continues.
  Always use `lambda_terminate_safe`, which verifies `ssh_key_names`
  ownership before issuing the API call.
- **GH200 is ARM** — confirm stack compatibility before launching.
- **Lambda instances are ephemeral** — pull artifacts before terminating.
- **Default firewall is SSH-only** — use SSH tunnels for Jupyter / TensorBoard
  rather than opening ports.
- **Never replace system Python** — install an additional version alongside
  it (e.g. `python3.13-full`) and use a venv.
