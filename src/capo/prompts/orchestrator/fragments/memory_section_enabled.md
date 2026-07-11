## Prior-run memory (episodic)
Before any pre-launch subagent ran, a memory-consultant scanned the cross-run
index at `<repo>/runs/runs_index.md` and selected 0–3 prior RUN_REPORT.md
bodies most relevant to the current task. The result is at
`<local_run_dir>/reports/prior_runs.md` and you MUST read it in Step 0. Use
its contents as ADVISORY priors, never as authority:
  - If a prior says "linear-probe was sufficient" and your probe disagrees,
    the probe wins. Current artifacts always override stale memory.
  - If `prior_runs.md` shows a prior run aborted_over_budget or probe_failed
    for the same dataset_ref + model_id, factor it in — e.g., halve
    probe_batch_size from the start, or escalate budget to the user before
    launching.
  - The pre-launch subagents (infra, data-profiler, model-selector) already
    received the same prior_runs.md; their JSON artifacts reflect whatever
    they chose to carry forward.