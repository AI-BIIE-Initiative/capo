"""
data_processing_orchestrator.py — Agent-driven raw FASTQ preprocessing on Lambda.

Provisions a fresh Lambda CPU instance, lets the agent discover FASTQ files and
infer the experimental structure (species, date, sorts, libraries) from filenames,
runs the 4-step processing pipeline with all available CPU workers, and pulls the
labeled PLM-ready CSV back locally.

The agent reads the relevant skill documentation itself — it is not told which skill
to use. It discovers what pipeline to build by reading the skills/ directory and
inspecting the input data.

This module also hosts EvaluationHarnessOrchestrator, a leakage-isolated
external evaluation harness that runs two systems (CAPO + General Coding Agent)
on the same raw inputs under matched budgets, freezes their candidate outputs,
and then runs the deterministic Stage-2 evaluation against a held-out gold set.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from capo.evaluation import (
    CandidateAdapter,
    SystemRun,
    generate_harness_plots,
    run_stage2_evaluation,
    write_evaluation_config,
    write_processed_examples,
    write_raw_data_manifest,
    write_run_metadata,
)
from capo.observability import progress as ip
from capo.orchestration.agent_runner import AgentRunner
from capo.orchestration.orchestration import _REPO_ROOT_ORCH, _SKILLS_DIR


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_DATA_PROCESSING_SYSTEM_PROMPT = """\
You are a raw biological data preprocessing specialist for the CAPO framework.
You orchestrate end-to-end conversion of raw sequencing reads into labeled
datasets ready for protein language model (PLM) training.

Your job is:
(0) Read any relevant skill documentation from the skills/ directory on your own —
    you are not told which skill to use; discover it by examining the data.
(1) Discover raw sequencing files in the input directory and infer the full
    experimental structure (species, date, sort gates, sub-library assignments)
    from filenames alone.
(2) Write a self-contained processing script using the constants and pipeline code
    from the skill documentation you read.
(3) Provision or connect to a Lambda instance
(4) Upload data and script, install dependencies, and run the pipeline with all
    available CPU workers.
(5) Monitor until terminal, pull the output back locally, and write a summary.

You have direct access to:
  - Lambda MCP tools prefixed mcp__lambda-repl__
  - Read, Bash, Write, Edit for local file inspection and script preparation

Operating principles:
  1. Call lambda_list_instances to find a reusable
     instance and if not any or if not matching user preference, go straight to lambda_provision_instance.
  2. DISCOVER THE DATA. Use Bash to list the input directory and read filenames.
     Infer species, date, sort IDs, and library assignments from the naming pattern
     — do not rely on pre-configured sample names.
  3. READ THE SKILL FIRST. Before writing any code, read the relevant SKILL.md
     from the skills/ directory. Use the biological constants and step code
     verbatim — do not invent your own pipeline.
  4. PARALLELISM IS MANDATORY. The processing script must use ThreadPoolExecutor
     (Step 1, one worker per sample) and ProcessPoolExecutor (Step 2, one process
     per file) with max_workers = max(1, os.cpu_count() - 1).
  5. ALL DECISIONS ARE BACKED BY JSON. Write processing_config.json before
     uploading; write processing_summary.json at the end. A missing or
     unparseable artifact is a hard failure.
  6. TERMINATE ONLY IF CONFIGURED. Read the terminate_after parameter. If true,
     call lambda_terminate_safe after outputs are confirmed local. Otherwise
     report the ssh_alias so the user can inspect the instance.
  7. EMIT PHASE BOUNDARIES. Use ip.emit("[data-processing] ...") at the start
     of each major step so the user can follow progress.
  8. CLASSIFY FAILURES. On any terminal failure, classify the root cause into
     exactly one of: fastp_not_found / empty_csv / all_filtered / oom /
     subprocess_error / unknown. Write {local_run_dir}/reports/failure.json
     with: {run_id, error_class, message, stdout_tail}.
  9. OUTPUT SCHEMA IS NON-NEGOTIABLE. Every per-species CSV emitted under
     outputs/ must contain a literal `species` column (every row carrying the
     ortholog name from the config, e.g. "possum", "human_new") and an
     `aa_sequence` column (the translated amino-acid sequence). The full
     required column set is: species, Sequence, Library, aa_sequence, count,
     binding, sort. The species column is mandatory — encoding the species
     only in the filename is a contract violation; the downstream evaluation
     harness reads columns, not filenames. If the skill code you copy does not
     already add `df["species"] = config["species"]` before writing, add that
     one line yourself.
 10. WAIT, DON'T POLL. For any operation expected to take >5 minutes (rsync of
     multi-GB FASTQs, fastp on the full input, long-running training), do NOT
     tight-loop `ls -lah` or `du -sh` to check progress — each poll consumes
     one agent turn and the total budget is 300 turns. Instead, issue a single
     blocking shell call:
       Bash: rsync ... && md5sum <files>            # foreground
       Bash: wait $PID && md5sum <files>            # already backgrounded
     One call, one turn, blocks until done. Reserve polling for genuinely
     unknown-duration waits (instance booting), and never poll faster than
     once every 60 seconds.
 11. RAW DATA IS SHARED PER-INSTANCE. The same FASTQ files are reused across
     runs and across systems in the harness. Stage them ONCE at
     ~/raw_data/<basename(input_dir)>/ on the instance, then symlink into the
     per-run input/ directory. Never re-rsync raw bytes into a per-run input/.
     If ~/raw_data/<basename> already contains the *.fastq.gz files, skip the
     upload entirely and symlink straight away.
"""


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_DATA_PROCESSING_PROMPT_TEMPLATE = """\
Run raw sequencing data preprocessing on Lambda using the CAPO framework.

## Run parameters
- run_id:          {run_id}
- input_dir:       {input_dir}
- output_dir:      {output_dir}
- local_run_dir:   {local_run_dir}
- key_path:        {key_path}
- ssh_key_name:    {ssh_key_name}
- instance_type:   {instance_type}   (null = agent picks best CPU-heavy type)
- max_cost_usd:    {max_cost_usd}
- terminate_after: {terminate_after}
- skills_dir:      {skills_dir}

## Steps — execute strictly in order

### Step 0 — Read skill documentation
Scan {skills_dir}/ for SKILL.md files whose description mentions FASTQ, paired-end,
yeast-display, RBD, or binding screen.
Read the matching SKILL.md fully. Extract:
  - All biological constants (constant sequences, WT amino acid sequence, library config)
  - All step functions (run_step1, run_step2, preprocess_rbd, label_sort, main)
  - The output CSV contract (columns, index=False requirement)
  - Failure modes and hard constraints

Do not use prior knowledge — the skill file is the single authoritative source
for constants and pipeline logic.

### Step 1 — Discover FASTQ data
Run via Bash:
  ls {input_dir}/*_R1.fastq.gz

From the filenames, infer:
  - species (e.g. "mouse", "human", "bat", "hamster" — word in the basename)
  - date (6-digit YYMMDD pattern)
  - sort IDs (integer after "_s" or "_sort" in the basename)
  - library assignments: "lib1" or "lib2" (case-insensitive) anywhere in name
  - pool type: "bind" or "non" (or "nonbind") anywhere in the name
  - sample basenames (strip _R1.fastq.gz suffix)

Group samples into sort gates. Each sort gate must have exactly four basenames:
  binder_lib1, binder_lib2, non_lib1, non_lib2.

Build the processing config dict:
{{
  "input_dir": "{input_dir}",
  "output_dir": "<remote_working_dir>",
  "species":    "<inferred>",
  "date":       "<inferred>",
  "sorts": [
    {{
      "sort_id":     <int>,
      "binder_lib1": "<basename>",
      "binder_lib2": "<basename>",
      "non_lib1":    "<basename>",
      "non_lib2":    "<basename>"
    }}
  ]
}}

Write it to {local_run_dir}/processing_config.json.

### Step 2 — Write the processing script
Write {local_run_dir}/process_fastq.py that:
  1. Imports the biological constants exactly as read from the SKILL.md
  2. Implements all step functions (run_step1, run_step2, preprocess_rbd,
     label_sort, run_step4, main) verbatim from the skill
  3. Accepts --config <path_to_json> as its sole CLI argument
  4. Loads the config JSON and calls main(config)
  5. Uses max_workers = max(1, os.cpu_count() - 1) in both executors

The script must be fully self-contained — no imports beyond biopython, pandas,
and the standard library.

### Step 3 — Reuse the existing A10 instance
REUSE the running A10 instance at IP 150.136.243.39 — do NOT provision a new
one and do NOT call lambda_provision_instance. The FASTQ files from the
previous run are already uploaded there, which avoids a multi-hour re-upload.

  mcp__lambda-repl__lambda_preflight(key_path="{key_path}")

Resolve the instance_id via lambda_list_instances (filter by ip="150.136.243.39")
and verify status="active" before opening the session. The SSH alias will be
"lambda-<instance_id>".

If — and only if — that instance is no longer alive (terminated, unreachable,
or status != active), fall back to provisioning a fresh A10:

  mcp__lambda-repl__lambda_list_instance_types()
  mcp__lambda-repl__lambda_provision_instance(
      instance_type="gpu_1x_a10",
      ssh_key_name="{ssh_key_name}",
      name="data-proc-{run_id}"
  )
  # then poll lambda_list_instances until status="active" + IP assigned.

Record instance_id and IP.

### Step 4 — Open session + upload
  mcp__lambda-repl__lambda_start_session(
      host=<IP>,
      user="ubuntu",
      key_path="{key_path}",
      remote_workdir="~/capo_runs/{run_id}",
      local_workdir="{local_run_dir}"
  )
  mcp__lambda-repl__lambda_ensure_remote_tmux(
      ssh_alias=<alias>,
      key_path="{key_path}"
  )

Create remote directory structure:
  ssh ... "mkdir -p ~/capo_runs/{run_id}/outputs ~/capo_runs/{run_id}/input"

Upload the local run dir (script + config):
  mcp__lambda-repl__lambda_upload_run(
      ssh_target=<alias>,
      local_run_dir="{local_run_dir}",
      remote_run_dir="~/capo_runs/{run_id}",
      key_path="{key_path}"
  )

Stage the raw FASTQ files into a SHARED canonical location on the instance
so re-runs (and the sibling system in the harness) reuse the same upload.

  RAW_BASENAME=$(basename {input_dir})              # e.g. human_raw
  REMOTE_RAW=~/raw_data/$RAW_BASENAME               # e.g. ~/raw_data/human_raw

  # 1. Check if the canonical raw dir already has the FASTQs.
  ssh ... "mkdir -p $REMOTE_RAW && \\
           ls $REMOTE_RAW/*.fastq.gz >/dev/null 2>&1 && echo PRESENT || echo MISSING"

  # 2. If MISSING — upload ONCE to the canonical path (single blocking call).
  #    If PRESENT — skip this step entirely.
  Bash: rsync -avz --partial -e "ssh -i {key_path} -o StrictHostKeyChecking=no" \\
      {input_dir}/ <alias>:$REMOTE_RAW/ && echo RAW_RSYNC_DONE

  # 3. Symlink the canonical FASTQs into the per-run input/ dir. Always.
  ssh ... "ln -sf $REMOTE_RAW/*.fastq.gz ~/capo_runs/{run_id}/input/"

NEVER rsync the raw bytes into ~/capo_runs/{run_id}/input/ directly. The
per-run input/ holds symlinks only; ~/raw_data/<basename> holds the bytes.

Update processing_config.json on the remote so "input_dir" points to
~/capo_runs/{run_id}/input and "output_dir" to ~/capo_runs/{run_id}/outputs.
Overwrite the remote copy via lambda_run_command (python -c 'import json, pathlib; ...').

### Step 5 — Install dependencies
Run inside capo_remote tmux:
  conda install -c bioconda fastp -y

Then:
  pip install biopython pandas

After both complete, verify:
  fastp --version

If fastp is not on PATH after install, fail with error_class=fastp_not_found.

### Step 6 — Run pipeline
Send to capo_remote tmux:
  nohup python ~/capo_runs/{run_id}/process_fastq.py \\
      --config ~/capo_runs/{run_id}/processing_config.json \\
      > ~/capo_runs/{run_id}/outputs/stdout.log 2>&1 & \\
  echo $! > ~/capo_runs/{run_id}/outputs/proc.pid

Capture the PID. Confirm it is running:
  cat ~/capo_runs/{run_id}/outputs/proc.pid
  ps -p <pid> > /dev/null 2>&1 && echo UP || echo DEAD

### Step 7 — Monitor until terminal
Every 60 seconds, poll via lambda_run_command:
  tail -20 ~/capo_runs/{run_id}/outputs/stdout.log

Stop when stdout.log contains "Wrote" followed by a sequence count (success),
or an exception/traceback (failure).

On success: continue to Step 8.
On failure: read the full stdout.log, classify the error, write
{local_run_dir}/reports/failure.json and exit.

### Step 8 — Pull outputs + write summary
Rsync outputs back:
  Bash: rsync -avz --partial -e "ssh -i {key_path} -o StrictHostKeyChecking=no" \\
    <alias>:~/capo_runs/{run_id}/outputs/ {local_run_dir}/outputs/

Locate the output CSV (pattern: <species>_<date>.csv) in
{local_run_dir}/outputs/.

Parse it with pandas to extract:
  total_sequences = len(df)
  bind_count      = (df["binding"] == "bind").sum()
  non_count       = (df["binding"] == "non").sum()
  sort_ids        = sorted(df["sort"].unique().tolist())

Assert the required columns are present and populated:
  required = {{"species", "aa_sequence", "binding", "sort"}}
  missing = required - set(df.columns)
  assert not missing, f"output CSV missing required columns: {{missing}}"
  assert df["species"].notna().all() and (df["species"].astype(str).str.len() > 0).all(), \\
      "species column has empty/NaN rows"

If the assertion fails, treat it as a terminal failure with error_class=empty_csv,
write the failure.json, and stop.

Read the instance hourly rate from skills/cost-estimation/references/lambda-pricing.md.
Compute actual_cost_usd = elapsed_hours × hourly_rate.

Write {local_run_dir}/reports/processing_summary.json:
{{
  "run_id":           "{run_id}",
  "instance_type":    "<type>",
  "state":            "completed",
  "output_csv":       "<path>",
  "total_sequences":  <int>,
  "bind_count":       <int>,
  "non_count":        <int>,
  "sorts":            [<int>, ...],
  "actual_cost_usd":  <float>,
  "elapsed_seconds":  <int>
}}

### Step 9 — Terminate (if configured)
If terminate_after is true:
  mcp__lambda-repl__lambda_terminate_safe(
      instance_id=<id>,
      expected_ssh_key_names=["{ssh_key_name}"]
  )
  Confirm termination.
Otherwise:
  Report the ssh_alias (<alias>) and remote run dir
  (~/capo_runs/{run_id}) for manual inspection.

## Begin
Proceed now. Start with Step 0.
"""


# ---------------------------------------------------------------------------
# General Coding Agent — system prompt + template (skills-free baseline)
# ---------------------------------------------------------------------------
#
# Same task, same MCP tools, same Lambda flow, same output contract — but the
# agent has NO access to the skills/ directory and is expected to reverse-
# engineer the FASTQ preprocessing pipeline from first principles. This is the
# fair-comparison baseline against CAPO.
#

_GCA_SYSTEM_PROMPT = """\
You are a senior bioinformatician asked to clean paired-end sequencing reads
from a yeast-display ACE2 binding screen and produce a single labeled CSV
ready for protein language model (PLM) training.

ABSOLUTE RULES:
  1. You MUST NOT read, list, glob, or otherwise access any file under
     "skills/" in the repository. You also MUST NOT load any documentation
     that originated under skills/. This is a leakage control. If you find
     yourself reaching for skills/, stop and reason from first principles.
  2. You MUST NOT access the held-out evaluation dataset. You will be told
     when evaluation begins; until then, treat any HuggingFace dataset named
     "BIIE-AI/ace2_binding" as forbidden.
  3. You are NOT given a pre-baked pipeline. You design the pipeline based
     only on the raw FASTQ filenames and standard bioinformatics knowledge.
  4. Output contract is fixed (see prompt): one row per unique full-length
     RBD protein variant, per species. Columns: species, aa_seq, sort,
     binding (bind/non). The output unit is the protein variant assayed —
     NOT a sequencing read or read fragment.

You have direct access to:
  - Lambda MCP tools prefixed mcp__lambda-repl__
  - Read, Bash, Write, Edit for local file inspection and script preparation

Operating principles:
  1. Provision a fresh Lambda CPU instance directly (an A10!!).
  2. Discover the experimental design from filenames alone (species, date,
     sort gates, lib1/lib2). Yeast-display ACE2 screens conventionally split
     each species into two source libraries and gate sorts; you infer the
     groupings from basename patterns.
  3. Design a pipeline: paired-end adapter+quality trim → constant-region
     anchor extraction → translation of the RBD ORF → per-(species, sort)
     dedup → label as 'bind' or 'non' based on the sort gate. Use whichever
     standard tools you prefer (fastp / cutadapt / Biopython / pandas).
  4. Parallelism is mandatory: ThreadPoolExecutor or ProcessPoolExecutor with
     max_workers = max(1, os.cpu_count() - 1).
  5. All decisions backed by JSON. Write processing_config.json before
     running; write reports/processing_summary.json at the end.
  6. EMIT PHASE BOUNDARIES with ip.emit("[general-coding-agent] ...").
  7. On terminal failure, classify into exactly one of: fastp_not_found /
     empty_csv / all_filtered / oom / subprocess_error / unknown. Write
     {local_run_dir}/reports/failure.json.
  8. WAIT, DON'T POLL. For any operation expected to take >5 minutes (rsync of
     multi-GB FASTQs, fastp on the full input, long-running training), do NOT
     tight-loop `ls -lah` or `du -sh` to check progress — each poll consumes
     one agent turn and the total budget is 300 turns. Instead, issue a single
     blocking shell call:
       Bash: rsync ... && md5sum <files>            # foreground
       Bash: wait $PID && md5sum <files>            # already backgrounded
     One call, one turn, blocks until done. Reserve polling for genuinely
     unknown-duration waits (instance booting), and never poll faster than
     once every 60 seconds.
  9. RAW DATA IS SHARED PER-INSTANCE. The same FASTQ files are reused across
     runs and across systems in the harness. Stage them ONCE at
     ~/raw_data/<basename(input_dir)>/ on the instance, then symlink into the
     per-run input/ directory. Never re-rsync raw bytes into a per-run input/.
     If ~/raw_data/<basename> already contains the *.fastq.gz files, skip the
     upload entirely and symlink straight away.
"""

_GCA_PROMPT_TEMPLATE = """\
Run raw paired-end FASTQ preprocessing on Lambda to produce a labeled CSV for
downstream protein language model training.

## Run parameters
- run_id:          {run_id}
- input_dir:       {input_dir}
- output_dir:      {output_dir}
- local_run_dir:   {local_run_dir}
- key_path:        {key_path}
- ssh_key_name:    {ssh_key_name}
- instance_type:   {instance_type}   (null = pick a CPU-heavy type ≥ 8 vCPUs)
- max_cost_usd:    {max_cost_usd}     (HARD CAP — stop if elapsed × hourly > this)
- max_runtime_hours: {max_runtime_hours}  (HARD CAP — stop if exceeded)
- terminate_after: {terminate_after}

## Output contract (NON-NEGOTIABLE)
The output unit is one row per unique full-length RBD protein variant, per
species — NOT one row per sequencing read.

The canonical RBD length is defined by this wild-type SARS-CoV-2 RBD sequence:
  NITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNLAPFFTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVSGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYSFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST

All output sequences must be full canonical length RBDs matching the above.

Columns (at minimum):
  - species   (string, e.g. "mouse", "cattle", "ihbat", "pangolin")
  - aa_seq    (full-length RBD protein variant — amino-acid sequence)
  - sort      (integer sort gate id parsed from filename)
  - binding   (literal string "bind" or "non")

Write one CSV per species at:
  {local_run_dir}/outputs/<species>_<date>.csv
where <date> is the inferred YYMMDD pattern from the filenames.

## Steps — execute strictly in order

### Step 1 — Discover FASTQ data
Run via Bash:
  ls {input_dir}/*_R1.fastq.gz

From filenames alone, infer:
  - species (single word in the basename)
  - date (6-digit YYMMDD)
  - sort id (integer after "_s" or "_sort")
  - library assignment (lib1 / lib2, case-insensitive)
  - pool type (bind / non / nonbind)

Group samples into sort gates with exactly four basenames each:
  binder_lib1, binder_lib2, non_lib1, non_lib2.

Build processing_config.json with the inferred structure (same shape as a
sensible {{species, date, sorts: [{{sort_id, binder_lib1, binder_lib2,
non_lib1, non_lib2}}]}} dict).

Write it to {local_run_dir}/processing_config.json.

### Step 2 — Design and write process_fastq.py
You design the pipeline from first principles. Recommended stages:
  (a) paired-end adapter + quality trim (fastp or cutadapt)
  (b) anchor on constant flanking sequences expected in this yeast-display
      construct (you may inspect a few reads with Bash to discover them)
  (c) translate the RBD ORF to amino acids
  (d) collapse duplicate reads per (species, sort, library)
  (e) emit per-species CSV with columns {{species, aa_seq, sort, binding}}

Use max_workers = max(1, os.cpu_count() - 1) for any parallel stage. Script
must accept --config <path> as its sole CLI argument and call main(config).

### Step 3 — Reuse the existing A10 instance
REUSE the running A10 instance at IP 150.136.243.39 — do NOT provision a new
one and do NOT call lambda_provision_instance. The FASTQ files from the
previous run are already uploaded there, which avoids a multi-hour re-upload.

  mcp__lambda-repl__lambda_preflight(key_path="{key_path}")

Resolve instance_id via lambda_list_instances (filter by ip="150.136.243.39")
and verify status="active" before opening the session. SSH alias is
"lambda-<instance_id>".

Fall back to a fresh A10 only if that instance is no longer alive:
  mcp__lambda-repl__lambda_list_instance_types()
  mcp__lambda-repl__lambda_provision_instance(
      instance_type="gpu_1x_a10",
      ssh_key_name="{ssh_key_name}",
      name="data-proc-gca-{run_id}",
  )

### Step 4 — Upload + run
  mcp__lambda-repl__lambda_start_session(...)
  mcp__lambda-repl__lambda_ensure_remote_tmux(...)
  mcp__lambda-repl__lambda_upload_run(
      ssh_target=<alias>, local_run_dir="{local_run_dir}",
      remote_run_dir="~/capo_runs/{run_id}", key_path="{key_path}",
  )

Stage the raw FASTQ files into a SHARED canonical location so re-runs and
the sibling CAPO system reuse the same upload:

  RAW_BASENAME=$(basename {input_dir})              # e.g. human_raw
  REMOTE_RAW=~/raw_data/$RAW_BASENAME               # e.g. ~/raw_data/human_raw

  # 1. Check whether the canonical raw dir already has the FASTQs.
  ssh ... "mkdir -p $REMOTE_RAW && \\
           ls $REMOTE_RAW/*.fastq.gz >/dev/null 2>&1 && echo PRESENT || echo MISSING"

  # 2. If MISSING — upload ONCE to the canonical path (single blocking call).
  #    If PRESENT — skip this step entirely.
  Bash: rsync -avz --partial -e "ssh -i {key_path} -o StrictHostKeyChecking=no" \\
      {input_dir}/ <alias>:$REMOTE_RAW/ && echo RAW_RSYNC_DONE

  # 3. Symlink the canonical FASTQs into the per-run input/ dir. Always.
  ssh ... "ln -sf $REMOTE_RAW/*.fastq.gz ~/capo_runs/{run_id}/input/"

NEVER rsync raw bytes into ~/capo_runs/{run_id}/input/ directly.

Install fastp + Biopython + pandas. Send to capo_remote tmux:
  nohup python ~/capo_runs/{run_id}/process_fastq.py \\
      --config ~/capo_runs/{run_id}/processing_config.json \\
      > ~/capo_runs/{run_id}/outputs/stdout.log 2>&1 & \\
  echo $! > ~/capo_runs/{run_id}/outputs/proc.pid

### Step 5 — Monitor under budget
Poll every 60s. Watch for:
  - Success: "Wrote" + sequence count in stdout
  - Failure: exception/traceback
  - Budget breach: elapsed_hours × hourly_rate > {max_cost_usd}
                   OR elapsed_hours > {max_runtime_hours}

On budget breach: kill the remote pid, classify as state="budget_exceeded",
and continue to Step 6 with whatever output exists.

### Step 6 — Pull outputs + write summary
Rsync outputs back to {local_run_dir}/outputs/.
Parse the per-species CSVs into a combined frame (pandas) to extract:
  total_sequences = total rows
  bind_count      = (binding == "bind").sum()
  non_count       = (binding == "non").sum()

Compute actual_cost_usd from the instance hourly rate (you may use
~/.ssh-side knowledge of Lambda pricing — current cpu_8x is ~$0.30/hr;
adjust based on the instance you provisioned).

Write {local_run_dir}/reports/processing_summary.json:
{{
  "run_id": "{run_id}",
  "instance_type": "<type>",
  "state": "completed" | "failed" | "budget_exceeded",
  "output_csv_glob": "{local_run_dir}/outputs/*.csv",
  "total_sequences": <int>,
  "bind_count": <int>,
  "non_count": <int>,
  "actual_cost_usd": <float>,
  "elapsed_seconds": <int>,
  "raw_reads_processed": <int>,
  "reads_retained_after_qc": <int>
}}

### Step 7 — Terminate (if configured)
If terminate_after is true, lambda_terminate_safe(...). Otherwise report
the ssh_alias and remote run dir.

## Begin
Proceed now. Start with Step 1. You have NOT been given any skill
documentation — design the pipeline yourself.
"""


# ---------------------------------------------------------------------------
# HuggingFace Hub data staging preamble
# ---------------------------------------------------------------------------

def _hf_data_preamble(input_hf_ref: str, run_id: str) -> str:
    """Prompt preamble injected when raw FASTQs live on HuggingFace Hub.

    Overrides the local rsync instructions in Steps 1 and 4 of both prompt
    templates.  Lambda instances have datacenter-speed access to HF Hub
    (~500 Mbps), so a 25 GB dataset downloads in minutes rather than hours.
    """
    repo_name = input_hf_ref.rstrip("/").split("/")[-1]
    return (
        f"## Data source — HuggingFace Hub (overrides Steps 1 and 4)\n"
        f"\n"
        f"Raw FASTQs live in the private HF dataset `{input_hf_ref}` — they are NOT\n"
        f"on the local machine. Two steps change:\n"
        f"\n"
        f"**Step 1 — FASTQ discovery:** list repo files from the Hub instead of a\n"
        f"local `ls`:\n"
        f"    Bash: python3 -c \"from huggingface_hub import list_repo_files; \\\n"
        f"        print('\\n'.join(f for f in list_repo_files('{input_hf_ref}',\n"
        f"        repo_type='dataset') if f.endswith('_R1.fastq.gz')))\"\n"
        f"Use the returned basenames to infer species, date, sort IDs, and libraries\n"
        f"exactly as you would from a local ls. Write processing_config.json as normal.\n"
        f"\n"
        f"**Step 4 — staging:** replace the rsync block with an HF Hub download:\n"
        f"    REMOTE_RAW=~/raw_data/{repo_name}\n"
        f"    # idempotent — skip if data already present\n"
        f"    ssh <alias> \"mkdir -p $REMOTE_RAW && \\\n"
        f"        ls $REMOTE_RAW/*.fastq.gz >/dev/null 2>&1 && echo PRESENT || \\\n"
        f"        (pip install -q huggingface_hub && \\\n"
        f"         huggingface-cli download {input_hf_ref} \\\n"
        f"             --repo-type dataset --local-dir $REMOTE_RAW && echo DOWNLOADED)\"\n"
        f"    ssh <alias> \"ln -sf $REMOTE_RAW/*.fastq.gz ~/capo_runs/{run_id}/input/\"\n"
        f"\n"
        f"For private repos, set the HF token on the remote before downloading:\n"
        f"    ssh <alias> \"huggingface-cli login --token <HF_TOKEN>\"\n"
        f"or pass it inline:\n"
        f"    ssh <alias> \"HUGGING_FACE_HUB_TOKEN=<token> huggingface-cli download ...\"\n"
        f"\n"
        f"All other steps (provision, install fastp, run pipeline, pull outputs) are\n"
        f"unchanged.\n"
        f"\n"
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DataProcessingResult:
    run_id: str
    local_run_dir: Path
    output_csv: Path | None
    state: str              # "completed" | "failed" | "unknown"
    total_sequences: int | None
    bind_count: int | None
    non_count: int | None
    instance_type: str | None
    actual_cost_usd: float | None
    agent_cost_usd: float | None
    answer: str


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DataProcessingOrchestrator:
    """
    Agent-driven raw FASTQ preprocessing on a freshly provisioned Lambda instance.

    The agent discovers the data, reads the processing skill, builds the pipeline
    script, provisions a new instance, runs with maximum CPU parallelism, and
    returns the labeled PLM-ready CSV locally.

    All run-tunable parameters are required — values come from the YAML
    config (scripts/configs/raw_data_processing.yaml). The harness sets the
    trailing internal-only kwargs (system_kind, system_display_name,
    local_run_dir_override) when it instantiates per-system runners.

    Example::

        orch = DataProcessingOrchestrator(
            key_path="~/.ssh/lambda_key",
            ssh_key_name="my-key",
            input_dir="/data/raw_fastq",
            output_dir="/data/processed",
            max_cost_usd=80.0,
            max_runtime_hours=12.0,
            instance_type=None,
            terminate_after=False,
            model_name="claude-sonnet-4-6",
            max_turns=300,
        )
        result = orch.run_sync()
        print(result.output_csv, result.total_sequences)
    """

    def __init__(
        self,
        key_path: str | Path,
        ssh_key_name: str,
        input_dir: str | Path,
        output_dir: str | Path,
        max_cost_usd: float,
        max_runtime_hours: float,
        instance_type: str | None,
        terminate_after: bool,
        model_name: str,
        max_turns: int,
        # Internal-only kwargs (set by the harness, not by the YAML)
        cwd: str | Path | None = None,
        system_kind: str = "capo",
        system_display_name: str | None = None,
        local_run_dir_override: str | Path | None = None,
        input_hf_ref: str | None = None,
    ) -> None:
        self.key_path = str(Path(key_path).expanduser().resolve())
        self.ssh_key_name = ssh_key_name
        self.input_dir = str(Path(input_dir).expanduser().resolve())
        self.input_hf_ref = input_hf_ref or None
        self.output_dir = str(Path(output_dir).expanduser().resolve())
        self.max_cost_usd = max_cost_usd
        self.max_runtime_hours = max_runtime_hours
        self.instance_type = instance_type or "null"
        self.terminate_after = terminate_after
        if system_kind not in ("capo", "general_coding_agent"):
            raise ValueError(f"Unknown system_kind={system_kind!r}; expected capo|general_coding_agent")
        self.system_kind = system_kind
        self.system_display_name = system_display_name or (
            "CAPO" if system_kind == "capo" else "General Coding Agent"
        )
        self.local_run_dir_override = Path(local_run_dir_override).expanduser() if local_run_dir_override else None

        sys_prompt = (
            _DATA_PROCESSING_SYSTEM_PROMPT if system_kind == "capo" else _GCA_SYSTEM_PROMPT
        )
        self._runner = AgentRunner(
            model_name=model_name,
            allowed_tools=["Read", "Bash", "Write", "Edit", "mcp__lambda-repl__*"],
            system_prompt=sys_prompt,
            permission_mode="acceptEdits",
            max_turns=max_turns,
            cwd=str(cwd or _REPO_ROOT_ORCH),
        )

    async def run(
        self,
        run_id: str | None = None,
    ) -> DataProcessingResult:
        if run_id:
            rid = run_id
        else:
            ts = datetime.now().strftime("%Y%m%d-%H%M")
            h  = uuid.uuid4().hex[:4]
            kind_tag = "capo" if self.system_kind == "capo" else "gca"
            rid = f"proc-fastq-{kind_tag}-{ts}-{h}"

        if self.local_run_dir_override is not None:
            local_run_dir = self.local_run_dir_override
        else:
            local_run_dir = _REPO_ROOT_ORCH / "lambda" / "runs" / "data-processing" / rid
        local_run_dir.mkdir(parents=True, exist_ok=True)

        outputs_dir = local_run_dir / "outputs"
        reports_dir = local_run_dir / "reports"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        emitter = ip.ProgressEmitter(
            stdout_log=outputs_dir / ip.RUN_LOG_NAME,
            stderr_log=outputs_dir / ip.RUN_ERR_LOG_NAME,
        )
        token = ip.set_emitter(emitter)
        try:
            ip.emit(f"[data-processing] Starting run {rid}")
            ip.emit(f"[data-processing] input_dir  = {self.input_dir}")
            ip.emit(f"[data-processing] output_dir = {self.output_dir}")

            preamble = _hf_data_preamble(self.input_hf_ref, rid) if self.input_hf_ref else ""
            template = (
                _DATA_PROCESSING_PROMPT_TEMPLATE
                if self.system_kind == "capo"
                else _GCA_PROMPT_TEMPLATE
            )
            common_fields = dict(
                run_id=rid,
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                local_run_dir=str(local_run_dir),
                key_path=self.key_path,
                ssh_key_name=self.ssh_key_name,
                instance_type=self.instance_type,
                max_cost_usd=self.max_cost_usd,
                terminate_after=str(self.terminate_after).lower(),
            )
            if self.system_kind == "capo":
                prompt = preamble + template.format(skills_dir=str(_SKILLS_DIR), **common_fields)
            else:
                # general coding agent is stopped after a fixed time budget instead of a cost budget
                prompt = preamble + template.format(
                    max_runtime_hours=self.max_runtime_hours, **common_fields
                )

            result = await self._runner.generate(prompt=prompt)

            # Parse processing_summary.json if it was written
            summary_path = reports_dir / "processing_summary.json"
            output_csv: Path | None = None
            total_sequences: int | None = None
            bind_count: int | None = None
            non_count: int | None = None
            instance_type: str | None = None
            actual_cost_usd: float | None = None
            state = "unknown"

            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    state           = summary.get("state", "unknown")
                    total_sequences = summary.get("total_sequences")
                    bind_count      = summary.get("bind_count")
                    non_count       = summary.get("non_count")
                    instance_type   = summary.get("instance_type")
                    actual_cost_usd = summary.get("actual_cost_usd")
                    csv_str         = summary.get("output_csv")
                    if csv_str:
                        output_csv = Path(csv_str)
                except (json.JSONDecodeError, OSError):
                    pass
            elif (reports_dir / "failure.json").exists():
                state = "failed"
            else:
                # Fallback: look for any CSV in outputs/
                csvs = list(outputs_dir.glob("*.csv"))
                if csvs:
                    output_csv = csvs[0]
                    state = "completed"

            emitter.emit_final_summary(rid, state, result.total_cost_usd)

            return DataProcessingResult(
                run_id=rid,
                local_run_dir=local_run_dir,
                output_csv=output_csv,
                state=state,
                total_sequences=total_sequences,
                bind_count=bind_count,
                non_count=non_count,
                instance_type=instance_type,
                actual_cost_usd=actual_cost_usd,
                agent_cost_usd=result.total_cost_usd,
                answer=result.answer,
            )
        except Exception as exc:
            ip.error(f"[data-processing] Run {rid} failed: {exc}")
            raise
        finally:
            ip._emitter.reset(token)

    def run_sync(self, run_id: str | None = None) -> DataProcessingResult:
        coro = self.run(run_id=run_id)
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()


# ===========================================================================
# Leakage-isolated external evaluation harness
# ===========================================================================
#
# Runs N systems (default: CAPO + General Coding Agent) on the same raw inputs
# under matched budgets, freezes their candidate outputs into the canonical
# format described in the framework spec, then runs the deterministic Stage-2
# evaluation against a held-out gold set (BIIE-AI/ace2_binding by default).
#
# The leakage guarantee comes from the call graph:
#   1. The Stage-1 sub-run uses DataProcessingOrchestrator. The system prompts
#      forbid touching the gold dataset.
#   2. Stage-2 is a separate code path in this module — pure Python, no agent —
#      that lazily imports HuggingFace `datasets` only after every system's
#      Stage-1 sub-run has returned. There is no code path where gold can load
#      before preprocessing finishes.


@dataclass
class SystemSpec:
    """One row of the harness systems: config list."""
    name: str                        # "CAPO" | "General Coding Agent"
    kind: str                        # "capo" | "general_coding_agent"
    setting: str = "budget_matched"  # "budget_matched" | "best_effort"
    model_name: str = "claude-sonnet-4-6"
    max_turns: int = 300


@dataclass
class HarnessResult:
    eval_run_id: str
    eval_run_dir: Path
    system_results: list[DataProcessingResult] = field(default_factory=list)
    summary_metrics_csv: Path | None = None
    per_species_metrics_csv: Path | None = None
    efficiency_metrics_csv: Path | None = None
    error_analysis_csv: Path | None = None
    statistical_tests_csv: Path | None = None
    gold_alignment_parquet: Path | None = None
    gold_alignment_review_csv: Path | None = None
    plots_dir: Path | None = None
    state: str = "unknown"           # "completed" | "preprocessing_failed" | "eval_failed"


class EvaluationHarnessOrchestrator:
    """Two-zone harness: Stage-1 preprocessing per system, Stage-2 evaluation.

    Parameters mirror DataProcessingOrchestrator for the shared inputs
    (Lambda credentials, paths, budget) and add an systems list and an
    eval_config describing the gold set, match keys, and statistical tests.

    Output layout::

        <out_root>/<eval_run_id>/
          README.md
          evaluation_config.yaml           (frozen contract)
          run_metadata.json                (harness-level provenance)
          systems/
            <system_name>/
              evaluation_config.yaml       (per-system copy)
              run_metadata.json            (per-system provenance)
              raw_data_manifest.csv
              processed_examples.parquet
              processed_examples_sample.csv
              artifact_hashes.sha256       (FREEZE LINE — gold loads only after this)
              outputs/                     (raw agent CSVs + stdout)
              reports/                     (processing_summary.json, failure.json)
          evaluation/
            gold_alignment.parquet
            gold_alignment_review.csv
            summary_metrics.csv
            per_species_metrics.csv
            efficiency_metrics.csv
            error_analysis.csv
            statistical_tests.csv
    """

    def __init__(
        self,
        key_path: str | Path,
        ssh_key_name: str,
        input_dir: str | Path,
        output_dir: str | Path,
        systems: list[SystemSpec],
        eval_config: dict,
        max_cost_usd: float,
        max_runtime_hours: float,
        instance_type: str | None,
        terminate_after: bool,
        input_hf_ref: str | None = None,
    ) -> None:
        if not systems:
            raise ValueError("EvaluationHarnessOrchestrator requires at least one system")
        self.key_path = str(Path(key_path).expanduser().resolve())
        self.ssh_key_name = ssh_key_name
        self.input_dir = str(Path(input_dir).expanduser().resolve())
        self.output_dir = str(Path(output_dir).expanduser().resolve())
        self.systems = systems
        self.eval_config = eval_config
        self.max_cost_usd = max_cost_usd
        self.max_runtime_hours = max_runtime_hours
        self.instance_type = instance_type
        self.terminate_after = terminate_after
        self.input_hf_ref = input_hf_ref or None

    # ----- public entry --------------------------------------------------

    async def run(self, eval_run_id: str | None = None) -> HarnessResult:
        if eval_run_id is None:
            ts = datetime.now().strftime("%Y%m%d-%H%M")
            eval_run_id = f"ace2-eval-{ts}-{uuid.uuid4().hex[:4]}"

        eval_root = _REPO_ROOT_ORCH / "lambda" / "runs" / "data-processing-eval" / eval_run_id
        eval_root.mkdir(parents=True, exist_ok=True)
        systems_root = eval_root / "systems"
        systems_root.mkdir(exist_ok=True)
        eval_dir = eval_root / "evaluation"
        eval_dir.mkdir(exist_ok=True)

        ip.emit(f"[harness] eval_run_id = {eval_run_id}")
        ip.emit(f"[harness] eval_root   = {eval_root}")

        # 1. Freeze the pre-registered evaluation contract at the top level
        write_evaluation_config(eval_root, self.eval_config)

        # 2. Harness-level run metadata (system list + budget + leakage stance)
        harness_metadata = {
            "eval_run_id": eval_run_id,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "systems": [
                {"name": s.name, "kind": s.kind, "setting": s.setting, "model_name": s.model_name}
                for s in self.systems
            ],
            "budget": {
                "max_cost_usd": self.max_cost_usd,
                "max_runtime_hours": self.max_runtime_hours,
            },
            "gold_dataset_name": self.eval_config.get("gold_set", {}).get("dataset_ref"),
            "gold_access_allowed_during_preprocessing": False,
            "leakage_controls": [
                "system prompts forbid reading the gold dataset during preprocessing",
                "candidate outputs are SHA256-hashed before gold loading",
                "Stage-2 verifies all freezes and refuses to load gold otherwise",
                "GoldEvaluator imports HuggingFace `datasets` lazily, only after verify_all_freezes()",
            ],
        }
        write_run_metadata(eval_root, harness_metadata)

        # 3. Run each system sequentially. Sharing one Lambda budget across
        #    parallel runs adds noise to cost accounting, so serialize.
        system_results: list[DataProcessingResult] = []
        system_runs: list[SystemRun] = []
        for spec in self.systems:
            sys_dir = systems_root / _slug(spec.name)
            sys_dir.mkdir(parents=True, exist_ok=True)
            ip.emit(f"[harness] starting system={spec.name} kind={spec.kind} setting={spec.setting}")

            orch = DataProcessingOrchestrator(
                key_path=self.key_path,
                ssh_key_name=self.ssh_key_name,
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                max_cost_usd=self.max_cost_usd,
                max_runtime_hours=self.max_runtime_hours,
                instance_type=self.instance_type,
                terminate_after=self.terminate_after,
                model_name=spec.model_name,
                max_turns=spec.max_turns,
                system_kind=spec.kind,
                system_display_name=spec.name,
                local_run_dir_override=sys_dir,
                input_hf_ref=self.input_hf_ref,
            )
            try:
                res = await orch.run()
            except Exception as exc:
                ip.error(f"[harness] system {spec.name} crashed: {exc}")
                # Synthesize a failed result so the harness still produces
                # provenance — Stage 2 will detect missing parquet and abort.
                res = DataProcessingResult(
                    run_id=f"{spec.kind}-crash",
                    local_run_dir=sys_dir,
                    output_csv=None, state="failed",
                    total_sequences=None, bind_count=None, non_count=None,
                    instance_type=None, actual_cost_usd=None, agent_cost_usd=None,
                    answer=str(exc),
                )
            system_results.append(res)

            # ---- Freeze this system's output into the canonical schema
            sys_run = await self._freeze_system_output(spec, res, sys_dir, eval_run_id)
            if sys_run is not None:
                system_runs.append(sys_run)

        # 4. Stage 2 — gold load + evaluation
        if len(system_runs) < 1:
            ip.error("[harness] no system produced a frozen Stage-1 artifact — aborting eval")
            self._write_readme(eval_root, [], harness_metadata, eval_status="preprocessing_failed")
            return HarnessResult(
                eval_run_id=eval_run_id, eval_run_dir=eval_root,
                system_results=system_results, state="preprocessing_failed",
            )

        try:
            ip.emit(f"[harness] launching Stage-2 evaluation with {len(system_runs)} system(s)")
            outputs = run_stage2_evaluation(
                systems=system_runs,
                eval_config=self.eval_config,
                out_dir=eval_dir,
                rng_seed=int(self.eval_config.get("rng_seed", 0)),
            )
            state = "completed"
        except Exception as exc:
            ip.error(f"[harness] Stage-2 evaluation failed: {exc}")
            outputs = {}
            state = "eval_failed"

        # generate comparison plots from Stage-2 CSVs (non-fatal on failure).
        plots_dir: Path | None = None
        if state == "completed":
            try:
                plots_dir = generate_harness_plots(eval_dir)
                if plots_dir:
                    ip.emit(f"[harness] comparison plots written to {plots_dir}")
            except Exception as exc:
                ip.error(f"[harness] plot generation failed (non-fatal): {exc}")

        self._write_readme(eval_root, system_runs, harness_metadata, eval_status=state)

        return HarnessResult(
            eval_run_id=eval_run_id,
            eval_run_dir=eval_root,
            system_results=system_results,
            summary_metrics_csv=outputs.get("summary_metrics_csv"),
            per_species_metrics_csv=outputs.get("per_species_metrics_csv"),
            efficiency_metrics_csv=outputs.get("efficiency_metrics_csv"),
            error_analysis_csv=outputs.get("error_analysis_csv"),
            statistical_tests_csv=outputs.get("statistical_tests_csv"),
            gold_alignment_parquet=outputs.get("gold_alignment_parquet"),
            gold_alignment_review_csv=outputs.get("gold_alignment_review_csv"),
            plots_dir=plots_dir,
            state=state,
        )

    def run_sync(self, eval_run_id: str | None = None) -> HarnessResult:
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.run(eval_run_id=eval_run_id)).result()

    # ----- helpers -------------------------------------------------------

    async def _freeze_system_output(
        self,
        spec: SystemSpec,
        result: DataProcessingResult,
        sys_dir: Path,
        eval_run_id: str,
    ) -> SystemRun | None:
        """Convert one system's agent CSV output into the canonical frozen package.

        Returns a SystemRun ready for Stage-2 evaluation, or None if the
        agent did not produce a usable candidate dataset.
        """
        outputs_dir = sys_dir / "outputs"
        # Collect every CSV the agent produced under outputs/ — one per species
        candidate_csvs = sorted(outputs_dir.glob("*.csv")) if outputs_dir.exists() else []
        if not candidate_csvs:
            ip.error(f"[harness] system {spec.name} produced no CSV in {outputs_dir}")
            return None

        # Per-system evaluation config (copy of the harness contract — frozen
        # alongside the parquet so the artifact set is self-contained).
        write_evaluation_config(sys_dir, self.eval_config)

        # Per-system run metadata
        elapsed = 0.0
        actual_cost = 0.0
        if result.actual_cost_usd is not None:
            actual_cost = float(result.actual_cost_usd)
        # elapsed_seconds lives in processing_summary.json; load it if present
        summary_path = sys_dir / "reports" / "processing_summary.json"
        raw_reads = retained_reads = None
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                elapsed = float(summary.get("elapsed_seconds", 0.0))
                raw_reads = summary.get("raw_reads_processed")
                retained_reads = summary.get("reads_retained_after_qc")
            except Exception:
                pass

        sys_metadata = {
            "run_id": result.run_id,
            "eval_run_id": eval_run_id,
            "system_name": spec.name,
            "system_kind": spec.kind,
            "comparison_setting": spec.setting,
            "model_name": spec.model_name,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "gold_dataset_name": self.eval_config.get("gold_set", {}).get("dataset_ref"),
            "gold_access_allowed_during_preprocessing": False,
            "runtime_seconds": elapsed,
            "estimated_cost_usd": actual_cost,
            "agent_cost_usd": result.agent_cost_usd,
            "instance_type": result.instance_type,
            "input_dir": self.input_dir,
        }
        write_run_metadata(sys_dir, sys_metadata)

        # Raw data manifest — record every FASTQ under input_dir
        manifest_rows: list[dict] = []
        input_path = Path(self.input_dir)
        if input_path.exists():
            for r1 in sorted(input_path.glob("*_R1.fastq.gz")):
                basename = r1.name[:-len("_R1.fastq.gz")]
                species_expected = _guess_species_from_basename(basename)
                library_id = _guess_library_from_basename(basename)
                file_size = r1.stat().st_size
                manifest_rows.append({
                    "raw_file_id": basename,
                    "file_path_or_uri": str(r1),
                    "library_id": library_id,
                    "species_expected": species_expected,
                    "read_count": "",     # leave blank — agent computes if needed
                    "file_size_bytes": file_size,
                    "sha256": _sha256_head(r1, 1 << 20),  # head-bytes hash for speed
                    "included_in_run": True,
                    "exclusion_reason": "",
                })
        write_raw_data_manifest(sys_dir, manifest_rows)

        # Load + concatenate every per-species CSV the agent emitted
        frames: list[pd.DataFrame] = []
        for csv_path in candidate_csvs:
            try:
                frames.append(pd.read_csv(csv_path))
            except Exception as exc:
                ip.error(f"[harness] failed to read {csv_path}: {exc}")
        if not frames:
            return None
        raw_concat = pd.concat(frames, ignore_index=True)

        # Lift to canonical schema. write_processed_examples rejects any
        # gold-derived columns — that's the load-bearing leakage check.
        adapter = CandidateAdapter(run_id=result.run_id, system_name=spec.name)
        canonical = adapter.adapt(raw_concat)
        write_processed_examples(sys_dir, canonical)
        ip.emit(f"[harness] wrote {spec.name} candidate dataset ({len(canonical)} rows)")

        return SystemRun(
            name=spec.name,
            setting=spec.setting,
            run_dir=sys_dir,
            runtime_seconds=elapsed,
            cost_usd=actual_cost,
            raw_reads_processed=raw_reads,
            reads_retained_after_qc=retained_reads,
            failure_count=0 if result.state == "completed" else 1,
            total_attempts=1,
        )

    def _write_readme(
        self,
        eval_root: Path,
        system_runs: list[SystemRun],
        harness_metadata: dict,
        eval_status: str,
    ) -> Path:
        lines: list[str] = []
        lines.append("# Leakage-Isolated External Evaluation")
        lines.append("")
        lines.append(f"- **Eval run ID:** `{harness_metadata['eval_run_id']}`")
        lines.append(f"- **Started:** {harness_metadata['started_at_utc']}")
        lines.append(f"- **Gold dataset:** `{harness_metadata.get('gold_dataset_name')}`")
        lines.append(f"- **Final state:** `{eval_status}`")
        lines.append("")
        lines.append("## Systems")
        lines.append("")
        for s in harness_metadata["systems"]:
            lines.append(f"- **{s['name']}** ({s['kind']}, {s['setting']}, model=`{s['model_name']}`)")
        lines.append("")
        lines.append("## Leakage controls")
        lines.append("")
        for ctrl in harness_metadata["leakage_controls"]:
            lines.append(f"- {ctrl}")
        lines.append("")
        lines.append("## Files in this package")
        lines.append("")
        lines.append("| File | Format | Purpose |")
        lines.append("| --- | --- | --- |")
        lines.append("| `evaluation_config.yaml` | YAML | Frozen pre-registered evaluation contract |")
        lines.append("| `run_metadata.json` | JSON | Harness-level run provenance |")
        lines.append("| `systems/<name>/processed_examples.parquet` | Parquet | One system's frozen candidate dataset |")
        lines.append("| `systems/<name>/processed_examples_sample.csv` | CSV | Human-readable sample of the same |")
        lines.append("| `systems/<name>/raw_data_manifest.csv` | CSV | Per-FASTQ manifest with file checksums |")
        lines.append("| `systems/<name>/artifact_hashes.sha256` | SHA256 | Freeze line — gold loads only after this |")
        lines.append("| `evaluation/summary_metrics.csv` | CSV | Headline results (macro + micro) |")
        lines.append("| `evaluation/per_species_metrics.csv` | CSV | Species-stratified results |")
        lines.append("| `evaluation/efficiency_metrics.csv` | CSV | Runtime, cost, throughput, compression |")
        lines.append("| `evaluation/error_analysis.csv` | CSV | Error taxonomy counts + rates |")
        lines.append("| `evaluation/statistical_tests.csv` | CSV | Bootstrap CIs + McNemar p-values |")
        lines.append("| `evaluation/gold_alignment.parquet` | Parquet | Full row-level alignment with gold |")
        lines.append("| `evaluation/gold_alignment_review.csv` | CSV | Human-readable alignment review |")
        lines.append("")
        lines.append("## Reading the result")
        lines.append("")
        lines.append("The headline metric is **macro-averaged annotation exact match** across species, "
                     "computed under a budget-matched comparison. See `evaluation/summary_metrics.csv` "
                     "for the headline table and `evaluation/per_species_metrics.csv` for species-level "
                     "breakdowns. Statistical comparisons (paired bootstrap CIs on CAPO − baseline plus "
                     "McNemar tests on paired annotation correctness) live in "
                     "`evaluation/statistical_tests.csv`.")
        lines.append("")
        lines.append("**Important evaluation caveat.** Each species is sourced from two yeast-display "
                     "libraries, so the cleaned candidate dataset is expected to contain many more "
                     "sequences than the gold set. The evaluation does NOT force every cleaned sequence "
                     "to align — it measures (a) how much of the gold set is recovered "
                     "(`gold_coverage`), (b) whether the recovered labels match "
                     "(`annotation_exact_match`), and (c) the size of the full generated candidate space "
                     "alongside (`candidate_dataset_size` in `efficiency_metrics.csv`).")
        path = eval_root / "README.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Small filename + path helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()).strip("_") or "system"


def _guess_species_from_basename(basename: str) -> str:
    """Best-effort species extraction from a FASTQ basename for the manifest.

    Conservative: returns "" if nothing recognisable is found.
    """
    bn = basename.lower()
    for cand in (
        "human_new", "humannew", "ihbat", "human", "mouse", "rat", "dog", "cat",
        "cattle", "cow", "bovine", "horse", "equine", "monkey", "macaque",
        "mink", "bat", "pangolin", "hamster",
    ):
        if cand in bn:
            return cand
    return ""


def _guess_library_from_basename(basename: str) -> str:
    bn = basename.lower()
    if "lib1" in bn:
        return "lib1"
    if "lib2" in bn:
        return "lib2"
    return ""


def _sha256_head(path: Path, n_bytes: int) -> str:
    """SHA256 of the first n_bytes of a file.

    A full hash of multi-GB FASTQs is too slow for the harness's quick-look
    manifest; a head-hash is enough to detect accidental input swaps (since
    the gzip header + first reads are file-unique). The full file is hashed
    again at freeze time only for the canonical Stage-1 artifacts that fit
    in memory.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()
