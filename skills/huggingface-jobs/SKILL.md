---
name: huggingface-jobs
description: This skill should be used when users want to run any workload on Hugging Face Jobs infrastructure. Covers UV scripts, Docker-based jobs, hardware selection, cost estimation, authentication with tokens, secrets management, timeout configuration, and result persistence. Designed for protein language model fine-tuning workloads including data preprocessing, sequence tokenization, supervised training, experiments, batch jobs, and any Python-based tasks. Should be invoked for tasks involving cloud compute, GPU workloads, or when users mention running protein model training or fine-tuning jobs on Hugging Face infrastructure without local setup.
license: Complete terms in LICENSE.txt
---

# Running Workloads on Hugging Face Jobs

## Overview

Run any workload on fully managed Hugging Face infrastructure. No local setup required—jobs run on cloud CPUs, GPUs, or TPUs and can persist results to the Hugging Face Hub.

**Common use cases:**
- **Protein Data Processing** - Transform, filter, tokenize, or analyze sequence datasets
- **Batch Inference** - Run inference on thousands of protein sequences
- **Experiments & Benchmarks** - Reproducible protein ML experiments
- **Model Training** - Fine-tune protein language models
- **Synthetic Data Generation** - Generate or augment biological datasets
- **Development & Testing** - Test code without local GPU setup
- **Scheduled Jobs** - Automate recurring tasks

**For model training specifically:** Use this skill for protein language model fine-tuning workflows on Hugging Face Jobs.

## When to Use This Skill

Use this skill when users want to:
- Run Python workloads on cloud infrastructure
- Execute jobs without local GPU/TPU setup
- Process protein or biological sequence data at scale
- Run batch inference or experiments on protein models
- Fine-tune protein language models
- Schedule recurring tasks
- Use GPUs/TPUs for any workload
- Persist results to the Hugging Face Hub

## Key Directives

When assisting with jobs:

1. **ALWAYS use `hf_jobs()` MCP tool** - Submit jobs using `hf_jobs("uv", {...})` or `hf_jobs("run", {...})`. The `script` parameter accepts Python code directly. Do NOT save to local files unless the user explicitly requests it. Pass the script content as a string to `hf_jobs()`.

2. **Always handle authentication** - Jobs that interact with the Hub require `HF_TOKEN` via secrets. See Token Usage section below.

3. **Provide job details after submission** - After submitting, provide job ID, monitoring URL, estimated time, and note that the user can request status checks later.

4. **Set appropriate timeouts** - Default 30min may be insufficient for long-running fine-tuning tasks.

## Prerequisites Checklist

Before starting any job, verify:

### ✅ **Account & Authentication**
- Hugging Face Account with [Pro](https://hf.co/pro), [Team](https://hf.co/enterprise), or [Enterprise](https://hf.co/enterprise) plan (Jobs require paid plan)
- Authenticated login: Check with `hf_whoami()`
- **HF_TOKEN for Hub Access** ⚠️ CRITICAL - Required for any Hub operations (push models/datasets, download private repos, etc.)
- Token must have appropriate permissions (read for downloads, write for uploads)

### ✅ **Token Usage** (See Token Usage section for details)

**When tokens are required:**
- Pushing models/datasets to Hub
- Accessing private repositories
- Using Hub APIs in scripts
- Any authenticated Hub operations

**How to provide tokens:**
```python
# hf_jobs MCP tool — $HF_TOKEN is auto-replaced with real token:
{"secrets": {"HF_TOKEN": "$HF_TOKEN"}}

# HfApi().run_uv_job() — MUST pass actual token:
from huggingface_hub import get_token
secrets={"HF_TOKEN": get_token()}
````

**⚠️ CRITICAL:** The `$HF_TOKEN` placeholder is ONLY auto-replaced by the `hf_jobs` MCP tool. When using `HfApi().run_uv_job()`, you MUST pass the real token via `get_token()`. Passing the literal string `"$HF_TOKEN"` results in a 9-character invalid token and 401 errors.

## Token Usage Guide

### Understanding Tokens

**What are HF Tokens?**

* Authentication credentials for Hugging Face Hub
* Required for authenticated operations (push, private repos, API access)
* Stored securely on your machine after `hf auth login`

**Token Types:**

* **Read Token** - Can download models/datasets, read private repos
* **Write Token** - Can push models/datasets, create repos, modify content
* **Organization Token** - Can act on behalf of an organization

### When Tokens Are Required

**Always Required:**

* Pushing models/datasets to Hub
* Accessing private repositories
* Creating new repositories
* Modifying existing repositories
* Using Hub APIs programmatically

**Not Required:**

* Downloading public models/datasets
* Running jobs that don't interact with Hub
* Reading public repository information

### How to Provide Tokens to Jobs

#### Method 1: Automatic Token (Recommended)

```python
hf_jobs("uv", {
    "script": "your_script.py",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}  # ✅ Automatic replacement
})
```

**How it works:**

* `$HF_TOKEN` is a placeholder that gets replaced with your actual token
* Uses the token from your logged-in session (`hf auth login`)
* Most secure and convenient method
* Token is encrypted server-side when passed as a secret

**Benefits:**

* No token exposure in code
* Uses your current login session
* Automatically updated if you re-login
* Works seamlessly with MCP tools

#### Method 2: Explicit Token (Not Recommended)

```python
hf_jobs("uv", {
    "script": "your_script.py",
    "secrets": {"HF_TOKEN": "hf_abc123..."}  # ⚠️ Hardcoded token
})
```

**When to use:**

* Only if automatic token doesn't work
* Testing with a specific token
* Organization tokens (use with caution)

**Security concerns:**

* Token visible in code/logs
* Must manually update if token rotates
* Risk of token exposure

#### Method 3: Environment Variable (Less Secure)

```python
hf_jobs("uv", {
    "script": "your_script.py",
    "env": {"HF_TOKEN": "hf_abc123..."}  # ⚠️ Less secure than secrets
})
```

**Difference from secrets:**

* `env` variables are visible in job logs
* `secrets` are encrypted server-side
* Always prefer `secrets` for tokens

### Using Tokens in Scripts

**In your Python script, tokens are available as environment variables:**

```python
# /// script
# dependencies = ["huggingface-hub"]
# ///

import os
from huggingface_hub import HfApi

# Token is automatically available if passed via secrets
token = os.environ.get("HF_TOKEN")

# Use with Hub API
api = HfApi(token=token)

# Or let huggingface_hub auto-detect
api = HfApi()  # Automatically uses HF_TOKEN env var
```

**Best practices:**

* Don't hardcode tokens in scripts
* Use `os.environ.get("HF_TOKEN")` to access
* Let `huggingface_hub` auto-detect when possible
* Verify token exists before Hub operations

### Token Verification

**Check if you're logged in:**

```python
from huggingface_hub import whoami
user_info = whoami()  # Returns your username if authenticated
```

**Verify token in job:**

```python
import os
assert "HF_TOKEN" in os.environ, "HF_TOKEN not found!"
token = os.environ["HF_TOKEN"]
print(f"Token starts with: {token[:7]}...")  # Should start with "hf_"
```

### Common Token Issues

**Error: 401 Unauthorized**

* **Cause:** Token missing or invalid
* **Fix:** Add `secrets={"HF_TOKEN": "$HF_TOKEN"}` to job config
* **Verify:** Check `hf_whoami()` works locally

**Error: 403 Forbidden**

* **Cause:** Token lacks required permissions
* **Fix:** Ensure token has write permissions for push operations
* **Check:** Token type at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

**Error: Token not found in environment**

* **Cause:** `secrets` not passed or wrong key name
* **Fix:** Use `secrets={"HF_TOKEN": "$HF_TOKEN"}` (not `env`)
* **Verify:** Script checks `os.environ.get("HF_TOKEN")`

**Error: Repository access denied**

* **Cause:** Token doesn't have access to private repo
* **Fix:** Use token from account with access
* **Check:** Verify repo visibility and your permissions

### Token Security Best Practices

1. **Never commit tokens** - Use `$HF_TOKEN` placeholder or environment variables
2. **Use secrets, not env** - Secrets are encrypted server-side
3. **Rotate tokens regularly** - Generate new tokens periodically
4. **Use minimal permissions** - Create tokens with only needed permissions
5. **Don't share tokens** - Each user should use their own token
6. **Monitor token usage** - Check token activity in Hub settings

### Complete Token Example

```python
# Example: Push results to Hub
hf_jobs("uv", {
    "script": """
# /// script
# dependencies = ["huggingface-hub", "datasets"]
# ///

import os
from huggingface_hub import HfApi
from datasets import Dataset

# Verify token is available
assert "HF_TOKEN" in os.environ, "HF_TOKEN required!"

# Use token for Hub operations
api = HfApi(token=os.environ["HF_TOKEN"])

# Create and push dataset
data = {"text": ["Hello", "World"]}
dataset = Dataset.from_dict(data)
dataset.push_to_hub("username/my-dataset", token=os.environ["HF_TOKEN"])

print("✅ Dataset pushed successfully!")
""",
    "flavor": "cpu-basic",
    "timeout": "30m",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}  # ✅ Token provided securely
})
```

## Quick Start: Two Approaches

### Approach 1: UV Scripts (Recommended)

UV scripts use PEP 723 inline dependencies for clean, self-contained workloads.

**MCP Tool:**

```python
hf_jobs("uv", {
    "script": """
# /// script
# dependencies = ["transformers", "torch", "datasets"]
# ///

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Your protein fine-tuning workload here
dataset = load_dataset("your/protein-dataset")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = AutoModelForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D", num_labels=2)
""",
    "flavor": "a10g-small",
    "timeout": "2h"
})
```

**CLI Equivalent:**

```bash
hf jobs uv run my_script.py --flavor a10g-small --timeout 2h
```

**Python API:**

```python
from huggingface_hub import run_uv_job
run_uv_job("my_script.py", flavor="a10g-small", timeout="2h")
```

**Benefits:** Direct MCP tool usage, clean code, dependencies declared inline, no file saving required

**When to use:** Default choice for protein language model fine-tuning, sequence preprocessing, training, evaluation, and custom training logic requiring `hf_jobs()`

#### Custom Docker Images for UV Scripts

By default, UV scripts use `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`. For ML workloads with complex dependencies, use pre-built images:

```python
hf_jobs("uv", {
    "script": "train_protein_model.py",
    "image": "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    "flavor": "a10g-large"
})
```

**CLI:**

```bash
hf jobs uv run --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel --flavor a10g-large train_protein_model.py
```

**Benefits:** Faster startup, pre-installed dependencies, optimized for training frameworks

#### Python Version

By default, UV scripts use Python 3.12. Specify a different version:

```python
hf_jobs("uv", {
    "script": "my_script.py",
    "python": "3.11",  # Use Python 3.11
    "flavor": "cpu-basic"
})
```

**Python API:**

```python
from huggingface_hub import run_uv_job
run_uv_job("my_script.py", python="3.11")
```

#### Working with Scripts

⚠️ **Important:** There are *two* "script path" stories depending on how you run Jobs:

* **Using the `hf_jobs()` MCP tool (recommended in this repo)**: the `script` value must be **inline code** (a string) or a **URL**. A local filesystem path (like `"./scripts/foo.py"`) won't exist inside the remote container.
* **Using the `hf jobs uv run` CLI**: local file paths **do work** (the CLI uploads your script).

**Common mistake with `hf_jobs()` MCP tool:**

```python
# ❌ Will fail (remote container can't see your local path)
hf_jobs("uv", {"script": "./scripts/foo.py"})
```

**Correct patterns with `hf_jobs()` MCP tool:**

```python
# ✅ Inline: read the local script file and pass its *contents*
from pathlib import Path
script = Path("huggingface-jobs/scripts/foo.py").read_text()
hf_jobs("uv", {"script": script})

# ✅ URL: host the script somewhere reachable
hf_jobs("uv", {"script": "https://huggingface.co/datasets/uv-scripts/.../raw/main/foo.py"})

# ✅ URL from GitHub
hf_jobs("uv", {"script": "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py"})
```

**CLI equivalent (local paths supported):**

```bash
hf jobs uv run ./scripts/foo.py -- --your --args
```

#### Adding Dependencies at Runtime

Add extra dependencies beyond what's in the PEP 723 header:

```python
hf_jobs("uv", {
    "script": "train_protein_model.py",
    "dependencies": ["transformers", "torch>=2.0", "datasets", "evaluate"],
    "flavor": "a10g-small"
})
```

**Python API:**

```python
from huggingface_hub import run_uv_job
run_uv_job("train_protein_model.py", dependencies=["transformers", "torch>=2.0", "datasets", "evaluate"])
```

### Approach 2: Docker-Based Jobs

Run jobs with custom Docker images and commands.

**MCP Tool:**

```python
hf_jobs("run", {
    "image": "python:3.12",
    "command": ["python", "-c", "print('Hello from HF Jobs!')"],
    "flavor": "cpu-basic",
    "timeout": "30m"
})
```

**CLI Equivalent:**

```bash
hf jobs run python:3.12 python -c "print('Hello from HF Jobs!')"
```

**Python API:**

```python
from huggingface_hub import run_job
run_job(image="python:3.12", command=["python", "-c", "print('Hello!')"], flavor="cpu-basic")
```

**Benefits:** Full Docker control, use pre-built images, run any command
**When to use:** Need specific Docker images, non-standard training environments, custom CUDA stacks, complex environments

**Example with GPU:**

```python
hf_jobs("run", {
    "image": "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    "command": ["python", "-c", "import torch; print(torch.cuda.get_device_name())"],
    "flavor": "a10g-small",
    "timeout": "1h"
})
```

**Using Hugging Face Spaces as Images:**

You can use Docker images from HF Spaces:

```python
hf_jobs("run", {
    "image": "hf.co/spaces/lhoestq/duckdb",
    "command": ["duckdb", "-c", "SELECT 'Hello from DuckDB!'"],
    "flavor": "cpu-basic"
})
```

**CLI:**

```bash
hf jobs run hf.co/spaces/lhoestq/duckdb duckdb -c "SELECT 'Hello!'"
```

## Hardware Selection

> **Reference:** [HF Jobs Hardware Docs](https://huggingface.co/docs/hub/en/spaces-config-reference) (updated 07/2025)

| Workload Type                     | Recommended Hardware                   | Use Case                                |
| --------------------------------- | -------------------------------------- | --------------------------------------- |
| Protein data processing, testing  | `cpu-basic`, `cpu-upgrade`             | Lightweight tasks                       |
| Small protein models, quick tests | `t4-small`                             | Small ESM / lightweight sequence models |
| Medium protein models             | `t4-medium`, `l4x1`                    | Mid-size protein LM fine-tuning         |
| Large protein models              | `a10g-small`, `a10g-large`             | Full fine-tuning, larger batch sizes    |
| Very large protein models         | `a100-large`                           | Large PLMs, long sequences              |
| Batch inference                   | `a10g-large`, `a100-large`             | High-throughput sequence scoring        |
| Multi-GPU workloads               | `l4x4`, `a10g-largex2`, `a10g-largex4` | Parallel training or large PLMs         |
| TPU workloads                     | `v5e-1x1`, `v5e-2x2`, `v5e-2x4`        | JAX/Flax, TPU-optimized                 |

**All Available Flavors:**

* **CPU:** `cpu-basic`, `cpu-upgrade`
* **GPU:** `t4-small`, `t4-medium`, `l4x1`, `l4x4`, `a10g-small`, `a10g-large`, `a10g-largex2`, `a10g-largex4`, `a100-large`
* **TPU:** `v5e-1x1`, `v5e-2x2`, `v5e-2x4`

**Guidelines:**

* Start with smaller hardware for testing
* Scale up based on model size, sequence length, and batch size
* Use multi-GPU for larger protein models or long-sequence workloads
* Use TPUs for JAX/Flax workloads
* See `references/hardware_guide.md` for detailed specifications

## Critical: Saving Results

**⚠️ EPHEMERAL ENVIRONMENT—MUST PERSIST RESULTS**

The Jobs environment is temporary. All files are deleted when the job ends. If results aren't persisted, **ALL WORK IS LOST**.

### Persistence Options

**1. Push to Hugging Face Hub (Recommended)**

```python
# Push models
model.push_to_hub("username/protein-model-name", token=os.environ["HF_TOKEN"])

# Push datasets
dataset.push_to_hub("username/protein-dataset-name", token=os.environ["HF_TOKEN"])

# Push artifacts
api.upload_file(
    path_or_fileobj="results.json",
    path_in_repo="results.json",
    repo_id="username/results",
    token=os.environ["HF_TOKEN"]
)
```

**2. Use External Storage**

```python
# Upload to S3, GCS, etc.
import boto3
s3 = boto3.client('s3')
s3.upload_file('results.json', 'my-bucket', 'results.json')
```

**3. Send Results via API**

```python
# POST results to your API
import requests
requests.post("https://your-api.com/results", json=results)
```

### Required Configuration for Hub Push

**In job submission:**

```python
# hf_jobs MCP tool:
{"secrets": {"HF_TOKEN": "$HF_TOKEN"}}

# HfApi().run_uv_job():
from huggingface_hub import get_token
secrets={"HF_TOKEN": get_token()}
```

**In script:**

```python
import os
from huggingface_hub import HfApi

# Token automatically available from secrets
api = HfApi(token=os.environ.get("HF_TOKEN"))

# Push your results
api.upload_file(...)
```

### Verification Checklist

Before submitting:

* [ ] Results persistence method chosen
* [ ] Token in secrets if using Hub (MCP: `"$HF_TOKEN"`, Python API: `get_token()`)
* [ ] Script handles missing token gracefully
* [ ] Test persistence path works

**See:** `references/hub_saving.md` for detailed Hub persistence guide

## Timeout Management

**⚠️ DEFAULT: 30 MINUTES**

Jobs automatically stop after the timeout. For long-running tasks like protein model training, always set a custom timeout.

### Setting Timeouts

**MCP Tool:**

```python
{
    "timeout": "2h"
}
```

**Supported formats:**

* Integer/float: seconds (e.g., `300` = 5 minutes)
* String with suffix: `"5m"` (minutes), `"2h"` (hours), `"1d"` (days)
* Examples: `"90m"`, `"2h"`, `"1.5h"`, `300`, `"1d"`

**Python API:**

```python
from huggingface_hub import run_job, run_uv_job

run_job(image="python:3.12", command=[...], timeout="2h")
run_uv_job("script.py", timeout=7200)  # 2 hours in seconds
```

### Timeout Guidelines

| Scenario                | Recommended | Notes                         |
| ----------------------- | ----------- | ----------------------------- |
| Quick test              | 10-30 min   | Verify setup                  |
| Protein data processing | 1-2 hours   | Depends on dataset size       |
| Batch inference         | 2-4 hours   | Large sequence batches        |
| Fine-tuning experiments | 4-8 hours   | Multiple epochs / evaluations |
| Long-running training   | 8-24 hours  | Large PLMs or long sequences  |

**Always add 20-30% buffer** for setup, network delays, checkpointing, and cleanup.

**On timeout:** Job killed immediately, all unsaved progress lost

## Cost Estimation

For detailed runtime pricing guidance, hardware cost tradeoffs, and general cost optimization heuristics, see `cost-estimation`.

## Monitoring and Tracking

### Check Job Status

**MCP Tool:**

```python
# List all jobs
hf_jobs("ps")

# Inspect specific job
hf_jobs("inspect", {"job_id": "your-job-id"})

# View logs
hf_jobs("logs", {"job_id": "your-job-id"})

# Cancel a job
hf_jobs("cancel", {"job_id": "your-job-id"})
```

**Python API:**

```python
from huggingface_hub import list_jobs, inspect_job, fetch_job_logs, cancel_job

# List your jobs
jobs = list_jobs()

# List running jobs only
running = [j for j in list_jobs() if j.status.stage == "RUNNING"]

# Inspect specific job
job_info = inspect_job(job_id="your-job-id")

# View logs
for log in fetch_job_logs(job_id="your-job-id"):
    print(log)

# Cancel a job
cancel_job(job_id="your-job-id")
```

**CLI:**

```bash
hf jobs ps
hf jobs logs <job-id>
hf jobs cancel <job-id>
```

**Remember:** Wait for user to request status checks. Avoid polling repeatedly.

### Job URLs

After submission, jobs have monitoring URLs:

```
https://huggingface.co/jobs/username/job-id
```

View logs, status, and details in the browser.

## Common Workload Patterns

This repository ships ready-to-run UV scripts in `huggingface-jobs/scripts/`. Prefer using them instead of inventing new templates.

### Pattern 1: Protein Dataset Preprocessing

**What it does:** loads a protein dataset, validates sequences, tokenizes or reformats fields, and pushes the processed dataset back to the Hub.

**Requires:** CPU or GPU depending on preprocessing + **write** token if it pushes a dataset.

```python
from pathlib import Path

script = Path("huggingface-jobs/scripts/preprocess-protein-dataset.py").read_text()
hf_jobs("uv", {
    "script": script,
    "script_args": [
        "--input-dataset", "username/raw-protein-dataset",
        "--output-dataset", "username/processed-protein-dataset",
        "--sequence-column", "sequence",
        "--label-column", "label",
    ],
    "flavor": "cpu-upgrade",
    "timeout": "2h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"},
})
```

### Pattern 2: Protein Language Model Fine-Tuning

**What it does:** loads a protein language model and labeled sequence dataset, tokenizes sequences, fine-tunes the model, evaluates checkpoints, and pushes the trained model back to the Hub.

**Requires:** GPU + **write** token if it pushes a model.

```python
from pathlib import Path

script = Path("huggingface-jobs/scripts/finetune-protein-lm.py").read_text()
hf_jobs("uv", {
    "script": script,
    "script_args": [
        "--model-id", "facebook/esm2_t6_8M_UR50D",
        "--dataset-id", "username/protein-binding-dataset",
        "--sequence-column", "sequence",
        "--label-column", "label",
        "--output-repo", "username/esm2-binding-finetuned",
    ],
    "flavor": "a10g-large",
    "timeout": "8h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"},
})
```

### Pattern 3: Protein Embedding or Batch Inference

**What it does:** runs a pretrained protein language model over many sequences to generate embeddings, scores, or predictions, then saves the outputs.

**Requires:** GPU for large-scale inference; token needed **only** if uploading outputs.

```python
from pathlib import Path

script = Path("huggingface-jobs/scripts/protein-batch-inference.py").read_text()
hf_jobs("uv", {
    "script": script,
    "script_args": [
        "--dataset-id", "username/protein-dataset",
        "--model-id", "facebook/esm2_t6_8M_UR50D",
        "--output-repo", "username/protein-inference-results",
    ],
    "flavor": "a10g-small",
    "timeout": "4h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"},
})
```

## Common Failure Modes

### Out of Memory (OOM)

**Fix:**

1. Reduce batch size
2. Reduce max sequence length
3. Use gradient accumulation
4. Upgrade hardware: t4 → a10g → a100

### Job Timeout

**Fix:**

1. Check logs for actual runtime
2. Increase timeout with buffer: `"timeout": "8h"`
3. Optimize code for faster execution
4. Save checkpoints regularly

### Hub Push Failures

**Fix:**

1. Add token to secrets: MCP uses `"$HF_TOKEN"` (auto-replaced), Python API uses `get_token()` (must pass real token)
2. Verify token in script: `assert "HF_TOKEN" in os.environ`
3. Check token permissions
4. Verify repo exists or can be created

### Missing Dependencies

**Fix:**
Add to PEP 723 header:

```python
# /// script
# dependencies = ["transformers", "datasets", "torch", "evaluate"]
# ///
```

### Authentication Errors

**Fix:**

1. Check `hf_whoami()` works locally
2. Verify token in secrets — MCP: `"$HF_TOKEN"`, Python API: `get_token()` (NOT `"$HF_TOKEN"`)
3. Re-login: `hf auth login`
4. Check token has required permissions

## Troubleshooting

**Common issues:**

* Job times out → Increase timeout, optimize code
* Results not saved → Check persistence method, verify HF_TOKEN
* Out of Memory → Reduce batch size or sequence length, upgrade hardware
* Import errors → Add dependencies to PEP 723 header
* Authentication errors → Check token, verify secrets parameter

**See:** `references/troubleshooting.md` for complete troubleshooting guide

## Resources

### References (In This Skill)

* `references/token_usage.md` - Complete token usage guide
* `references/hardware_guide.md` - Hardware specs and selection
* `references/hub_saving.md` - Hub persistence guide
* `references/troubleshooting.md` - Common issues and solutions

### Scripts (In This Skill)

* `scripts/preprocess-protein-dataset.py` - Protein dataset cleaning, validation, and formatting
* `scripts/finetune-protein-lm.py` - Protein language model fine-tuning and Hub push
* `scripts/protein-batch-inference.py` - Protein embeddings / scoring / prediction over Hub datasets

## Key Takeaways

1. **Submit scripts inline** - The `script` parameter accepts Python code directly; no file saving required unless user requests
2. **Jobs are asynchronous** - Don't wait/poll; let user check when ready
3. **Always set timeout** - Default 30 min may be insufficient; set appropriate timeout
4. **Always persist results** - Environment is ephemeral; without persistence, all work is lost
5. **Use tokens securely** - MCP: `secrets={"HF_TOKEN": "$HF_TOKEN"}`, Python API: `secrets={"HF_TOKEN": get_token()}` — `"$HF_TOKEN"` only works with MCP tool
6. **Choose appropriate hardware** - Start small, scale up based on protein model size and sequence length
7. **Use UV scripts** - Default to `hf_jobs("uv", {...})` with inline scripts for Python workloads
8. **Handle authentication** - Verify tokens are available before Hub operations
9. **Monitor jobs** - Provide job URLs and status check commands
10. **Optimize costs** - Choose right hardware, set appropriate timeouts

## Quick Reference: MCP Tool vs CLI vs Python API

| Operation        | MCP Tool                             | CLI                                           | Python API                  |
| ---------------- | ------------------------------------ | --------------------------------------------- | --------------------------- |
| Run UV script    | `hf_jobs("uv", {...})`               | `hf jobs uv run script.py`                    | `run_uv_job("script.py")`   |
| Run Docker job   | `hf_jobs("run", {...})`              | `hf jobs run image cmd`                       | `run_job(image, command)`   |
| List jobs        | `hf_jobs("ps")`                      | `hf jobs ps`                                  | `list_jobs()`               |
| View logs        | `hf_jobs("logs", {...})`             | `hf jobs logs <id>`                           | `fetch_job_logs(job_id)`    |
| Cancel job       | `hf_jobs("cancel", {...})`           | `hf jobs cancel <id>`                         | `cancel_job(job_id)`        |
| Schedule UV      | `hf_jobs("scheduled uv", {...})`     | `hf jobs scheduled uv run SCHEDULE script.py` | `create_scheduled_uv_job()` |
| Schedule Docker  | `hf_jobs("scheduled run", {...})`    | `hf jobs scheduled run SCHEDULE image cmd`    | `create_scheduled_job()`    |
| List scheduled   | `hf_jobs("scheduled ps")`            | `hf jobs scheduled ps`                        | `list_scheduled_jobs()`     |
| Delete scheduled | `hf_jobs("scheduled delete", {...})` | `hf jobs scheduled delete <id>`               | `delete_scheduled_job()`    |
