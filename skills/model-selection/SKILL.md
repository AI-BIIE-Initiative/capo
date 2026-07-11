---
name: model-selection
description: >
  Select the best protein language model (PLM) and fine-tuning strategy based on task, compute,
  openness and label availability. Covers representation models, generative/diffusion models,
  structure predictors, inverse folding, fitness scoring, nucleotide LMs, and biomedical NLP.
  Use when the user asks to embed, score, generate, fine-tune, fold, design binders, or predict
  variant effects. Route to structure tools (boltz) only if the user wants structure prediction
  without sequence design.
license: MIT
user-invokable: true
compatibility: >
  Registry: hf.co/datasets/BIIE-AI/protein-model-registry (private, 30 models)
  Local source: model_registry/registry_src/models.jsonl
  Policy: model_registry/registry_src/selection_policy.yaml
  Requires HF_TOKEN to load registry programmatically; routing logic works offline.
---

# Model Selection

> **Registry:** [`BIIE-AI/protein-model-registry`](https://huggingface.co/datasets/BIIE-AI/protein-model-registry) (private HF dataset repo)
> **Local source of truth:** `model_registry/registry_src/models.jsonl`
> **Routing policy:** `model_registry/registry_src/selection_policy.yaml`
> **Enum definitions:** `model_registry/references/taxonomy.md` — read if you need to interpret `category`, `architecture_type`, `compute_tier`, `commercial_use`, `open_weights`, or any other categorical field
> **Fine-tuning code (ESM family):** `esm/SKILL.md`

---

## What this returns

Three lanes — always:

```
Best fit:   <registry_id>  (<hf_repo_id>)
            Why: <one sentence>
            Tradeoff: <one sentence>

Budget:     <registry_id>  (<hf_repo_id>)
            Why: <one sentence>

Frontier:   <registry_id>  (<hf_repo_id>)
            Why: <one sentence>

Rejected (Optional):
- <registry_id>: <reason>
```

---

## Step 1 — Extract constraints from the request

| Field | What to infer |
|---|---|
| `task_type` | primary task from the task enum in selection_policy.yaml routing |
| `compute_budget` | CPU / low-mem GPU / mid-mem GPU / high-mem GPU / multi-GPU / API |
| `label_count` | none / <500 / 500–10k / 10k–100k / >100k |
| `requires_open_weights` | explicit or implied by "local", "no API" |
| `commercial_use` | explicit or implied by production / deployment context |
| `input_modality` | sequence only / structure available / function available / nucleotide / text |

If not stated, assume: `compute_budget = single_gpu_midmem`, `commercial_use = unknown`.

---

## Step 2 — Hard filter

Exclude any model that fails a required constraint:

- `task_type` not in model `primary_tasks` or `secondary_tasks` → **exclude**
- `compute_tier` > compute_budget → **exclude** (keep if no other candidate survives)
- `gated = true` and user requires frictionless access → **penalize heavily**
- `commercial_use = non_commercial_only | research_only` and commercial deploy implied → **exclude**
- `category = nucleotide_language_model | biomedical_text_model` for protein tasks → **exclude** (and vice versa)

---

## Step 3 — Route by task

Load the ordered candidate list from `selection_policy.yaml → routing[task_type]`. The policy embeds preferred ordering; apply the compute filter from Step 2 on top.

Task → default routing (abbreviated; full list in policy):

| Task | Preferred candidates |
|---|---|
| `embedding` | esmc_300m → esm2_t33_650m → saprot_650m → ankh3_large |
| `variant_effect_prediction` | esm1v_650m → tranception → saprot_650m |
| `sequence_generation` | protgpt2 → dplm_650m → esm3_sm_open |
| `controlled_generation` | esm3_sm_open → dplm2_150m → nv_la_proteina |
| `folding` | esmfold_v1 → boltz_2 |
| `inverse_folding` | esm_if1 → dplm_650m → dplm2_150m |
| `binder_design` | boltzgen_1 → nv_la_proteina → boltz_2 |
| `motif_scaffolding` | dplm_650m → dplm2_150m → nv_la_proteina |
| `complex_prediction` / `binding_affinity_prediction` | boltz_2 |
| `nucleotide_representation` | nucleotide_transformer_v2_500m |
| `biomedical_nlp` | biomedbert_base → biobert_base |

---

## Step 4 — Rank and apply penalties

Score = `0.40 × task_fit + 0.20 × modality_fit + 0.15 × compute_fit + 0.10 × openness_fit + 0.10 × maturity + 0.05 × deployment`

Penalties (from policy):

| Condition | Penalty |
|---|---|
| Gated + user wants frictionless | −0.20 |
| Over compute budget | −0.25 |
| Commercial-use unknown when commercial implied | −0.15 |
| Legacy model or prefer_*_over_this tag | −0.10 |
| Non-commercial license + commercial deploy | −0.30 |

---

## Step 5 — Fine-tuning strategy (when labels present)

| Labels | Strategy |
|---|---|
| 0 | `zero-shot` (PLL / masked-marginal / model inference) |
| < 500 | `linear-probe` (freeze backbone) |
| 500 – 10k | `lora` r=8–16 (default) |
| 10k – 100k | `lora` r=16–32 + optional top-block unfreeze |
| > 100k | `full-finetune` (start from LoRA checkpoint) |

For ESM family implementation details, read `esm/SKILL.md`.

---

## Guardrails

- **Structure prediction only** (no sequence design) → route to `folding` or `complex_prediction`, not `embedding`
- **Nucleotide input** → only `nucleotide_language_model` category; never route to protein models
- **No compute stated** → assume `single_gpu_midmem` (24 GB class); note assumption in output
- **MSA available** → note that msa_transformer (not in main registry) may be worth exploring; warn about MSA pipeline cost
- **ESM3 / ESMC** → non-commercial license; flag explicitly if commercial use implied
- **NV-La-Proteina** → gated, research-only; flag explicitly
