---
title: Protein Model Registry — Canonical Taxonomy
description: Authoritative enum definitions for all categorical fields in models.jsonl
---

# Registry Taxonomy

All enum values used in **Registry:** [`BIIE-AI/protein-model-registry`](https://huggingface.co/datasets/BIIE-AI/protein-model-registry) (primary) and `registry_src/models.jsonl`. Validation is enforced by `schema/model_registry.schema.json`.

---

## `category`

Top-level functional classification of the model.

| Value | Description |
|---|---|
| `protein_representation_model` | Encoder trained for embeddings, MLM, and supervised downstream tasks |
| `protein_sequence_generator` | Generates protein sequences (decoder-only, discrete diffusion, autoregressive) |
| `protein_sequence_structure_generator` | Jointly generates sequence + 3D structure |
| `protein_diffusion_generator` | Diffusion or flow-matching model for protein design (may include structure output) |
| `protein_structure_predictor` | Predicts 3D structure from sequence (folding / complex prediction) |
| `protein_inverse_folding_model` | Designs sequences given a fixed 3D backbone |
| `protein_fitness_model` | Scores variants or predicts fitness / mutation effects |
| `nucleotide_language_model` | Language model over DNA/RNA sequences |
| `biomedical_text_model` | Language model over biomedical literature / clinical text |

---

## `architecture_type`

Mechanism of the model, independent of its category.

| Value | Description |
|---|---|
| `encoder_mlm` | Bidirectional transformer with masked language modeling objective (BERT-style) |
| `encoder_decoder` | Full encoder-decoder transformer (T5-style) |
| `decoder_only` | Autoregressive left-to-right transformer (GPT-style) |
| `discrete_diffusion` | Iterative masked/denoising diffusion over discrete tokens |
| `multimodal_diffusion` | Diffusion over multiple modalities (sequence + structure tokens) |
| `flow_matching` | Continuous or discrete flow matching generative model |
| `structure_prediction` | End-to-end structure prediction (e.g., Evoformer, ESMFold trunk) |
| `inverse_folding` | GVP-Transformer or similar conditioned on 3D coordinates |
| `retrieval_autoregressive` | Autoregressive model augmented with retrieval at inference time |

---

## `input_modalities` / `output_modalities` / `conditioning_modalities`

| Value | Description |
|---|---|
| `protein_sequence` | Amino acid sequence string |
| `protein_structure` | 3D backbone coordinates or structure tokens |
| `protein_function` | GO terms, InterPro IDs, free-text functional keywords |
| `nucleotide_sequence` | DNA or RNA nucleotide sequence |
| `biomedical_text` | Free-form biomedical / clinical text |
| `small_molecule` | Ligand or drug-like small molecule |
| `embeddings` | Dense vector representations |
| `token_logits` | Per-position vocabulary logits |
| `binding_affinity` | Predicted binding affinity (Kd / ΔG) |
| `fitness_score` | Predicted fitness or variant effect score |

---

## `primary_tasks` / `secondary_tasks`

| Value | Description |
|---|---|
| `embedding` | Extract dense sequence representations |
| `masked_reconstruction` | Fill-mask / MLM pretraining objective |
| `sequence_generation` | Generate new amino acid sequences |
| `controlled_generation` | Generate sequences conditioned on structure, function, or motif constraints |
| `folding` | Predict 3D structure from sequence |
| `inverse_folding` | Design sequence for a given 3D backbone |
| `motif_scaffolding` | Build a protein around a fixed functional motif |
| `binder_design` | Design a protein that binds a specified target |
| `fitness_prediction` | Predict absolute or relative fitness of sequences |
| `variant_effect_prediction` | Score the effect of mutations (zero-shot or supervised) |
| `complex_prediction` | Predict multi-chain complex structure |
| `binding_affinity_prediction` | Predict numerical binding affinity |
| `nucleotide_representation` | Extract embeddings from DNA/RNA |
| `biomedical_nlp` | NER, RE, classification over biomedical text |
| `finetuning` | Secondary tag indicating the model supports supervised fine-tuning well |

---

## `size_bucket`

Coarse parameter-count tier with a manual override field (`compute_override_reason`) for models that are operationally heavier than their parameter count suggests (e.g., structure generators).

| Value | Parameter range |
|---|---|
| `tiny` | ≤ 50M |
| `small` | 51M – 300M |
| `medium` | 301M – 800M |
| `large` | 801M – 3B |
| `xlarge` | 3B – 10B |
| `frontier` | > 10B, or structure/all-atom heavy models regardless of param count |

---

## `compute_tier`

Operational compute requirement for inference.

| Value | Description |
|---|---|
| `cpu_ok` | Runs reasonably on CPU |
| `single_gpu_lowmem` | Single GPU ≤ 12 GB |
| `single_gpu_midmem` | Single GPU 12–24 GB |
| `single_gpu_highmem` | Single GPU ≥ 40 GB (A100/H100) |
| `multi_gpu` | Requires multi-GPU or is impractical on a single card |
| `api_only` | No open weights; accessed via external API |

---

## `commercial_use`

| Value | Meaning |
|---|---|
| `yes` | Clearly permissive license (MIT, Apache 2.0) |
| `likely_yes` | Permissive but license edge cases exist |
| `non_commercial_only` | Explicitly restricted to non-commercial use |
| `research_only` | Restricted to academic / research use |
| `unknown` | License not clearly stated or not yet reviewed |

---

## `open_weights`

| Value | Meaning |
|---|---|
| `yes` | Weights downloadable with no restrictions |
| `yes_gated` | Weights available after agreement / token on HF |
| `partial` | Some weights / checkpoints available |
| `no` | API-only; no weights released |
| `unknown` | Not yet determined |

---

## `status` / `maturity`

| Field | Values |
|---|---|
| `status` | `active` \| `legacy` \| `deprecated` |
| `maturity` | `established` (≥2 years, widely benchmarked) \| `recent` (< 2 years) \| `experimental` (preprint / limited evaluation) |
