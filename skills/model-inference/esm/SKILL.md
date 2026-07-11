---
name: esm-inference
description: >
  Use this skill when generating protein embeddings, scoring mutation or variant effects,
  predicting protein structure from sequence, or generating protein sequences with ESM models.
  Covers ESM2 and ESM C embeddings (HuggingFace or native fair-esm), ESM-1v zero-shot
  variant scoring, ESMFold single-chain structure prediction, and ESM3 masked infilling and
  generation. Use even when the user asks broadly for "protein representations" or "variant
  effects" without naming ESM explicitly. Do not use for ESM model selection, LoRA setup,
  or fine-tuning — see model-selection/esm for those tasks.
compatibility: transformers ≥4.30, torch ≥2.0. ESMFold/ESM-1v - fair-esm (pip install fair-esm). ESM3/ESM C - pip install esm.
---

# ESM Inference

For model choice and fine-tuning strategy, see [`model-selection/esm/SKILL.md`](../../model-selection/esm/SKILL.md).

---

## When to use / When NOT to use

| Use this skill | Use model-selection/esm instead |
|---|---|
| Embedding sequences with a known model | Choosing between ESM2, ESM C, ESM-1v, ESMFold, ESM3 |
| Zero-shot variant / mutation scoring | LoRA or full fine-tuning setup |
| Structure prediction with ESMFold | Prediction heads and training loops |
| ESM3 generation or masked infilling | ESM3 fine-tuning on `esm3-sm-open-v1` |
| ESM C high-quality protein representations | Choosing between ESM2 and ESM C for embeddings |

---

## Package table

| Use case | Package | Install |
|---|---|---|
| ESM2 standard embeddings and inference | **transformers** (preferred) | `pip install transformers` |
| ESMFold, MSA Transformer, ESM-IF1, exact FAIR paths | **fair-esm** | `pip install fair-esm` |
| ESM-1v zero-shot variant scoring | **fair-esm** | `pip install fair-esm` |
| ESM3 open model or Forge API | **official esm** | `pip install esm` |
| ESM C (ESM Cambrian) local or Forge API | **official esm** | `pip install esm` |

---

## ESM2 — HuggingFace (recommended)

Suppose user wants to do inference with: `facebook/esm2_t33_650M_UR50D`. See model-selection/esm for size tradeoffs.

```python
import torch
from transformers import EsmModel, EsmTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").eval().to(device)

inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=1022)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

# Mean pool over residue positions, excluding BOS (index 0) and EOS
hidden = outputs.last_hidden_state          # (B, L+2, D)
token_hidden = hidden[:, 1:-1, :]           # strip BOS and EOS columns
token_mask = inputs["attention_mask"][:, 1:-1].unsqueeze(-1).float()
pooled = (token_hidden * token_mask).sum(1) / token_mask.sum(1).clamp(min=1.0)  # (B, D)
```

> For variable-length batches, the vectorised `[:, 1:-1]` slice slightly over-includes the EOS
> of shorter sequences. Use `embed_esm2_hf()` from `scripts/esm_inference.py` for correct
> per-sequence exclusion.

---

## ESM2 — native fair-esm

Use when ESMFold, MSA Transformer, or ESM-IF1 are also needed in the same environment.
For standalone ESM2 embeddings, prefer the HuggingFace path above.

```python
import torch, esm

model, alphabet = esm.pretrained.load_model_and_alphabet_hub("facebook/esm2_t33_650M_UR50D")
batch_converter = alphabet.get_batch_converter()
model = model.eval().to(device)

data = [("seq_0", "MKTAYIAK"), ("seq_1", "ACDEFGH")]
_, _, tokens = batch_converter(data)
tokens = tokens.to(device)

with torch.no_grad():
    out = model(tokens, repr_layers=[33], return_contacts=False)

reps = out["representations"][33]  # (B, L+2, D)
# Mean pool excluding BOS/EOS per sequence
embeddings = [reps[i, 1:len(seq)+1].mean(0).cpu().numpy() for i, (_, seq) in enumerate(data)]
```

See also `dimensionality-reduction/scripts/embedding_backends.py` for a batched implementation
of this path (`compute_esm_embeddings`).

---

## ESM-1v — zero-shot variant scoring

ESM-1v is a zero-shot variant effect predictor. It is **not** a general embedder.

```python
import torch, esm

model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
batch_converter = alphabet.get_batch_converter()
model.eval()

def masked_marginal_score(seq: str, mut_pos: int, mut_aa: str, wt_aa: str) -> float:
    """
    Log-odds of substitution wt_aa → mut_aa at mut_pos (1-indexed, UniProt convention).
    BOS is at token index 0, so residue at position P sits at token index P — no off-by-one.
    """
    _, _, tokens = batch_converter([("protein", seq)])
    tokens[0, mut_pos] = alphabet.mask_idx
    with torch.no_grad():
        logits = model(tokens)["logits"][0, mut_pos]
    return (logits[alphabet.get_idx(mut_aa)] - logits[alphabet.get_idx(wt_aa)]).item()
```

For production: average scores over all 5 ESM-1v models
(`esm1v_t33_650M_UR90S_1` through `_5`). Use `score_variants_esm1v()` in the script, which
handles ensemble loading and averaging.

---

## ESMFold — structure prediction

Single-sequence structure predictor. Returns a PDB string.

```python
import torch, esm

model = esm.pretrained.esmfold_v1().eval().cuda()
# For sequences >400 aa on a 24 GB GPU, add:
# model.set_chunk_size(64)

with torch.no_grad():
    pdb_str = model.infer_pdb("MKTAYIAKQRQISFVK...")

with open("prediction.pdb", "w") as f:
    f.write(pdb_str)
```

**Never fine-tune ESMFold.** Treat it as a frozen predictor.

---

## ESM3 — embeddings and generation

ESM3 is from EvolutionaryScale (not Meta). Requires `pip install esm` (the official package,
distinct from fair-esm).

```python
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig

model = ESM3.from_pretrained("esm3-sm-open-v1").to("cuda").eval()

# Sequence-track embeddings — shape (L, d_model), no BOS/EOS
protein = ESMProtein(sequence="MKTAYIAKQRQISFVK...")
encoded = model.encode(protein)
embeddings = encoded.sequence   # (L, d_model)

# Masked infilling — use "_" for positions to fill
protein = ESMProtein(sequence="MKTAY____QRQISFVK")
output  = model.generate(protein, GenerationConfig(track="sequence", num_steps=8))
print(output.sequence)
```

Forge API for larger hosted models (`esm3-medium-2024-08`, `esm3-large-2024-08`):

```python
from esm.sdk import client as esm_client
forge = esm_client(model="esm3-medium-2024-08",
                   url="https://forge.evolutionaryscale.ai",
                   token=os.environ["ESM_API_KEY"])
output = forge.generate(protein, GenerationConfig(track="sequence", num_steps=8))
```

---

## ESM C — embeddings

ESM Cambrian (ESM C) is EvolutionaryScale's representation-focused family — the high-performance
replacement for ESM2. **Not generative.** Requires `pip install esm` (same package as ESM3).

Local models: `esmc_300m` (300M), `esmc_600m` (600M).
Forge API: `esmc-6b-2024-12` (6B parameters).

```python
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

client = ESMC.from_pretrained("esmc_300m").to("cuda").eval()

protein = ESMProtein(sequence="MKTAYIAKQRQISFVK...")
protein_tensor = client.encode(protein)
logits_output = client.logits(
    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
)
embeddings = logits_output.embeddings  # (L, d_model) — per-residue, no BOS/EOS
```

Forge API (6B model):

```python
from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.sdk.api import ESMProtein, LogitsConfig

client = ESM3ForgeInferenceClient(
    model="esmc-6b-2024-12",
    url="https://forge.evolutionaryscale.ai",
    token=os.environ["ESM_API_KEY"],
)
# Same encode → logits interface as the local model
```

---

## Hard constraints

- **ESM2 max length:** 1022 tokens. Always pass `truncation=True, max_length=1022` to the tokenizer.
- **ESMFold:** never fine-tune. Use `model.set_chunk_size(64)` for sequences >400 aa on 24 GB GPU.
- **ESM-1v `mut_pos`:** 1-indexed. BOS is at token 0; residue P is at token P. Off-by-one produces wrong scores silently.
- **ESM-1v ensemble:** average over all 5 models for production variant effect predictions.
- **ESM3 license:** EvolutionaryScale Community License — non-commercial use only for `esm3-sm-open-v1`. Flag before production deployment.
- **ESM C license:** Cambrian Open License Agreement — check `evolutionaryscale.ai/policies/cambrian-open-license-agreement` for commercial use terms. Different from ESM3's Community License.
- **Batched tokenization:** always pass `padding=True` together with `truncation=True`. Never pass a list of sequences without padding.

---

## Script

`scripts/esm_inference.py` provides importable functions and a CLI for all inference modes above.

```
python scripts/esm_inference.py embed-esm2-hf --sequence MKTAYIAKQRQISFVK --output emb.npy
python scripts/esm_inference.py score-variants --sequence MKTAY... --mutation A1G --output scores.csv
python scripts/esm_inference.py fold --sequences seqs.txt --output-dir pdb_outputs/
python scripts/esm_inference.py embed-esm3 --sequences seqs.txt --output emb.npz
python scripts/esm_inference.py generate-esm3 --sequence "MKTAY____ISFVK"
python scripts/esm_inference.py embed-esmc --sequences seqs.txt --model esmc_300m --output emb.npz
python scripts/esm_inference.py --help
```
