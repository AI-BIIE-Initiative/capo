---
name: esm
description: ESM2, ESM-1v, ESMFold (Meta / facebookresearch) and ESM3 (EvolutionaryScale) protein language models. Covers fine-tuning workflows, embeddings, zero-shot scoring, and multimodal generation. Always read model-selection SKILL.md first to confirm model choice before using this skill.
compatibility: PyTorch ≥2.0, transformers ≥4.30, peft ≥0.6. ESM3 requires `pip install esm`.
---

# ESM Family

**Meta / facebookresearch (ESM2, ESM-1v, ESMFold):** https://github.com/facebookresearch/esm
**EvolutionaryScale (ESM3):** https://github.com/evolutionaryscale/esm

---

## ESM2 — Encoder, fine-tuning, embeddings, scoring

### Models

| Model | Params | HuggingFace ID | Best for |
|---|---|---|---|
| `esm2_t6_8M` | 8M | `facebook/esm2_t6_8M_UR50D` | Fast screening, CPU |
| `esm2_t12_35M` | 35M | `facebook/esm2_t12_35M_UR50D` | Small-data linear probe |
| `esm2_t30_150M` | 150M | `facebook/esm2_t30_150M_UR50D` | Balanced speed/quality |
| `esm2_t33_650M` | 650M | `facebook/esm2_t33_650M_UR50D` | **Default** — most tasks |
| `esm2_t36_3B` | 3B | `facebook/esm2_t36_3B_UR50D` | Max quality, LoRA only |
| `esm2_t48_15B` | 15B | `facebook/esm2_t48_15B_UR50D` | Research; frozen only |

Max sequence length: **1022 tokens** for all ESM2 variants.

> **ESM-1v** (`esm1v_t33_650M_UR90S_1` through `_5`): 5-model ensemble trained on UniRef90. Best for **zero-shot variant effect** via masked-marginal scoring. Not a general embedder.
>
> **ESMFold** (`esm.pretrained.esmfold_v1()`): Single-sequence structure prediction. Do not fine-tune — use as a frozen predictor downstream of sequence design.

---

### Loading (HuggingFace — recommended)

```python
from transformers import EsmModel, EsmTokenizer

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model     = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

inputs  = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=1022)
outputs = model(**inputs)

# Mean pool over non-padding positions
mask = inputs["attention_mask"].unsqueeze(-1).float()
embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)  # (B, d_model)
```

---

### Fine-tuning strategy by label regime

| Labels | Strategy | Notes |
|---|---|---|
| 0 | Zero-shot PLL / masked-marginal | No gradient updates |
| < 500 | **Linear probe** — freeze backbone | `model.requires_grad_(False)`, train head only |
| 500–10k | **LoRA** (r=8–16) | Preferred default |
| 10k–100k | **LoRA** (r=16–32) + optional top-2 block unfreeze | Monitor val loss |
| > 100k | **Full fine-tune** | Start from LoRA checkpoint; lr=1e-5 |

### LoRA setup

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["query", "key", "value", "dense"],
    bias="none",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

### Hyperparameters

| Setting | Linear probe | LoRA | Full fine-tune | Source |
|---|---|---|---|---|
| LR | 1e-3 | 3e-4 | 1e-5 | LoRA paper (Hu et al., ICLR 2022) for LoRA LR; BERT fine-tuning range (Devlin et al., 2019) for 1e-5 |
| Optimizer | AdamW, wd=0.01 | AdamW, wd=0.01 | AdamW, wd=0.01 | AdamW paper (Loshchilov & Hutter, ICLR 2019) |
| Precision | fp16/bf16 | bf16 | bf16 | bf16 avoids fp16 gradient underflow on Ampere+ GPUs — hardware convention, not paper-derived |
| Max tokens/batch | 4096 | 2048–4096 | 1024–2048 | Empirical — sized for ESM2-650M on 24GB GPU; scale down for larger models |

### Prediction heads

```python
import torch.nn as nn

class SequenceHead(nn.Module):
    def __init__(self, d_model: int, n_out: int = 1):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_out))

    def forward(self, hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)   # mean pool
        return self.head(pooled)

# Per-residue labeling: skip BOS (idx 0) and EOS (idx -1) tokens
residue_out = nn.Linear(d_model, n_labels)(outputs.last_hidden_state[:, 1:-1, :])
```

### Zero-shot variant effect (ESM-1v masked-marginal)

```python
import esm, torch

model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
batch_converter  = alphabet.get_batch_converter()
model.eval()

def masked_marginal_score(seq: str, mut_pos: int, mut_aa: str, wt_aa: str) -> float:
    """Log-odds of mutation at mut_pos (1-indexed)."""
    data = [("protein", seq)]
    _, _, tokens = batch_converter(data)
    tokens[0, mut_pos] = alphabet.mask_idx
    with torch.no_grad():
        logits = model(tokens)["logits"][0, mut_pos]
    return (logits[alphabet.get_idx(mut_aa)] - logits[alphabet.get_idx(wt_aa)]).item()

# For production: average over the ensemble of 5 ESM-1v models
```

---

## ESM3 — Multimodal generation (sequence + structure + function)

> ESM3 is from **EvolutionaryScale**, not Meta. It is a generative model that jointly reasons over three biological tracks. Reference: https://github.com/evolutionaryscale/esm

### Three tracks — any can be masked or conditioned independently

| Track | Representation | Example use |
|---|---|---|
| **Sequence** | Amino acid tokens | Infill masked positions, embed sequence |
| **Structure** | 3D backbone coordinates → VQ-VAE tokens | Condition on known structure |
| **Function** | GO terms, InterPro IDs, free-text keywords | Condition on biological function |

### Models

| Model | Params | Access |
|---|---|---|
| `esm3-sm-open-v1` | 1.4B | Open weights — `pip install esm` |
| `esm3-small-2024-08` | ~300M | EvolutionaryScale Forge API |
| `esm3-medium-2024-08` | ~7B | EvolutionaryScale Forge API |
| `esm3-large-2024-08` | ~98B | EvolutionaryScale Forge API |

```bash
pip install esm   # open model + Forge API client
```

### Core usage (open model — local GPU)

```python
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig

model = ESM3.from_pretrained("esm3-sm-open-v1").to("cuda").eval()

# Embeddings from sequence track
protein    = ESMProtein(sequence="MKTAYIAKQRQISFVK...")
encoded    = model.encode(protein)
embeddings = encoded.sequence   # (L, d_model)

# Masked sequence infilling
protein = ESMProtein(sequence="MKTAY____QRQISFVK")   # _ = masked positions
output  = model.generate(protein, GenerationConfig(track="sequence", num_steps=8))
print(output.sequence)
```

### Structure-conditioned sequence design

```python
import numpy as np
from esm.sdk.api import ESMProtein, GenerationConfig

# coords: (L, 3, 3) — N, CA, C backbone atoms per residue
coords  = np.load("backbone.npy")
protein = ESMProtein(coordinates=coords, sequence="_" * len(coords))
output  = model.generate(protein, GenerationConfig(track="sequence", num_steps=16))
print(output.sequence)   # designed sequence for given backbone
```

### Via EvolutionaryScale Forge API (medium / large models)

```python
from esm.sdk import client
from esm.sdk.api import ESMProtein, GenerationConfig

forge = client(
    model="esm3-medium-2024-08",
    url="https://forge.evolutionaryscale.ai",
    token="<ESM_API_KEY>",   # or set ESM_API_KEY env var
)
protein = ESMProtein(sequence="MKTAY____ISFVK")
output  = forge.generate(protein, GenerationConfig(track="sequence", num_steps=8))
```

### Fine-tuning ESM3

Supported locally on `esm3-sm-open-v1` (1.4B) only. Use the same LoRA regime as ESM2 — target sequence track hidden states for prediction heads. Multi-track inputs (partial structure or function tokens) can serve as conditioning context even in supervised tasks.

For larger models, fine-tuning requires EvolutionaryScale enterprise API access.

### ESM2 vs ESM3 — when to use which

| Need | ESM2 | ESM3 |
|---|---|---|
| Embeddings for downstream ML | ✓ fast, HF ecosystem | ✓ richer, multimodal |
| Supervised fine-tuning | ✓ full HF ecosystem | ✓ open model only |
| Zero-shot variant effect | ✓ ESM-1v masked-marginal | ✓ sequence track |
| Sequence design / generation | ✗ encoder only | ✓ generative |
| Structure-conditioned design | ✗ | ✓ native |
| Function-conditioned design | ✗ | ✓ native |
| Structure prediction | via ESMFold (frozen) | ✓ native |
| Cost / infrastructure | Lower; GPU or HF Inference | Higher; Forge API for large models |

---

## Quick decision

```
Embeddings or supervised fine-tuning only?        → ESM2 (650M default)
Zero-shot variant/mutation effect?                → ESM-1v (masked-marginal ensemble)
Structure prediction from sequence?               → ESMFold (frozen, don't fine-tune)
Sequence design, structure- or function-conditioned generation?  → ESM3
```
