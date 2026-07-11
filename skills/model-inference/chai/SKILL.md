---
name: chai
description: >
  Use this skill when predicting the structure of protein complexes, protein-ligand poses,
  or multi-chain assemblies (protein, DNA, RNA, small molecules) with Chai-1.
  Use when the user wants AF3-level accuracy without preparing MSAs upfront — Chai-1 runs
  well on ESM embeddings alone and can optionally add MSAs or templates for improved accuracy.
  Use even when the user doesn't mention Chai explicitly but asks to "fold a complex",
  "predict a binding pose", or "dock a ligand". Do not use for binding affinity prediction
  (use model-inference/boltz) or for sequence embeddings / variant scoring (use model-inference/esm).
  Prefer Boltz when YAML-style constraints, pocket specification, or commercial-scale affinity
  ranking are needed.
compatibility: Linux only. Python ≥3.10, CUDA + bfloat16 GPU. Recommended: A100/H100 80 GB; A10/A30/RTX 4090 for small complexes. pip install chai_lab.
---

# Chai-1 Inference

---

## When to use / When NOT to use

| Use Chai-1 | Use a different skill |
|---|---|
| Protein-protein complex structure prediction | Binding affinity prediction → model-inference/boltz |
| Protein-ligand pose prediction (no affinity needed) | Pocket/contact constraints → model-inference/boltz |
| Multi-chain complexes with DNA or RNA | Sequence embeddings / variant scoring → model-inference/esm |
| Fast runs without pre-computed MSAs (ESM embeddings default) | Single-chain only → ESMFold via model-inference/esm |
| Commercial projects (Apache 2.0 license) | Fine-tuning Chai-1 |

---

## Chai-1 vs Boltz-2

| | Chai-1 | Boltz-2 |
|---|---|---|
| Accuracy | Matches AF3 | Matches AF3 |
| Default mode | ESM embeddings (no MSA required) | Requires MSA or `--use_msa_server` |
| MSA strategy | ColabFold MMseqs2; `aligned.pqt` files | Taxonomy-based + dense pairing |
| Input format | FASTA | YAML |
| Affinity prediction | **Not supported** | Supported |
| Constraints | Inter-chain contacts, covalent bonds | Pocket, contact, bond (richer API) |
| Samples by default | 5 | 1 |
| License | Apache 2.0 (commercial OK) | MIT (commercial OK) |
| Primary support | Chai Discovery | MIT / open community |

Use Boltz-2 when you need affinity scores, pocket constraints, or YAML-level control over inputs.
Use Chai-1 when you want quick multi-chain structure predictions without MSA setup.

---

## Installation

**Preferred — pip from PyPI:**

```bash
pip install chai_lab==0.6.1
```

**Latest development version (updates daily):**

```bash
pip install git+https://github.com/chaidiscovery/chai-lab.git
```

Model weights download automatically from `chaidiscovery/chai-1` on HuggingFace on first run.
Default storage: `<site-packages>/chai_lab/downloads`. Override with the `CHAI_DOWNLOADS_DIR`
environment variable before importing `chai_lab`.

To pre-download and pin weights explicitly:

```python
import os
from huggingface_hub import snapshot_download

local_dir = os.environ.get("CHAI_DOWNLOADS_DIR", str(Path.home() / ".chai" / "weights"))
snapshot_download(repo_id="chaidiscovery/chai-1", local_dir=local_dir)
```

Then set `CHAI_DOWNLOADS_DIR` to that path in your environment before running any inference.

---

## FASTA input format

Each entity in a complex is a separate FASTA entry with a type-prefixed header.
One FASTA file = one complex.

```
>protein|name=receptor
AGSHSMRYFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPR...
>protein|name=nanobody
GAAL
>ligand|name=inhibitor
CC(=O)Oc1ccccc1C(=O)O
>rna|name=guide-rna
AUCGAUCGAUCG
>dna|name=template-strand
ATCGATCGATCG
```

**Header format:** `>type|name=<identifier>` — the `type|` prefix is mandatory.
**Ligands:** encoded as SMILES strings (one ligand per entry).
**Modified residues:** inline PTM codes in parentheses, e.g. `AAA(SEP)AAA` for phosphoserine.
**Multiple identical chains:** add separate entries with distinct names.

---

## CLI usage

```bash
# Fast run — ESM embeddings only, 5 samples by default
chai-lab fold input.fasta output_folder

# With MSA + templates from public servers (recommended for better accuracy)
chai-lab fold --use-msa-server --use-templates-server input.fasta output_folder

# Custom ColabFold server
chai-lab fold --use-msa-server --msa-server-url http://your-server input.fasta output_folder

# See all options
chai-lab fold --help
```

---

## Python API

**Basic (ESM embeddings, no MSA):**

```python
import shutil
from pathlib import Path
from chai_lab.chai1 import run_inference

fasta_path = Path("input.fasta")
output_dir = Path("outputs")
shutil.rmtree(output_dir, ignore_errors=True)   # output dir MUST be empty
output_dir.mkdir()

candidates = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    num_trunk_recycles=3,        # refinement iterations; default 3
    num_diffn_timesteps=200,     # diffusion steps; default 200
    seed=42,
    device="cuda:0",
    use_esm_embeddings=True,     # default; set False only if providing MSAs
)

cif_paths = candidates.cif_paths              # list of Path objects, one per sample
agg_scores = [rd.aggregate_score.item() for rd in candidates.ranking_data]
best_cif = cif_paths[0]                       # highest aggregate_score
```

**With MSA directory (precomputed `aligned.pqt` files):**

```python
candidates = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    seed=42,
    device="cuda:0",
    use_esm_embeddings=True,
    msa_directory=Path("msas/"),   # directory containing chain_name.aligned.pqt files
    use_msa_server=False,          # mutually exclusive with msa_directory
)
```

**With MSA auto-generated from server:**

```python
candidates = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    seed=42,
    device="cuda:0",
    use_esm_embeddings=True,
    use_msa_server=True,           # calls ColabFold MMseqs2; mutually exclusive with msa_directory
)
```

---

## Output

```
outputs/
├── pred.model_idx_0.cif        # Highest-ranked structure (CIF format)
├── pred.model_idx_1.cif        # Additional samples (5 by default)
├── pred.model_idx_2.cif
├── pred.model_idx_3.cif
├── pred.model_idx_4.cif
└── scores.model_idx_N.npz      # Per-sample confidence metrics
```

### Parsing scores

```python
import numpy as np

scores = np.load(output_dir / "scores.model_idx_0.npz")
# scores["ptm"]           — predicted TM-score for the complex  [0, 1], higher=better
# scores["iptm"]          — interface TM-score                   [0, 1], higher=better
# scores["plddt"]         — per-residue confidence array         [0, 1], higher=better
# scores["clash_score"]   — steric clash penalty,                 lower=better
# scores["aggregate_score"] — primary ranking score (combination of above)
```

Thresholds: `ptm > 0.7` → confident global fold; `iptm > 0.5` → confident interface.

For convenience, aggregate scores are also accessible via `candidates.ranking_data` without
loading the `.npz` files:

```python
for i, (cif, rd) in enumerate(zip(candidates.cif_paths, candidates.ranking_data)):
    print(f"model_{i}: aggregate={rd.aggregate_score:.3f}  path={cif}")
```

---

## Hard constraints

- **Linux only.** No macOS or Windows support.
- **GPU required:** CUDA + bfloat16. Use A100/H100 80 GB for large complexes; A10/A30/RTX 4090 for smaller ones.
- **Output directory must be empty** when calling `run_inference`. Always `rmtree` before calling.
- **FASTA header:** `>type|name=...` prefix is mandatory. Missing prefix causes silent incorrect parsing.
- **Ligands:** one SMILES per FASTA entry. Multi-fragment SMILES (`.` separator) are not recommended.
- **MSA format:** `aligned.pqt`, not `.a3m`. Convert with `chai_lab.data.io.msas.read_aligned_pqt_to_msas` or the script helper below.
- **`msa_directory` and `use_msa_server` are mutually exclusive.** Providing both raises an error.
- **No binding affinity output.** For affinity scoring, use model-inference/boltz.
- **Weights location:** if `CHAI_DOWNLOADS_DIR` is set, it must be set *before* `import chai_lab`.

---

## Script

`scripts/chai_inference.py` provides FASTA builders, a `run_inference` wrapper, and
score parsers.

```bash
python scripts/chai_inference.py download-weights
python scripts/chai_inference.py fold input.fasta --out-dir ./out/
python scripts/chai_inference.py fold input.fasta --out-dir ./out/ --use-msa-server
python scripts/chai_inference.py fold input.fasta --out-dir ./out/ --msa-dir ./msas/
python scripts/chai_inference.py --help
```
