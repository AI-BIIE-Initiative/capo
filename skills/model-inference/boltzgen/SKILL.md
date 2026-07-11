---
name: boltzgen
description: Generative protein design with BoltzGen: binder design, antibody/nanobody CDR design, peptide design, small-molecule binder generation, protein redesign. Covers YAML input construction, six protocols (protein-anything, peptide-anything, protein-small_molecule, antibody-anything, nanobody-anything, protein-redesign), pipeline step selection (design, inverse_folding, folding, affinity, filtering), and output parsing (ranked CIFs, metrics CSVs). Weights from boltzgen/boltzgen-1 on HuggingFace. Use when designing new protein sequences and structures for a target of interest, with optional binding affinity optimization. For complex structure prediction without design, see model-inference/boltz. For embedding-based design or variant scoring, see model-inference/esm.
compatibility: Python ≥3.11, CUDA required, GPU recommended. pip install boltzgen. Clone https://github.com/HannesStark/boltzgen and run from repo root.
---

# BoltzGen Inference

---

## When to use / When NOT to use

| Use this skill | Use a different skill |
|---|---|
| Designing protein binders or peptides | Structure prediction → model-inference/boltz |
| Antibody or nanobody CDR design | Embeddings only → model-inference/esm |
| Small-molecule binder generation | Variant effect scoring → model-inference/esm |
| Protein redesign / optimization | |

---

## Installation

```bash
pip install boltzgen huggingface_hub pyyaml
git clone https://github.com/HannesStark/boltzgen
cd boltzgen   # run all boltzgen commands from this root
```

Weights (~6 GB total) download automatically from `boltzgen/boltzgen-1` on HuggingFace on first run.
To pre-download and pin a specific version:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="boltzgen/boltzgen-1",
    local_dir="~/.boltzgen/boltzgen1",
)
# boltzgen1_diverse.ckpt    — diversity-optimized design weights
# boltzgen1_adherence.ckpt  — specification-adherent design weights
# boltzgen1_ifold.ckpt      — inverse folding weights
# boltz2_conf_final.ckpt    — bundled Boltz-2 structure prediction weights
# boltz2_aff.ckpt           — bundled Boltz-2 affinity prediction weights
```

Redirect the automatic cache with `--cache /path/to/cache` or `export HF_HOME=/path`.

---

## YAML input format

BoltzGen uses an `entities`-based schema (not Boltz-2's `sequences`).

```yaml
entities:
  - protein:
      id: B
      sequence: 80..140     # length range for the designed chain
  - file:
      path: 6m1u.cif        # target structure (path relative to repo root, or absolute)
      include:
        - chain:
            id: A
```

Include only the chains from the target file that are relevant to the design task.
Use absolute paths to avoid ambiguity when running from different working directories.

---

## Running designs

```bash
# Validation run (50 designs, top 10 selected) — run from cloned repo root
boltzgen run example/vanilla_protein/1g13prot.yaml \
  --output workbench/test_run \
  --protocol protein-anything \
  --num_designs 50 \
  --budget 10

# Production run
boltzgen run spec.yaml \
  --output results \
  --protocol protein-anything \
  --num_designs 10000 \
  --budget 100 \
  --devices 4

# Pre-download all model weights
boltzgen download all --cache /data/models

# Merge results from multiple parallel runs
boltzgen merge run_a run_b run_c --output merged

# Re-run only the filtering step with a tighter alpha
boltzgen run spec.yaml --output results --steps filtering --alpha 0.05 --budget 60

# Resume an interrupted run
boltzgen run spec.yaml \
  --output results \
  --protocol protein-anything \
  --num_designs 10000 \
  --budget 100 \
  --reuse
```

### Key options

| Option | Default | Notes |
|---|---|---|
| `--num_designs` | — | Total intermediate designs. Use 50 for validation; 10k–60k for production |
| `--budget` | — | Final set size after diversity-optimized selection |
| `--protocol` | — | See protocols table below |
| `--cache` | `~/.cache` | Model storage directory (or set `HF_HOME`) |
| `--alpha` | 0.5 | Quality-diversity tradeoff: 0.0 = maximize quality, 1.0 = maximize diversity |
| `--reuse` | off | Resume an interrupted run without losing progress |
| `--devices` | 1 | Number of GPUs |
| `--diffusion_batch_size` | — | Samples per diffusion batch; tune for GPU VRAM |
| `--steps` | all | Selectively run pipeline steps (see step list below) |
| `--filter_biased` | off | Remove composition outliers |
| `--additional_filters` | — | Hard filter thresholds, e.g. `'ALA_fraction<0.3'` |
| `--refolding_rmsd_threshold` | — | RMSD cutoff for the refolding filter |
| `--force_download` | off | Redownload model weights even if cached |

### Protocols

| Protocol | Use case | Notes |
|---|---|---|
| `protein-anything` | Protein binder design | Includes design folding |
| `peptide-anything` | Peptide design | No cysteines, no design folding |
| `protein-small_molecule` | Small-molecule binder generation | Affinity prediction included |
| `antibody-anything` | Antibody CDR design | No cysteines, no design folding |
| `nanobody-anything` | Nanobody CDR design | No cysteines, no design folding |
| `protein-redesign` | Protein optimization | Uses `design_mask`; no design folding |

### Pipeline steps

Run a subset of steps with `--steps`:

| Step | Description |
|---|---|
| `design` | Diffusion-based backbone generation |
| `inverse_folding` | Sequence design onto the backbone |
| `folding` | Full-complex refolding with Boltz-2 |
| `design_folding` | Binder-only refolding |
| `affinity` | Binding affinity prediction (`protein-small_molecule` only) |
| `analysis` | Per-design quality metrics computation |
| `filtering` | Diversity selection and ranking |

---

## Output

```
output/
├── intermediate_designs/                    # Backbone structures from diffusion (CIF)
├── intermediate_designs_inverse_folded/     # After sequence design (CIF)
├── refold_cif/                              # Refolded full complexes (CIF)
├── final_ranked_designs/                    # Filtered, diversity-ranked final set (CIF)
├── metrics_*.csv                            # Per-design quality metrics
└── results.pdf                              # Summary report
```

`final_ranked_designs/` filenames encode rank (lower index = higher quality). Cross-reference with
`metrics_*.csv` for the full metric breakdown (pLDDT, RMSD, iPTM, diversity scores, filter flags).

---

## Hard constraints

- **Run from the cloned repo root** — BoltzGen example YAMLs reference structure files with paths relative to the YAML file's directory; the repo `example/` tree must be intact.
- **GPU required** — BoltzGen is not designed for CPU-only inference.
- **Models ~6 GB** — Ensure sufficient disk space in the cache directory.
- **No HuggingFace Inference Provider** — `boltzgen/boltzgen-1` is not deployed by any provider; weights must be downloaded locally first.
- **`final_ranked_designs/` requires `filtering` step** — If you run only `--steps design inverse_folding`, this directory will be empty.

---

## Script

`scripts/boltzgen_inference.py` provides YAML entity builders, a subprocess wrapper for `boltzgen run`,
merge and download helpers, and output parsers for metrics CSVs and ranked CIF files.

```bash
python scripts/boltzgen_inference.py download-weights              # pre-fetch from HuggingFace
python scripts/boltzgen_inference.py run spec.yaml \
    --output ./out/ --protocol protein-anything --num-designs 50 --budget 10
python scripts/boltzgen_inference.py merge run_a run_b --output merged
```
