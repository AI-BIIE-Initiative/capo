---
name: boltz
description: >
  Use this skill when predicting the structure or binding affinity of protein complexes,
  protein-ligand poses, or multi-chain assemblies involving DNA or RNA with Boltz-2.
  Use when the user wants to dock a small molecule, rank binders by predicted affinity,
  model a protein-protein interface, or apply pocket/contact constraints — even if they
  don't mention Boltz explicitly. Covers YAML input construction, MSA handling, affinity
  prediction, and parsing confidence and affinity outputs. Do not use for single-chain
  structure prediction (use ESMFold via model-inference/esm) or embeddings-only tasks.
compatibility: Python ≥3.10, CUDA ≥12.0, GPU VRAM ≥24 GB (48 GB for large complexes). pip install boltz.
---

# Boltz-2 Inference

---

## When to use / When NOT to use

| Use this skill | Use a different skill |
|---|---|
| Protein-protein complex structure prediction | Single-chain structure → ESMFold (faster) |
| Protein-ligand pose prediction | Embeddings only → model-inference/esm |
| Binding affinity prediction (hit discovery, lead opt) | Training / fine-tuning Boltz |
| DNA/RNA-protein complex prediction | |
| Pocket- or contact-constrained docking | |

---

## Installation

```bash
pip install boltz          # installs the boltz CLI and Python package
```

Weights download automatically from `boltz-community/boltz-2` on HuggingFace on first run.
To pre-download and pin a specific version:

```python
from huggingface_hub import snapshot_download
repo = snapshot_download(
    repo_id="boltz-community/boltz-2",
    local_dir="~/.boltz/boltz2",   # or any writable path
)
# repo/boltz2_conf.ckpt   (~2.3 GB) — structure prediction weights
# repo/boltz2_aff.ckpt    (~2.1 GB) — affinity prediction weights
```

Then point boltz to them with `--checkpoint` and `--affinity_checkpoint`.

### GPU kernel dependency (REQUIRED — read before any GPU run)

`pip install boltz` does **NOT** install the CUDA triangular-multiplication
kernel. Boltz lazily imports `cuequivariance_torch` *inside the first GPU
forward pass* (`boltz/model/layers/triangular_mult.py`), so a missing kernel
raises `ModuleNotFoundError: No module named 'cuequivariance_torch'` **only
after the model starts running**, on ANY GPU — including A100/H100, not just old
cards. This is the single most expensive boltz failure mode: it surfaces minutes
into a paid GPU run, not at install time.

The kernel needs **six** packages, not one. The trap that has cost real GPU
runs: `pip install boltz cuequivariance-torch` looks complete and even lets
`import cuequivariance_torch` succeed, yet the run still crashes minutes into the
first forward pass — because the package boltz's kernel actually *executes*,
`cuequivariance_ops_torch`, lives in a **different** wheel
(`cuequivariance-ops-torch-cu12`) that nothing pulls in automatically, and two
of its own dependencies (`platformdirs`, `networkx`) are present-but-too-old on a
default image. Install the whole set explicitly, from the NVIDIA index, and add
every line to the run's `requirements.txt`:

```bash
pip install --extra-index-url https://pypi.nvidia.com \
  boltz \
  cuequivariance \
  cuequivariance-torch \
  cuequivariance-ops-cu12 \
  cuequivariance-ops-torch-cu12 \
  "platformdirs>=3.0.0" \
  "networkx>=3.0"
# cuequivariance-ops-torch-cu12 → cuequivariance_ops_torch — the runtime kernel; NOT pulled by `pip install boltz`
# platformdirs>=3.0.0 → cuequivariance_ops_torch uses the ensure_exists kwarg (added in 3.0); a default 2.5.x is too old
# networkx>=3.0       → older networkx uses np.int (removed in numpy>=1.24), which blocks the cuequivariance_torch import
```

VERIFY THE RUNTIME CALL PATH — not just the top-level import — BEFORE spending
GPU time (cheap; do this in the env-check gate). `import cuequivariance_torch`
alone is the trap: it passes while the run still dies later, because the kernel
is `cuequivariance_ops_torch`, imported lazily inside the first GPU forward:

```bash
python - <<'PY'
import cuequivariance_torch, cuequivariance_ops_torch          # BOTH modules, by name
from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
print("KERNELS_OK", cuequivariance_ops_torch.__version__)
PY
```

If you genuinely cannot install the kernels (unsupported CUDA/GPU combo), pass
`--no_kernels` to `boltz predict` for a slower pure-PyTorch path (correct, ~2–3×
slower) — this is the guaranteed-progress fallback and should be tried before
giving up on a GPU run.

**Pin the proven versions.** A known-good, version-pinned combo lives in
`references/boltz2-cuda.lock` (proven on A100 SXM4 40 GB / CUDA 12: the whole
`cuequivariance*` quartet at `0.10.0`, `platformdirs 4.10.0`, `networkx>=3.0`).
Install with `pip install -r references/boltz2-cuda.lock` and copy those exact
lines into the run's `requirements.txt` rather than re-deriving versions per run
— silent drift on any of these reintroduces a fixed crash.

---

## YAML input format

YAML is the preferred (and only fully-featured) input format. FASTA is deprecated.

```yaml
version: 1
sequences:
  - protein:
      id: A                    # unique chain ID; use [A, B] for identical copies
      sequence: MKTAYIAK...
      msa: ./path/to/seq.a3m  # precomputed MSA; omit when using --use_msa_server
  - protein:
      id: [B, C]               # two identical chains
      sequence: ACDEFGH...
      msa: empty               # force single-sequence mode (not recommended)
  - ligand:
      id: D
      smiles: 'CC(=O)Oc1ccccc1C(=O)O'   # SMILES string
  - ligand:
      id: E
      ccd: ATP                 # CCD code — alternative to smiles, not both
  - dna:
      id: F
      sequence: ATCGATCG
  - rna:
      id: G
      sequence: AUCGAUCG
```

### Constraints (optional)

```yaml
constraints:
  - pocket:                    # residues forming the binding site
      binder: D                # chain ID of the ligand / binder
      contacts:
        - [A, 42]              # [chain_id, residue_index (1-indexed)]
        - [A, 87]
      max_distance: 6          # Angstroms, range 4–20, default 6
      force: false             # true → use potential to enforce
  - contact:
      token1: [A, 42]
      token2: [D, 1]
      max_distance: 5
      force: false
  - bond:                      # covalent bond (CCD ligands + canonical residues only)
      atom1: [A, 10, SG]       # [chain_id, residue_idx, atom_name]
      atom2: [D, 1, C1]
```

### Affinity prediction (optional)

```yaml
properties:
  - affinity:
      binder: D                # must be a ligand chain (not protein/DNA/RNA)
```

Affinity has two outputs (see Output section). Only one ligand per run. Ligand must have ≤128 heavy+H atoms; best results at ≤56 atoms. Only protein targets are supported — do not run with RNA/DNA targets.

### Validate ligand SMILES BEFORE writing YAML (non-negotiable for batches)

An invalid SMILES (e.g. a phosphorus over-valence) crashes boltz's RDKit
standardization mid-run with `getNumImplicitHs() called without preceding call
to calcImplicitValence()` and skips (or aborts) that input. When generating many
YAMLs programmatically, sanitize first and drop + count failures rather than
letting one bad row pollute or kill a multi-complex `boltz predict`:

```python
from rdkit import Chem
def smiles_ok(smi: str) -> bool:
    mol = Chem.MolFromSmiles(smi)                 # None on parse failure
    if mol is None:
        return False
    # catchErrors=True returns the first failing flag instead of raising;
    # SANITIZE_NONE (0) means everything passed.
    return Chem.SanitizeMol(mol, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE
# write YAML only for ligands where smiles_ok is True; also enforce the
# ≤128 heavy+H atom limit here. Log dropped_invalid_ligands=<N>.
```

---

## Running predictions

```bash
# Structure prediction — MSA auto-generated
boltz predict input.yaml --use_msa_server --out_dir ./predictions/

# Affinity prediction (ligand must be in the YAML properties section)
boltz predict affinity.yaml --use_msa_server --out_dir ./predictions/

# With pre-downloaded HuggingFace weights
boltz predict input.yaml \
  --use_msa_server \
  --checkpoint ~/.boltz/boltz2/boltz2_conf.ckpt \
  --affinity_checkpoint ~/.boltz/boltz2/boltz2_aff.ckpt \
  --out_dir ./predictions/

# Higher-quality (slower) — AF3-equivalent parameters
boltz predict input.yaml --use_msa_server --recycling_steps 10 --diffusion_samples 25

# Improve physical plausibility of poses
boltz predict input.yaml --use_msa_server --use_potentials
```

### Key options

| Option | Default | Notes |
|---|---|---|
| `--use_msa_server` | off | Auto-generate MSA via mmseqs2. Required if no `msa:` in YAML |
| `--use_potentials` | off | Better physical quality of poses |
| `--recycling_steps` | 3 | 3 is fast; 10 matches AF3 quality |
| `--diffusion_samples` | 1 | Number of structure samples per input |
| `--output_format` | mmcif | `pdb` or `mmcif` |
| `--checkpoint` | auto | Path to structure prediction weights |
| `--affinity_checkpoint` | auto | Path to affinity prediction weights |
| `--no_kernels` | off | Pure-PyTorch fallback when the `cuequivariance` kernels are unavailable (slower) — see §GPU kernel dependency |
| `--override` | off | Re-run even if predictions already exist (forces full recompute) |

**Resumability (multi-input runs) — the biggest cost lever on a partial failure.**
Boltz writes one output folder per input under `out_dir/<input_name>/`, and each
`embeddings_<name>.npz` is a final, self-contained artifact (~100 MB). Treat it as
cached, expensive work that must NEVER be recomputed once valid.

- Do NOT pass `--override` on the main predict call: it forces a full recompute
  and discards completed outputs. Boltz already skips inputs whose output exists,
  so leaving `--override` off makes a crashed batch resume where it stopped.
- In a GENERATED training/embedding pipeline, do not rely on boltz's skip alone —
  pre-filter explicitly so the intent is auditable and idempotent:

  ```python
  import zipfile
  def _valid_embedding(out_dir, name):           # cheap: reads only the zip directory
      npz = out_dir / name / f"embeddings_{name}.npz"
      try:
          if not npz.is_file() or npz.stat().st_size < 1024:
              return False
          with zipfile.ZipFile(npz) as zf:
              return any(i.file_size >= 10_000 for i in zf.infolist())
      except (zipfile.BadZipFile, OSError):
          return False

  expected = [y.stem for y in sorted(yaml_dir.glob("*.yaml"))]
  missing  = [n for n in expected if not _valid_embedding(out_dir, n)]
  # run `boltz predict` ONLY on the YAMLs whose name is in `missing`; never --override.
  if not missing:
      print("all embeddings present — skipping boltz predict")
  ```

- `--override` is appropriate ONLY in the one-complex feasibility probe (a clean
  isolated measurement), NEVER in the main multi-complex stage.

A crashed batch then resumes instead of recomputing the expensive complexes
already done — e.g. a run over 80 complexes that died at complex 60 re-does 20,
not 80. This is the exact contract `capo.persistence.run_inventory` relies on to
plan an artifact-aware resume.

---

## Output

```
predictions/
└── <input_name>/
    ├── <input_name>_model_0.cif          # Best structure (CIF format)
    ├── <input_name>_model_1.cif          # Additional samples (if diffusion_samples > 1)
    ├── confidence_<input_name>_model_0.json
    ├── affinity_<input_name>.json        # Only if affinity was requested
    ├── plddt_<input_name>_model_0.npz
    ├── pae_<input_name>_model_0.npz
    └── pde_<input_name>_model_0.npz
```

### Confidence scores

```python
import json
conf = json.load(open("confidence_<name>_model_0.json"))
# conf["confidence_score"]  — primary ranking score: 0.8*complex_plddt + 0.2*iptm
# conf["ptm"]               — predicted TM-score for the complex  [0, 1], higher=better
# conf["iptm"]              — interface TM-score                   [0, 1], higher=better
# conf["complex_plddt"]     — average pLDDT                        [0, 1], higher=better
# conf["complex_pde"]       — average PDE (Angstroms), lower=better
```

Thresholds: `ptm > 0.7` → confident global fold; `iptm > 0.5` → confident interface.

### Affinity scores

```python
aff = json.load(open("affinity_<name>.json"))
# aff["affinity_probability_binary"]  — P(binder) ∈ [0,1]; use for hit discovery / decoy separation
# aff["affinity_pred_value"]          — log10(IC50 in μM); use for lead optimization
#   lower = stronger binder: -3 → IC50 1 nM, 0 → IC50 1 μM, 2 → IC50 100 μM
#   convert to pIC50 (kcal/mol): (6 - y) * 1.364   where y = affinity_pred_value
```

Use `affinity_probability_binary` to separate binders from non-binders.
Use `affinity_pred_value` **only** to rank active molecules against each other (not vs. inactives).

---

## Hard constraints

- **MSA required** for proteins unless `--use_msa_server` is set. Single-sequence mode (`msa: empty`) significantly hurts accuracy.
- **`cuequivariance` kernels**: a `cuequivariance_torch` / `cuequivariance_ops_torch`
  ModuleNotFoundError or import error means the GPU kernels are not fully installed
  (see §GPU kernel dependency) — install the **full six-package set**
  (`cuequivariance cuequivariance-torch cuequivariance-ops-cu12 cuequivariance-ops-torch-cu12`
  + `platformdirs>=3.0.0` + `networkx>=3.0`, from `--extra-index-url https://pypi.nvidia.com`),
  or add `--no_kernels` to fall back to the slower pure-PyTorch path. The runtime
  kernel is `cuequivariance_ops_torch` (a separate wheel from `cuequivariance-ops-cu12`),
  so verify *that* import, not just `cuequivariance_torch`. This is NOT limited to old
  GPUs; a fresh `pip install boltz` hits it on A100/H100 too.
- **Affinity ligand size**: ≤128 atoms (heavy + H kept by RDKit RemoveHs); best results ≤56 atoms.
- **Affinity targets**: only protein chains supported. RNA/DNA targets will not crash but output is unreliable.
- **YAML preferred**: FASTA is deprecated and cannot express constraints, affinity, templates, or modifications.
- **`force: true` constraints**: require a `max_distance` or `threshold` to be specified alongside them.
- **Output is CIF by default**; use `--output_format pdb` or convert with Biopython (see script).

---

## Script

`scripts/boltz_inference.py` provides YAML builders, a subprocess wrapper for `boltz predict`,
and output parsers for confidence and affinity JSON files.

```bash
python scripts/boltz_inference.py download-weights        # pre-fetch from HuggingFace
python scripts/boltz_inference.py predict input.yaml --out-dir ./out/ --use-msa-server
python scripts/boltz_inference.py predict inputs/ --out-dir ./out/ --diffusion-samples 3
```
