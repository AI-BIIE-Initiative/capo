# CAPO Skills

A registry of composable skills for protein language model (PLM) fine-tuning** and the bioinformatics workflows around it — dataset profiling, preprocessing, clustering, model selection, training-code generation, inference, and experiment tracking. Each skill is a self-contained folder (a `SKILL.md` plus any scripts and references) that agentic coding systems can load on demand when its description matches the task at hand.

## How the skills fit together

A typical fine-tuning run flows through the registry like this:

1. **Profile** the dataset with `profiling-datasets` — it detects the modality, loads the data, runs the matching analysis, and recommends preprocessing.
2. **Preprocess** with the relevant `data-processing/*` skill, build homology-safe splits with `clustering`, and inspect representations with `dimensionality-reduction`.
3. **Select** a model and strategy with `model-selection`, then generate and self-repair the training scripts with `code-writing`.
4. **Train** on Hugging Face Jobs (`huggingface-jobs`) or a Lambda GPU (`cloud-provider-connection/lambda` + `lambda-session`), logging metrics with `trackio`.
5. **Run inference** with the relevant `model-inference/*` skill — embeddings, variant effects, structure prediction, or protein design.

Supporting skills (`uniprot`, `gtars`, `hf-cli`, `cost-estimation`) are used throughout.

## Available skills

### Pipeline entry points

| Skill | Description |
|---|---|
| `profiling-datasets` | **Start here for any dataset.** Detects format and modality, loads via the right loader, runs analysis, and recommends preprocessing. Primary input is a HF Hub dataset ID; local files (FASTA, FASTQ, BAM, CSV, FCS, H5AD, Parquet, …) are also supported. |
| `loading-data` | Internal router called by `profiling-datasets` (Stage 2). Dispatches a file path to the correct per-format loader (FASTA, FASTQ, BAM, CSV, Excel, FCS, H5/H5AD, JSONL, Parquet). Not invoked directly. |

### Dataset analysis (called internally by `profiling-datasets`)

| Skill | When used |
|---|---|
| `analysis/analyze-protein-sequences` | Protein / nucleotide sequences — length stats, alphabet composition, ESM2 length flags, duplicates, label and split balance. |
| `analysis/analyze-fastq-reads` | FASTQ raw reads — read-length and quality-score distributions, Q20/Q30 thresholds, per-position base composition. Flags FASTQ as non-PLM. |
| `analysis/analyze-bam-reads` | BAM/SAM aligned reads — mapping rate, MAPQ distribution, flag breakdown, contig counts. Report-only. |
| `analysis/analyze-tabular` | CSV / Parquet / Excel — per-column stats, correlations, class balance, leakage candidates, and instrument-aware recommendations (plate reader, Octet/BLI, Nanodrop, cell counter). |
| `analysis/analyze-fcs` | Flow cytometry FCS — per-channel distributions, FSC vs SSC scatter, compensation check. |
| `analysis/analyze-single-cell` | H5AD / H5 / MTX single-cell RNA-seq — shape, sparsity, scanpy QC metrics and plots. |

### Data preprocessing

| Skill | Description |
|---|---|
| `data-processing/protein-sequence-data` | Clean, validate, filter, mask labels, deduplicate, and split protein-sequence datasets. Consumes a Dataset Profile from `profiling-datasets`; does not reload or re-detect. |
| `data-processing/numerical-data` | Validate, impute, normalize, encode, and split tabular datasets into leakage-checked, split-aware feature matrices and targets. |
| `data-processing/single-cell` | QC, filtering, normalization, and highly-variable-gene selection for single-cell RNA-seq (Scanpy / Seurat). |

### Assay-to-dataset pipelines

| Skill | Description |
|---|---|
| `assay-fastq-to-plm-ready-dataset` | Convert paired-end FASTQ reads from a yeast-display RBD binding screen into a PLM-ready labeled CSV. Runs QC and paired-end merge (fastp), variable-region extraction, RBD constant-sequence reconstruction, DNA→AA translation, deduplication, and binder/non-binder labeling. |

### Sequence analysis and representation

| Skill | Description |
|---|---|
| `clustering` | Cluster protein sequences or embeddings (density-, centroid-, or hierarchical-based, or MMseqs2/CD-HIT sequence identity), evaluate clusters biologically, and generate homology-safe train/val/test splits. Includes an MMseqs2 sub-skill. |
| `dimensionality-reduction` | Reduce high-dimensional representations (one-hot, k-mer, ESM embeddings) to 2D/3D maps and compressed feature matrices via PCA, UMAP, and t-SNE. |
| `uniprot` | Retrieve protein sequences, functional annotations, domain boundaries, homologs, and PDB cross-references from UniProt. |
| `gtars` | High-performance genomic interval analysis (BED files, coverage tracks, overlap detection, tokenization for ML) via a Rust core with Python bindings. |

### Model selection and training

| Skill | Description |
|---|---|
| `model-selection` | Select the best PLM and fine-tuning strategy based on task, sequence length, compute, openness, and label availability. Includes an ESM sub-skill. |
| `code-writing` | Authoritative spec for the fine-tuning scripts the agent generates (`train.py`, `probe.py`, evaluation) and the contracts its code-repair step must preserve when patching them across the repair ladder. |
| `huggingface-jobs` | Run preprocessing, training, or inference on Hugging Face Jobs infrastructure — UV scripts, Docker jobs, GPU selection, token auth, secrets, timeouts, and result persistence. |
| `hf-cli` | Download, upload, and manage repositories, models, datasets, and Spaces on the Hugging Face Hub via the `hf` CLI. |
| `cost-estimation` | Estimate GPU compute cost and runtime across Hugging Face, AWS, GCP, Azure, and Lambda, with hardware-sizing and cost-optimization guidance. |

### Model inference

| Skill | Description |
|---|---|
| `model-inference/esm` | ESM2 / ESM C embeddings, ESM-1v zero-shot variant scoring, ESMFold single-chain structure prediction, and ESM3 masked infilling and generation. |
| `model-inference/ankh` | Ankh protein embeddings and Ankh3 sequence completion (ankh-base, ankh-large, ankh3-large, ankh3-xl) via HuggingFace. |
| `model-inference/prottrans` | ProtBert and ProtT5-XL-UniRef50 embeddings (Rostlab ProtTrans family) via HuggingFace. |
| `model-inference/boltz` | Boltz-2 — predict structure and binding affinity of protein complexes, protein-ligand poses, and multi-chain assemblies (protein, DNA, RNA, small molecules). |
| `model-inference/boltzgen` | BoltzGen — generative protein design: binders, antibody/nanobody CDRs, peptides, small-molecule binders, and protein redesign across six protocols. |
| `model-inference/chai` | Chai-1 — predict complex structure and binding poses at AF3-level accuracy from ESM embeddings alone (no MSA required); optional MSAs or templates for higher accuracy. |

### Cloud and compute

| Skill | Description |
|---|---|
| `cloud-provider-connection/lambda` | Manage the full Lambda On-Demand GPU lifecycle via MCP tools — SSH key discovery, provisioning, GPU verification, rsync transfer, cost tracking, and safe termination. |
| `lambda-session` | Manage multi-session parallel execution on a running Lambda instance (tmux workspace, remote session, run-state files). Complements `cloud-provider-connection/lambda` — covers what to do once the instance IP is known. |

### Experiment tracking

| Skill | Description |
|---|---|
| `tracking-experiments/trackio` | Track and visualize PLM fine-tuning runs with Trackio — Python logging API, training-diagnostic alerts, CLI metric retrieval, JSON output for automation, and HF Space syncing. |

## Installing the skills

Each skill is a self-contained folder. Copy the ones you need (or all of them) into your tool's skills directory — **globally** (available across all projects) or **per-project** (available only in that project).

| Tool | Global | Per-project |
|---|---|---|
| Claude Code | `~/.claude/skills/` | `.claude/skills/` |
| Codex | `~/.codex/skills/` | `.codex/skills/` |
| Gemini CLI | `~/.gemini/skills/` | `.gemini/skills/` |

For example, to install all skills globally for Claude Code:

```bash
mkdir -p ~/.claude/skills
cp -r skills/* ~/.claude/skills/
```

Swap the destination for another tool or for a project-level directory (e.g. `.claude/skills/`). Skills are cross-compatible across these ecosystems as long as they keep the same folder structure — a `SKILL.md` with frontmatter at the folder root.
