---
name: profiling-datasets-routing-rules
description: Format detection algorithm and skill routing table for profiling-datasets. HuggingFace Hub datasets are the primary input path. Read this when determining dataset type and which downstream skill to invoke.
---

# Routing Rules

> **Primary path:** HuggingFace Hub dataset ID. Check for this first before any file-based detection.
> **Local data/path**

---

## Step 1 — HuggingFace Hub detection (primary)

If the input looks like a Hub dataset ID (`"org/dataset-name"`, `"username/dataset-name"`, or a known HF URL), load it with a 5-row peek:

```python
from datasets import load_dataset
ds = load_dataset(dataset_id, split="train[:5]")
features = ds.features
splits = ds.info.splits  # get split names + row counts
```

Inspect `features` for bio-sequence columns (case-insensitive):

| Column name matches | dataset_type |
|--------------------------------------------------------------|--------------------------------------|
| `sequence` `seq` `aa_seq` `protein_sequence` `prot_seq` `aa` | `protein_sequence` (verify alphabet after load) |
| `dna` `dna_sequence` `nucleotide` `nt_seq`                   | `nucleotide_sequence`                |
| `mut_sequence` `mutant` `wt_sequence`                        | `protein_sequence` (variant_effect task) |
| `text` `content` `document` `abstract`                       | `text`                               |
| anything else with flat numeric/string columns               | `tabular`                            |

> **FASTA alphabet verification:** After loading a `protein_sequence` candidate, run alphabet detection (Step 4 below). If >90% chars are standard amino acids → confirm `protein_sequence`; else → reclassify as `nucleotide_sequence`.

Populate `hf_source` in the profile with `dataset_id`, `config_name`, `split_names`, `num_rows_per_split`, and `hf_features`.

---

## Step 2 — File extension detection

If input is a local file path, match against extension:

| Extension(s) | dataset_type | format | Loader skill |
|---|---|---|---|
| `.fasta` `.fa` `.faa` | `protein_sequence`* | `fasta` | `load-fasta` |
| `.fna` | `nucleotide_sequence`* | `fasta` | `load-fasta` |
| `.fastq` `.fq` | `raw_reads` | `fastq` | `load-fastq` |
| `.bam` `.sam` | `aligned_reads` | `bam` | `load-bam` |
| `.parquet` `.arrow` | `tabular` | `parquet` | `load-parquet` |
| `.csv` | `tabular` | `csv` | `load-csv` |
| `.tsv` `.tab` | `tabular` | `tsv` | `load-csv` |
| `.xlsx` `.xls` | `tabular` | `excel` | `load-excel` |
| `.jsonl` `.ndjson` | ambiguous | `jsonl` | `load-jsonl` (after Step 3) |
| `.asc` | `tabular` | `asc` | `load-asc` |
| `.fcs` | `flow_cytometry` | `fcs` | `load-fcs` |
| `.h5` `.h5ad` | `single_cell` | `h5` | `load-h5` |
| `.mtx` | `single_cell` | `mtx` | `load-mtx` (needs barcodes.tsv + features.tsv) |
| `.tif` `.tiff` `.scn` `.png` | `image` | `tif` | `load-tif` *(image — out of PLM scope)* |
| `.txt` or no extension | ambiguous | — | proceed to Step 3 |

> *`.fasta`/`.fa`/`.faa`* are assumed `protein_sequence`; *`.fna`* is assumed `nucleotide_sequence`. Both must be confirmed with alphabet detection after load (Step 4).

---

## Step 3 — Content sniff (first 5 lines)

| First-line pattern | dataset_type | format |
|---|---|---|
| Starts with `>` | `protein_sequence`* | `fasta` |
| Starts with `@` + identifier | `raw_reads` | `fastq` |
| Magic bytes `BAM\1` | `aligned_reads` | `bam` |
| Magic bytes `PAR1` | `tabular` | `parquet` |
| Consistent `,` or `\t` delimiter + header | `tabular` | `csv` or `tsv` |
| JSON `{...}` with short flat values | `tabular` | `jsonl` |
| JSON `{...}` with string values >200 chars | `text` | `jsonl` |
| Folder with `train/` `test/` `val/` subfolders | ambiguous | `files/folder` → ask user |

> *FASTA first-line sniff defaults to `protein_sequence`; confirm with alphabet detection (Step 4).

For JSONL: sniff **20–50 rows** (not just 5) before deciding `tabular` vs `text` — short documents can appear in the first few rows.

If two patterns match, prefer: `protein_sequence` > `nucleotide_sequence` > `raw_reads` > `tabular` > `text` > `multimodal`.

---

## Step 4 — FASTA alphabet detection

After loading FASTA sequences, determine `dataset_type` from sequence content:

```python
import re

PROTEIN_CHARS = set("ACDEFGHIKLMNPQRSTVWYBZXUO*-")
NUCLEOTIDE_CHARS = set("ACGTURYSWKMBDHVN-")

def detect_alphabet(sequences: list[str]) -> str:
    sample = "".join(sequences[:200]).upper()
    total = len(sample)
    if total == 0:
        return "unknown"
    pct_aa = sum(1 for c in sample if c in PROTEIN_CHARS) / total
    pct_nt = sum(1 for c in sample if c in NUCLEOTIDE_CHARS) / total
    if pct_aa > 0.90 and pct_nt < 0.70:
        return "protein"
    elif pct_nt > 0.90:
        return "dna"
    else:
        return "mixed"
```

| Alphabet result | dataset_type |
|---|---|
| `protein` (>90% AA chars, <70% nucleotide overlap) | `protein_sequence` |
| `dna` (>90% ACGTU chars) | `nucleotide_sequence` |
| `mixed` | `nucleotide_sequence` (warn) |

---

## Step 5 — Semantic column detection for tabular CSV/Parquet

Even after classifying as `tabular`, scan column names for hidden sequence content:

| Column name matches | Action |
|---|---|
| `sequence` `aa_seq` `protein_sequence` | Reclassify to `protein_sequence` if alphabet check passes |
| `dna` `nucleotide` `nt_seq` | Reclassify to `nucleotide_sequence` |
| `smiles` `mol` `inchi` | Flag as molecular data — tabular but note cheminformatics context |
| `patient_id` `donor_id` `subject_id` | Flag as clinical tabular — check for PHI |
| `well` `plate` `row` `col` or A1–H12 pattern | Flag as plate reader layout — suggest melt before analysis |
| `batch` `replicate` `experiment` | Flag batch column — recommend batch correction |

---

## Routing table

| dataset_type | format | Loader skill | Analysis skill |
|---|---|---|---|
| `protein_sequence` | hf_dataset / fasta | `load-fasta` | `analyze-protein-sequences` |
| `nucleotide_sequence` | hf_dataset / fasta (.fna) | `load-fasta` | `analyze-protein-sequences` (DNA subset) |
| `raw_reads` | fastq | `load-fastq` | `analyze-fastq-reads` |
| `aligned_reads` | bam / sam | `load-bam` | `analyze-bam-reads` |
| `tabular` | hf_dataset / parquet / csv / tsv / excel / jsonl | `load-parquet` / `load-csv` / `load-excel` / `load-jsonl` | `analyze-tabular` |
| `tabular` | asc | `load-asc` | `analyze-tabular` |
| `flow_cytometry` | fcs | `load-fcs` | `analyze-fcs` |
| `single_cell` | h5 / h5ad | `load-h5` | `analyze-single-cell` |
| `single_cell` | mtx | `load-mtx` | `analyze-single-cell` |
| `text` | hf_dataset / jsonl / txt | `load-jsonl` | `text-data-loader` *(future)* |
| `image` | tif / png / scn | `load-tif` | Out of PLM scope — stop and explain |
| `multimodal` | files / folder | — | Do not route — describe structure and ask user |

---

## Ambiguity rules

- **Two types match** → prefer more specific: `protein_sequence` > `nucleotide_sequence` > `raw_reads` > `tabular` > `text` > `multimodal`
- **No type matches** → set `loadability.status = warn`, list what was tried, ask the user
- **Downstream skill not yet implemented** → emit partial profile with `loadability.status = warn`; do not implement the missing skill inline
