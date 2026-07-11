---
name: dataset-profile-schema
description: Canonical output schema for the profiling-datasets skill. Designed for protein language model fine-tuning workloads. Every downstream profiling skill must return a JSON object conforming to this schema.
---

# Dataset Profile Schema

```jsonc
{
  // Detected source format
  "format": "hf_dataset | parquet | csv | tsv | fasta | fastq | bam | jsonl | fcs | h5ad | h5 | mtx | asc | unknown",

  // Inferred dataset modality
  "dataset_type": "protein_sequence | nucleotide_sequence | raw_reads | aligned_reads | tabular | flow_cytometry | single_cell | text | multimodal",

  // HuggingFace source metadata (nullable for local files)
  "hf_source": {
    "dataset_id": "your-org/dataset-name",
    "config_name": "default",              // HF config/subset name; null if not specified
    // split_names: list of available splits, or ["train"] if unsplit, or null if unknown
    "split_names": ["train", "validation", "test"],
    // num_rows_per_split: null if dataset has no splits or splits could not be resolved
    "num_rows_per_split": { "train": 8000, "validation": 1000, "test": 1000 },
    "hf_features": {                       // raw ds.features dict from HF
      "sequence": { "dtype": "string", "_type": "Value" },
      "label":    { "dtype": "float32", "_type": "Value" }
    }
  },

  // Dimensions
  "shape": {
    "rows": 10000,       // for tabular / text
    "cols": 12,          // for tabular (nullable for bio)
    "sequences": null    // for sequence types (nullable for tabular)
  },

  // Column / field schema
  "schema": [
    { "name": "sequence", "dtype": "string",  "nullable": false },
    { "name": "label",    "dtype": "float32", "nullable": true  }
  ],

  // Label information (nullable if no target column found)
  "label_info": {
    "target_columns": ["label"],

    // Biological task — what the label represents in protein science:
    //   fitness_prediction   — scalar or binary fitness/activity score from sequence
    //   variant_effect       — DMS / mutation scanning / ΔΔG / ΔTm
    //   binding_affinity     — protein-protein or protein-ligand binding score
    //   localization         — subcellular localization
    //   solubility           — solubility / expression level
    //   stability            — thermostability, Tm, ΔΔG unfolding
    //   enzyme_activity      — catalytic efficiency, kcat/Km, EC classification
    //   antigenicity         — immunogenicity, epitope prediction, MHC binding
    //   structure_class      — secondary/tertiary structure label (DSSP, SCOP, CATH)
    //   mlm_pretraining      — no labels; continued masked language modeling
    //   sequence_generation  — generative fine-tuning; target is another sequence
    //   zero_shot_scoring    — no fine-tuning; PLL/pseudo-log-likelihood scoring only
    //   embedding_only       — no labels; extract representations only
    //   unknown
    "task_type": "fitness_prediction",

    // ML formulation — how the loss and head are structured:
    //   binary_classification   — single sigmoid output, BCELoss
    //   multiclass_classification — softmax output, CrossEntropyLoss
    //   regression              — scalar output, MSELoss / Pearson target
    //   multilabel_classification — multiple sigmoid outputs, BCELoss per label
    //   sequence_to_sequence    — token-level or full-sequence generation
    //   masked_lm               — MLM objective, no external labels
    //   unknown
    "ml_formulation": "regression",

    "label_dtype": "float32 | int64 | string",
    "num_classes": null,   // populated for classification; null for regression
    "class_counts": null   // populated for discrete labels; null for regression
  },

  // Null / missing counts per column
  "missing": {
    "sequence": { "null_count": 0,  "null_pct": 0.0 },
    "label":    { "null_count": 42, "null_pct": 0.42 }
  },

  // Label distribution (nullable for regression / embedding_only / mlm_pretraining)
  "class_balance": {
    "0": { "count": 4800, "pct": 48.0 },
    "1": { "count": 5200, "pct": 52.0 }
  },

  // Basic stats for numeric label / feature columns
  "sample_stats": {
    "label": { "min": -3.2, "median": 0.1, "mean": 0.04, "max": 4.8, "std": 1.1 }
  },

  // Sequence-specific stats — required for protein_sequence / nucleotide_sequence / raw_reads
  // nullable for tabular, flow_cytometry, single_cell, text
  "sequence_stats": {
    "length_min": 50,
    "length_median": 312,
    "length_max": 1024,
    // read_type: what kind of sequence this is
    "read_type": "protein | dna | rna | mixed | unknown",
    // is_plm_ready: false if nucleotide_sequence, raw_reads, or aligned_reads
    "is_plm_ready": true,
    // PLM context: flag sequences that exceed common model limits
    "pct_over_512":  4.2,   // % sequences > 512 tokens (ESM2 default)
    "pct_over_1024": 0.8    // % sequences > 1024 tokens (ESM2 max for most variants)
  },

  // Split distribution — covers both HF splits and in-column split assignments
  // Edge cases:
  //   No splits at all       → null (single flat dataset, no train/val/test)
  //   Only "train" split     → { "train": 10000 } — note: no val/test, flag in warnings
  //   Split column present   → inferred from column values (e.g. "split", "fold", "set")
  //   HF splits + no column  → taken from ds.info.splits
  "split_info": {
    "source": "hf_splits | column:<name> | null",   // how splits were detected
    "splits": { "train": 8000, "val": 1000, "test": 1000 },  // null if no splits found
    // Homology-safety of the split. For protein_sequence MUST be one of:
    //   true   — splits confirmed cluster-aware (dataset card, cluster_id sibling column,
    //            or user answered "Cluster-aware" to the confirmation prompt)
    //   false  — splits are missing, train-only, OR random (user confirmed random / unknown)
    //   null   — ambiguous case awaiting user confirmation; needs_user_confirmation MUST be true
    // When false (final, after any user confirmation), Stage 4 MUST recommend
    // `clustering/mmseqs2` as the first preprocessing step. When null, Stage 4 MUST NOT
    // recommend mmseqs2 — the orchestrator finalises the value via AskUserQuestion first.
    "is_homology_safe": true,
    // needs_user_confirmation: true ONLY when a split column exists but no cluster_id /
    // family / group / fold sibling column accompanies it. The data-profiler emits
    // user_question; the orchestrator calls AskUserQuestion and updates is_homology_safe.
    "needs_user_confirmation": false,
    "user_question": null,   // populated only when needs_user_confirmation = true
    "evidence": "dataset card states 'UniRef30-clustered split'"
  },

  // Loadability check — always present
  "loadability": {
    "status": "ok | warn | fail",
    "parser": "datasets.load_dataset | pandas.read_csv | biopython.SeqIO | ...",
    "errors": []
  },

  // Data-quality and PLM-readiness warnings
  "warnings": [
    "42 rows missing label — 0.42% of dataset",
    "4.2% of sequences exceed 512 tokens — truncation needed for ESM2-650M",
    "Class imbalance detected: ratio 1:10 — consider weighted loss"
  ],

  // Plot file paths from Stage 3 analysis skill
  "plots": {
    "length_dist": "path/to/length_dist.png",
    "label_dist":  "path/to/label_dist.png"
  },

  // Stage 4 output
  // When split_info.is_homology_safe = false, preprocessing_skill MUST be "clustering/mmseqs2"
  // and "clustering/mmseqs2" MUST be preprocessing_steps[0].
  "preprocessing_skill": "protein-sequence-data-processing",   // or "clustering/mmseqs2" — null if not applicable
  "preprocessing_steps": ["deduplicate", "mask missing labels", "tokenize"],

  "routed_to": "analyze-protein-sequences"
}
```

---

## Constraints

- **Never** include raw data rows in the profile output
- **Always** include `loadability` — even if everything else is null
- `sequence_stats` must be non-null for `protein_sequence`, `nucleotide_sequence`, and `raw_reads`; null otherwise
- `sequence_stats.is_plm_ready` must be `false` for `nucleotide_sequence`, `raw_reads`, and `aligned_reads`
- `hf_source` must be non-null when input is a HF Hub dataset ID
- `class_balance` is required when `ml_formulation` is `binary_classification`, `multiclass_classification`, or `multilabel_classification`
- `class_balance` is null when `ml_formulation` is `regression`, `sequence_to_sequence`, or `masked_lm`
- Both `task_type` and `ml_formulation` must always be inferred independently — they are orthogonal
- `warnings` must be a list (empty `[]` is valid, `null` is not)
- Always flag sequence length overflow relative to ESM2 limits (512 / 1024) in `warnings` for `protein_sequence`
- If `split_info.splits` is null or contains only `"train"`, set `split_info.is_homology_safe = false`, `needs_user_confirmation = false`, and add to `warnings`: `"Splits missing or random — run skills/clustering/mmseqs2 to generate homology-safe train/val/test before training. Random splits leak homology and inflate metrics."`
- If splits were inferred from a column, record the column name in `split_info.source` (e.g. `"column:split"`). Then:
  - If an accompanying `cluster_id` / `family` / `group` / `fold` column is present → `is_homology_safe = true`, `needs_user_confirmation = false`
  - Otherwise → `is_homology_safe = null`, `needs_user_confirmation = true`, and populate `user_question` with the prompt the orchestrator will pass to `AskUserQuestion`
- The orchestrator finalises `is_homology_safe` (true/false) based on the user's answer; only then can Stage 4 routing be applied
- For `dataset_type = protein_sequence`, `preprocessing_skill` MUST be `"clustering/mmseqs2"` and `preprocessing_steps[0]` MUST be `"clustering/mmseqs2"` whenever the **finalised** `split_info.is_homology_safe = false`. While `needs_user_confirmation = true`, mmseqs2 MUST NOT be in `preprocessing_steps`.
- `sequence_stats.pct_over_512` and `pct_over_1024` are only meaningful for `protein_sequence` — set to null for other types
