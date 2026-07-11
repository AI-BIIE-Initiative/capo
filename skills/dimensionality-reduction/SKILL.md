---
name: dimensionality-reduction
description: Reduce high-dimensional protein representations (one-hot, k-mer, ESM embeddings) to lower dimensions for analysis and visualization. Produces 2D/3D maps, compressed feature matrices, and evaluation reports. Operates downstream of profiling-datasets and protein-sequence-data-processing.
user-invokable: true
compatibility: scikit-learn ≥1.3, umap-learn ≥0.5, numpy ≥1.23, pandas ≥1.5, matplotlib ≥3.6. ESM backends require torch ≥2.0 and fair-esm.
---

# Dimensionality Reduction

> **Design principle:** Use PCA/SVD as the analysis layer and t-SNE/UMAP as the visual layer. Never treat a 2D plot as the analysis itself. Cluster and compute statistics in the analysis space, not in the 2D visual space.

---

## When to use

- Visualizing protein sequence datasets coloured by species, binding label, lineage, or mutation load
- Compressing high-dimensional embeddings or one-hot matrices before clustering
- Generating publication-quality 2D scatter maps of sequence space
- Comparing multiple biological conditions in a shared embedding space

## When not to use

| Want… | Use instead |
|---|---|
| Cluster sequences | `skills/clustering/` |
| Load / profile data | `profiling-datasets` |
| Preprocess sequences | `protein-sequence-data-processing` |
| Fine-tune a protein LM | `huggingface-jobs` |

---

## Decision table

| Technique | Input | Use when | Role |
|---|---|---|---|
| `TruncatedSVD` | sparse one-hot / k-mer | Large sparse matrix | Default first step for sparse |
| `PCA` | dense embeddings | Dense ESM/PLM embeddings | Default first step for dense |
| `IncrementalPCA` | dense, large | Does not fit in RAM | Batch-mode alternative to PCA |
| `UMAP` | any compressed | General exploration, reusable map | Default nonlinear visualizer |
| `t-SNE` | small / sampled subset | Final publication figure | Strong local separation figure |

**Safe default pipelines:**
- Sparse one-hot / k-mer → `TruncatedSVD(n_components=50)` → `UMAP(n_components=2)`
- Dense ESM embedding → `PCA(n_components=50)` → `UMAP(n_components=2)`
- Publication figure → PCA/SVD → stratified sample → `t-SNE`

---

## Configuration

```python
from scripts.config_schema import DimRedConfig
cfg = DimRedConfig(
    input_path            = "data/sequences.csv",
    sequence_col          = "sequence",
    id_col                = "id",
    metadata_cols         = ["species", "binding_label"],
    feature_type          = "onehot_sparse",   # onehot_sparse | onehot_dense | kmer | embedding | precomputed
    pre_reducer           = "truncated_svd",   # pca | incremental_pca | truncated_svd | none
    pre_reducer_n         = 50,
    visual_reducer        = "umap",            # umap | tsne | none
    visual_reducer_params = {"n_neighbors": 30, "min_dist": 0.1, "metric": "cosine"},
    sample_strategy       = "stratified",      # random | stratified | none
    sample_n              = 20_000,
    sample_col            = "species",
    seed                  = 42,
    output_dir            = "outputs/dimred/",
)
```

---

## Workflow

Import from `scripts/`. Do not re-implement inline.

### Step 1 — Load and validate
```python
from scripts.io_utils import load_dataframe, validate_columns
df = load_dataframe(cfg.input_path)
validate_columns(df, required=[cfg.sequence_col, cfg.id_col])
```

### Step 2 — Build representation
```python
from scripts.feature_builders import build_onehot_sparse, build_kmer_matrix
from scripts.embedding_backends import compute_esm_embeddings

if cfg.feature_type == "onehot_sparse":
    X = build_onehot_sparse(df[cfg.sequence_col], max_len=cfg.max_len)
elif cfg.feature_type == "kmer":
    X = build_kmer_matrix(df[cfg.sequence_col], k=cfg.kmer_k)
elif cfg.feature_type == "embedding":
    X = compute_esm_embeddings(df[cfg.sequence_col], model_name=cfg.embedding_model,
                               batch_size=cfg.embedding_batch_size)
```

### Step 3 — Pre-reduce (linear)
```python
from scripts.reducers_linear import fit_truncated_svd, fit_pca, fit_incremental_pca

# Sparse input → TruncatedSVD; dense → PCA
reducer_lin, X_pre = fit_truncated_svd(X, n_components=cfg.pre_reducer_n, seed=cfg.seed)
# or: reducer_lin, X_pre = fit_pca(X, ...)
# or: reducer_lin, X_pre = fit_incremental_pca(X, ...)
```

### Step 4 — Sample (for large datasets)
```python
from scripts.sampling import stratified_sample, random_sample
idx = stratified_sample(df, label_col=cfg.sample_col, n=cfg.sample_n, seed=cfg.seed)
X_sample, df_sample = X_pre[idx], df.iloc[idx].reset_index(drop=True)
```

### Step 5 — Visual reduction (nonlinear)
```python
from scripts.reducers_nonlinear import run_umap, run_tsne
coords_2d, umap_model = run_umap(X_sample, seed=cfg.seed, **cfg.visual_reducer_params)
# or: coords_2d = run_tsne(X_sample, seed=cfg.seed)  # use on sample only
```

### Step 6 — Evaluate
```python
from scripts.evaluate_reduction import eval_linear_reduction, eval_neighborhood
report = eval_linear_reduction(reducer_lin)      # explained variance for PCA/SVD
report.update(eval_neighborhood(X_sample, coords_2d, seed=cfg.seed))
```

### Step 7 — Plot
```python
from scripts.plotting import scatter_2d_multi
scatter_2d_multi(coords_2d, df_sample, color_cols=cfg.metadata_cols, out_dir=cfg.output_dir)
```

### Step 8 — Export
```python
from scripts.export_outputs import save_reduction
save_reduction(coords_2d, df_sample, reducer_lin, report, cfg, umap_model=umap_model)
```

---

## Output contract

```
outputs/dimred/
├── reduced_coordinates.parquet   # id, dim_0, dim_1 + all metadata_cols
├── pre_reduced.npy               # linear analysis matrix (n × pre_reducer_n)
├── linear_reducer.joblib         # fitted PCA or TruncatedSVD (for transform on new data)
├── umap_model.joblib             # fitted UMAP model (supports transform on new data)
├── reduction_report.json         # explained_variance, trustworthiness, runtime, seed
├── run_config.yaml               # full config snapshot
├── scatter_<col>.png / .pdf      # one figure per metadata_col
```

---

## Hard constraints

- **Never** use t-SNE output coordinates for clustering — use PCA/UMAP analysis space
- **Never** interpret t-SNE or UMAP axis values biologically (axes have no physical meaning)
- **Never** run t-SNE on more than 100k rows without sampling first
- **Never** apply PCA to a sparse matrix — use TruncatedSVD (PCA would densify the matrix)
- **Always** record seed in `reduction_report.json`
- **Always** save `reduced_coordinates.parquet` with all metadata columns alongside 2D coords
- **Always** export both PNG and PDF for publication figures
- **Always** report explained variance when a linear pre-reducer is used
