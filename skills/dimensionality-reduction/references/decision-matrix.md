# Decision Matrix — Which Reducer?

## By input type

| Input | First step | Why |
|-------|-----------|-----|
| One-hot encoded sequences (sparse) | `TruncatedSVD` | Sparse-safe, avoids dense centering |
| K-mer count matrix (sparse) | `TruncatedSVD` | Same |
| Dense ESM / ProtT5 embeddings | `PCA` | Standard centered linear reduction |
| Dense but large (>500k rows) | `IncrementalPCA` | Batch-wise memory-efficient PCA |

## By goal

| Goal | Final step |
|------|-----------|
| Exploration, reusable map | `UMAP` |
| Publication figure (strong local separation) | `t-SNE` on stratified sample |
| Project new sequences after initial fit | `UMAP` (supports `transform()`) |
| Interpretable linear summary only | Stop at PCA |

## Standard pipeline templates

```
Sparse one-hot / k-mer  →  TruncatedSVD(n=50)   →  UMAP(2D)
Dense ESM embedding     →  PCA(n=50)             →  UMAP(2D)
Large dense dataset     →  IncrementalPCA(n=50)  →  UMAP(2D)
Publication figure      →  PCA(n=50)  →  sample  →  t-SNE(2D)
Pre-clustering step     →  PCA/SVD(n=50)         →  [hand off to clustering skill]
```
