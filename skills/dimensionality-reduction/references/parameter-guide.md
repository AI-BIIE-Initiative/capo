# Parameter Guide — Dimensionality Reduction

## PCA / IncrementalPCA
- `n_components`: Start with 30–100 for dense embeddings. Plot cumulative explained variance — 95% is a common threshold.
- PCA is deterministic when solver and data are fixed.
- IncrementalPCA: set `batch_size` to fit a batch in RAM. Results converge to standard PCA but may vary slightly.

## TruncatedSVD
- `n_components`: Same guidance as PCA. Default 50 is safe. Does not centre data — expected.
- Components are not orthogonal to the mean (unlike PCA). This is correct behaviour for TruncatedSVD.

## UMAP
- `n_neighbors` (default 15): Controls local vs global structure. Higher (30–50) → more global layout.
- `min_dist` (default 0.1): How tightly points are packed in 2D. Lower (0.01–0.05) → tighter clusters. Higher (0.5) → more spread.
- `metric`: Use `cosine` for embeddings; `euclidean` for PCA/SVD-reduced data.
- `n_components`: 2 for visualization. Use 10–50 for downstream clustering (`umap_intermediate` in clustering skill).
- Stochastic — always set `random_state`.

## t-SNE
- `perplexity` (default 30): Roughly the expected number of nearest neighbours. Try 50–100 for large datasets (>10k points).
- `learning_rate`: Use `"auto"` (sklearn ≥1.2) or 200.
- `init`: Use `"pca"` for more stable and reproducible embeddings (not `"random"`).
- `n_iter`: 1000 is usually enough. Increase to 2000 if loss has not converged.
- Always report the seed — two t-SNE runs with different seeds may look different even on identical data.
