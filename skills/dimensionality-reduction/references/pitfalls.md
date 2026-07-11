# Common Pitfalls — Dimensionality Reduction

## 1. Interpreting t-SNE or UMAP axes biologically
Axes have no physical meaning. A sequence being "on the right" tells you nothing about evolution, fitness, or function.
Only local relative positions within a single run carry signal.

## 2. Concluding evolutionary distance from 2D distance
Two islands being far apart in t-SNE does not mean they are evolutionarily distant.
t-SNE distorts global distances to preserve local neighborhoods. UMAP is better but still non-linear.

## 3. Clustering on t-SNE coordinates
Never cluster on 2D t-SNE output. The distortion of global structure makes cluster boundaries unreliable.
Cluster in PCA/SVD analysis space; project to t-SNE for visualization.

## 4. Clustering on UMAP 2D without care
UMAP preserves more global structure than t-SNE, but density and geometry are still affected by hyperparameters.
UMAP's own documentation treats clustering-on-UMAP as useful but requiring care.
If you cluster on UMAP, use intermediate dimensions (10–50, not 2) — see `umap_intermediate` in clustering skill.

## 5. Running t-SNE on a full large dataset
t-SNE is too slow and memory-heavy for >100k rows. Always sample first.
Use `stratified_sample()` to maintain class balance in the sample.

## 6. Using a single seed and presenting it as ground truth
Both t-SNE and UMAP are stochastic. Run multiple seeds and verify that biological patterns are consistent.
If patterns change across seeds, the signal may be weak.

## 7. Not saving metadata with coordinates
2D coordinates without IDs and metadata are useless for downstream work.
Always save `reduced_coordinates.parquet` with id + all metadata columns.

## 8. PCA on a sparse one-hot matrix
sklearn PCA centres the data before SVD, which densifies the sparse matrix in memory.
For a 10k × 200k one-hot matrix this can require hundreds of GB.
Use `TruncatedSVD` instead.
