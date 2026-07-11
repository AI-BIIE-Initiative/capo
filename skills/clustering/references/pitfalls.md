# Common Pitfalls — Clustering

## 1. Clustering on t-SNE 2D coordinates
Never do this. t-SNE distorts global structure to preserve local neighborhoods.
Cluster boundaries in 2D t-SNE space are artifacts of the projection, not real structure.
Cluster in PCA/SVD analysis space; project to t-SNE for visualization only.

## 2. Clustering on UMAP 2D by default
UMAP preserves more global structure than t-SNE but density and geometry are still shaped by hyperparameters.
If you cluster on UMAP space, use intermediate dimensions (10–50, not 2).
This must be explicit — set `pre_reducer = "umap_intermediate"` in ClusterConfig.
It is never the silent default.

## 3. Relying on silhouette alone
Silhouette measures geometric compactness in the feature space the algorithm optimised.
It does not measure biological usefulness. Always follow with `profile_clusters()`.

## 4. Discarding noise points silently
HDBSCAN labels outlier points as -1. These are not garbage.
Noise points often represent rare or ambiguous sequences — inspect and report them.
Never drop them without counting and reporting.

## 5. Using random row splits after clustering
If you cluster and then split rows randomly, sequences from the same cluster will leak
across train and test, inflating evaluation metrics.
Always use `assign_cluster_splits()` to keep entire clusters in one split.

## 6. Forgetting to set a seed
KMeans, GMM, and HDBSCAN (via approximation algorithms) have stochastic components.
Always set `seed` in ClusterConfig and include it in the run report.

## 7. Over-trusting internal metrics in high dimensions
Davies-Bouldin and Calinski-Harabasz degrade in high-dimensional spaces.
Use them for relative comparisons across parameter choices, not as absolute quality scores.

## 8. Not checking cluster size distribution
A dataset with one cluster containing 95% of points and many size-1 clusters is not useful.
Always report `cluster_size_stats.largest_frac` and flag if > 0.8.
