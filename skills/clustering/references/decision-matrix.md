# Decision Matrix — Which Clustering Algorithm?

## By biological context

| Situation | Recommended |
|-----------|------------|
| Protein embeddings, density varies, outliers expected | HDBSCAN |
| Large collection (>500k), K is approximate | MiniBatchKMeans |
| Dense ESM embeddings, want K control | KMeans |
| Need hierarchy / nested grouping | AgglomerativeClustering (on centroids) |
| Mixed / ambiguous sequence families | GaussianMixture |
| Clean dataset, uniform density | DBSCAN |

## Standard biological pipelines

```
ESM embeddings (dense)    →  PCA(50)            →  HDBSCAN
One-hot / k-mer (sparse)  →  TruncatedSVD(50)   →  HDBSCAN or MiniBatchKMeans
Large collection (>1M)    →  TruncatedSVD(50)   →  MiniBatchKMeans (sweep K)
Family / subfamily tree   →  PCA(50)            →  AgglomerativeClustering on centroids
Ambiguous boundaries      →  PCA(50)            →  GaussianMixture
```

## K selection (KMeans)

1. Run `sweep_k()` over range (e.g. 5–100 in steps of 5)
2. Plot silhouette vs K — pick local maximum
3. Plot inertia — look for elbow
4. Verify biological interpretability (species / lineage per cluster)

## HDBSCAN parameter selection

1. Run `sweep_hdbscan()` over `min_cluster_size` = [10, 25, 50, 100, 200]
2. Target noise fraction < 20%
3. Increase `min_cluster_size` if too many tiny clusters; decrease if noise is too high
4. `min_samples` controls conservatism — higher = more noise, tighter clusters
