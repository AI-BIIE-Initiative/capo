"""
Clustering evaluation metrics — computed in the analysis space, not in 2D visual space.

Internal metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) measure geometric
quality in the feature space the algorithm optimised. Use them for relative comparison
across parameter choices, not as biological ground truth.

Always follow with biological profiling (cluster_profiles.py) to verify enrichment.
"""
import numpy as np
import pandas as pd


def compute_cluster_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    sample_n: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Compute silhouette, Davies-Bouldin, Calinski-Harabasz, and size statistics.
    Excludes noise points (label == -1) from metric computation.

    Args:
        X:       Feature matrix in analysis space (pre-reduced).
        labels:  Cluster label array, aligned with X rows. -1 = noise.
        sample_n: Max rows for silhouette (O(n²) — sampled for speed).
        seed:    Random seed for sampling.

    Returns:
        dict with: n_clusters, n_noise, noise_frac, cluster_size_stats,
                   silhouette, davies_bouldin, calinski_harabasz, warnings.
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]
    n_clusters = len(set(labels_clean))
    n_noise = int((labels == -1).sum())
    noise_frac = n_noise / len(labels)

    # Cluster size statistics
    sizes = pd.Series(labels_clean).value_counts()
    size_stats = {
        "min":          int(sizes.min()),
        "max":          int(sizes.max()),
        "mean":         float(sizes.mean()),
        "median":       float(sizes.median()),
        "largest_frac": float(sizes.max() / len(labels)),
    }

    result = {
        "n_clusters":        n_clusters,
        "n_noise":           n_noise,
        "noise_frac":        round(noise_frac, 4),
        "cluster_size_stats": size_stats,
    }

    if n_clusters < 2 or len(X_clean) < 2:
        result.update({"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None})
        result["warnings"] = ["Less than 2 clusters found — metrics cannot be computed"]
        return result

    # Silhouette on a sample (O(n²))
    rng = np.random.default_rng(seed)
    n = min(sample_n, len(X_clean))
    idx = rng.choice(len(X_clean), n, replace=False)
    sil = silhouette_score(X_clean[idx], labels_clean[idx])

    result["silhouette"]        = round(float(sil), 4)
    result["davies_bouldin"]    = round(float(davies_bouldin_score(X_clean, labels_clean)), 4)
    result["calinski_harabasz"] = round(float(calinski_harabasz_score(X_clean, labels_clean)), 2)

    # Quality warnings
    warnings = []
    if size_stats["largest_frac"] > 0.8:
        warnings.append(
            f"Largest cluster = {size_stats['largest_frac']:.0%} of non-noise data — "
            "try increasing min_cluster_size or n_clusters"
        )
    if noise_frac > 0.2:
        warnings.append(
            f"High noise fraction: {noise_frac:.0%} — "
            "try lower min_cluster_size for HDBSCAN, or switch to KMeans"
        )
    if sil < 0:
        warnings.append("Silhouette < 0 — clusters are worse than random assignment")
    result["warnings"] = warnings

    return result
