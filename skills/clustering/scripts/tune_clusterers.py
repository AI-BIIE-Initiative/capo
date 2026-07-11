"""
Parameter sweeps for clustering algorithms.

Use before committing to a final model to identify good hyperparameters.
Results are returned as DataFrames — save them and plot silhouette vs parameter.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def sweep_k(
    X: np.ndarray,
    k_range=range(5, 50, 5),
    seed: int = 42,
    sample_n: int = 5000,
    mini_batch: bool = True,
) -> pd.DataFrame:
    """
    Sweep over K for KMeans/MiniBatchKMeans.
    Returns DataFrame with columns: k, inertia, silhouette.

    Args:
        X:         Feature matrix in analysis space (already pre-reduced).
        k_range:   Range of K values to try.
        seed:      Random seed.
        sample_n:  Max rows for silhouette computation (O(n²) — sample for speed).
        mini_batch: Use MiniBatchKMeans (faster) if True.
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans

    rng = np.random.default_rng(seed)
    n = min(sample_n, X.shape[0])
    idx = rng.choice(X.shape[0], n, replace=False)
    X_sil = X[idx]

    rows = []
    for k in k_range:
        Cls = MiniBatchKMeans if mini_batch else KMeans
        model = Cls(n_clusters=k, random_state=seed, n_init=3)
        labels_all = model.fit_predict(X)
        labels_sil = labels_all[idx]
        n_valid_classes = len(set(labels_sil) - {-1})
        sil = (
            silhouette_score(X_sil, labels_sil)
            if n_valid_classes > 1
            else float("nan")
        )
        row = {"k": k, "inertia": model.inertia_, "silhouette": round(sil, 4)}
        rows.append(row)
        print(f"  k={k:3d}  inertia={model.inertia_:.1f}  silhouette={sil:.4f}")

    return pd.DataFrame(rows)


def sweep_hdbscan(
    X: np.ndarray,
    min_sizes: list[int] | None = None,
    min_samples_values: list[int] | None = None,
    sample_n: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sweep over HDBSCAN min_cluster_size and min_samples.
    Returns DataFrame with: min_cluster_size, min_samples, n_clusters, noise_frac, silhouette.

    Higher min_samples → more conservative clusters, higher noise fraction.
    Target noise_frac < 0.20 for most biological datasets.
    """
    if min_sizes is None:
        min_sizes = [10, 25, 50, 100, 200]
    if min_samples_values is None:
        min_samples_values = [5, 10]

    rng = np.random.default_rng(seed)
    n = min(sample_n, X.shape[0])
    idx = rng.choice(X.shape[0], n, replace=False)
    X_sil = X[idx]

    rows = []
    for mcs in min_sizes:
        for ms in min_samples_values:
            try:
                import hdbscan as hdbscan_lib
                model = hdbscan_lib.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
            except ImportError:
                from sklearn.cluster import HDBSCAN
                model = HDBSCAN(min_cluster_size=mcs, min_samples=ms)

            labels = model.fit_predict(X)
            n_noise = int((labels == -1).sum())
            noise_frac = n_noise / len(labels)
            n_clusters = len(set(labels) - {-1})

            labels_sil = labels[idx]
            valid = labels_sil != -1
            sil = (
                silhouette_score(X_sil[valid], labels_sil[valid])
                if valid.sum() > 1 and len(set(labels_sil[valid])) > 1
                else float("nan")
            )

            row = {
                "min_cluster_size": mcs,
                "min_samples": ms,
                "n_clusters": n_clusters,
                "noise_frac": round(noise_frac, 3),
                "silhouette": round(sil, 4),
            }
            rows.append(row)
            print(
                f"  mcs={mcs:3d} ms={ms:2d}  "
                f"n_clusters={n_clusters:3d}  noise={noise_frac:.1%}  sil={sil:.4f}"
            )

    return pd.DataFrame(rows)
