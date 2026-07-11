"""
Unified clustering interface.

All functions accept a dense numpy array X and return (labels, fitted_model).
Noise points (HDBSCAN, DBSCAN) are labeled -1.
Use evaluate_clusters.compute_cluster_metrics() after fitting.
"""
import numpy as np
from typing import Any, Tuple


def fit_clusterer(X: np.ndarray, cfg) -> Tuple[np.ndarray, Any]:
    """
    Dispatch to the appropriate clusterer based on cfg.algorithm.
    Returns (labels array, fitted model).
    """
    algo = cfg.algorithm
    params = dict(cfg.algorithm_params)
    # Remove umap_n_components — it belongs to precluster_reduction, not clusterers
    params.pop("umap_n_components", None)

    dispatch = {
        "hdbscan":          lambda: fit_hdbscan(X, **params),
        "kmeans":           lambda: fit_kmeans(X, seed=cfg.seed, **params),
        "minibatch_kmeans": lambda: fit_minibatch_kmeans(X, seed=cfg.seed, **params),
        "dbscan":           lambda: fit_dbscan(X, **params),
        "agglomerative":    lambda: fit_agglomerative(X, **params),
        "gmm":              lambda: fit_gmm(X, seed=cfg.seed, **params),
    }
    if algo not in dispatch:
        raise ValueError(
            f"Unknown algorithm: {algo!r}. "
            "Choose: hdbscan | kmeans | minibatch_kmeans | dbscan | agglomerative | gmm"
        )
    return dispatch[algo]()


def fit_hdbscan(
    X: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    **kwargs,
) -> Tuple[np.ndarray, Any]:
    """
    HDBSCAN: best default for biological sequence data with varying density.
    Noise points are labeled -1. Tune min_cluster_size and min_samples with
    tune_clusterers.sweep_hdbscan().
    """
    try:
        import hdbscan
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples, **kwargs
        )
    except ImportError:
        from sklearn.cluster import HDBSCAN
        model = HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples, **kwargs
        )
    labels = model.fit_predict(X)
    return labels, model


def fit_kmeans(
    X: np.ndarray,
    n_clusters: int = 20,
    seed: int = 42,
    n_init: int = 10,
    **kwargs,
) -> Tuple[np.ndarray, Any]:
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init, **kwargs)
    return model.fit_predict(X), model


def fit_minibatch_kmeans(
    X: np.ndarray,
    n_clusters: int = 20,
    seed: int = 42,
    **kwargs,
) -> Tuple[np.ndarray, Any]:
    from sklearn.cluster import MiniBatchKMeans
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, **kwargs)
    return model.fit_predict(X), model


def fit_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 10,
    **kwargs,
) -> Tuple[np.ndarray, Any]:
    from sklearn.cluster import DBSCAN
    model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    return model.fit_predict(X), model


def fit_agglomerative(
    X: np.ndarray,
    n_clusters: int = 20,
    linkage: str = "ward",
    **kwargs,
) -> Tuple[np.ndarray, Any]:
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, **kwargs)
    return model.fit_predict(X), model


def fit_gmm(
    X: np.ndarray,
    n_components: int = 20,
    covariance_type: str = "full",
    seed: int = 42,
    **kwargs,
) -> Tuple[np.ndarray, Any]:
    from sklearn.mixture import GaussianMixture
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=seed,
        **kwargs,
    )
    model.fit(X)
    return model.predict(X), model
