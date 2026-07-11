"""
Evaluation metrics for dimensionality reduction.

Use these in the analysis space — before the final 2D visual step.
They measure how much information is preserved, not how the plot looks.
"""
import numpy as np


def eval_linear_reduction(reducer) -> dict:
    """
    Report explained variance for a fitted PCA or TruncatedSVD reducer.
    Returns per-component and cumulative explained variance.
    """
    evr = reducer.explained_variance_ratio_
    return {
        "explained_variance_per_component": evr.tolist(),
        "total_explained_variance": float(evr.sum()),
        "n_components": len(evr),
    }


def eval_neighborhood(
    X_high: np.ndarray,
    X_low: np.ndarray,
    n_neighbors: int = 15,
    sample_n: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Compute trustworthiness: how well local neighborhoods from high-dim space
    are preserved in the low-dim embedding. Range [0, 1], higher is better.

    Samples at most sample_n rows for speed (trustworthiness is O(n²)).

    Args:
        X_high:     High-dimensional representation (e.g. PCA/SVD output).
        X_low:      Low-dimensional embedding (UMAP/t-SNE output).
        n_neighbors: Neighbourhood size to evaluate.
        sample_n:   Maximum rows to use for computation.
    """
    from sklearn.manifold import trustworthiness

    rng = np.random.default_rng(seed)
    n = min(sample_n, X_high.shape[0])
    idx = rng.choice(X_high.shape[0], n, replace=False)
    score = trustworthiness(X_high[idx], X_low[idx], n_neighbors=n_neighbors)
    return {
        "trustworthiness": float(score),
        "trustworthiness_n_neighbors": n_neighbors,
        "trustworthiness_sample_n": n,
    }
