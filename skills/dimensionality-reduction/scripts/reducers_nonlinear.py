"""
Nonlinear dimensionality reducers: UMAP and t-SNE.

Both are for visualization only — axes have no physical meaning.
Do not use these outputs as the analysis space for clustering or statistics.

UMAP:  fast, scalable, supports transform() on new data, more stable global structure.
t-SNE: strong local separation, slow, stochastic, no transform() on new data.
       Only run on sampled subsets (< 100k rows).
"""
import numpy as np
from typing import Any, Tuple


def run_umap(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
    seed: int = 42,
    **kwargs,
) -> Tuple[np.ndarray, Any]:
    """
    Run UMAP. Returns (coords, fitted_umap_model).
    The fitted model supports transform() on new data.

    Args:
        X:           Pre-reduced dense array (output of PCA/SVD step).
        n_components: 2 for visualization, 10–50 for intermediate clustering space.
        n_neighbors: Controls local vs global structure. Higher → more global. Try 15–50.
        min_dist:    How tightly points cluster in 2D. Lower (0.01) → tighter. Higher (0.5) → spread.
        metric:      "cosine" for embeddings; "euclidean" for PCA/SVD-reduced data.
        seed:        Random state for reproducibility.
    """
    from umap import UMAP

    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        **kwargs,
    )
    coords = reducer.fit_transform(X)
    return coords, reducer


def run_tsne(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float | str = "auto",
    n_iter: int = 1000,
    init: str = "pca",
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """
    Run t-SNE. Returns 2D coordinates.

    WARNING: t-SNE does not support transform() on new data.
    WARNING: Do not cluster on t-SNE output.
    WARNING: Only run on sampled subsets — slow above 100k rows.
    WARNING: Stochastic — always set seed and report it.

    Args:
        X:           Pre-reduced dense array. Input shape recommended < 100k rows.
        perplexity:  Expected local neighbourhood size. Try 30–100 for large datasets.
        init:        "pca" (recommended, more stable) or "random".
        learning_rate: "auto" lets sklearn choose based on n_samples (sklearn ≥ 1.2).
    """
    from sklearn.manifold import TSNE

    if len(X) > 100_000:
        print(
            f"WARNING: t-SNE on {len(X):,} rows is very slow. "
            "Sample to < 100k rows first using scripts/sampling.py."
        )

    reducer = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        init=init,
        random_state=seed,
        **kwargs,
    )
    return reducer.fit_transform(X)
