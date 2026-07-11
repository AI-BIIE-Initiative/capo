"""
Linear dimensionality reducers: PCA, IncrementalPCA, TruncatedSVD.

Selection guide:
- Sparse input (one-hot, k-mer)    → fit_truncated_svd  (avoids dense centering)
- Dense input, fits in RAM         → fit_pca
- Dense input, large dataset       → fit_incremental_pca
"""
import numpy as np
import scipy.sparse as sp
from typing import Any, Tuple


def fit_truncated_svd(
    X,
    n_components: int = 50,
    seed: int = 42,
) -> Tuple[Any, np.ndarray]:
    """
    Fit TruncatedSVD. Preferred for sparse matrices (one-hot, k-mer).
    Does not center the data — safe for sparse inputs.
    Returns (fitted_reducer, X_reduced).
    """
    from sklearn.decomposition import TruncatedSVD

    n_components = min(n_components, X.shape[1] - 1)
    reducer = TruncatedSVD(n_components=n_components, random_state=seed)
    X_reduced = reducer.fit_transform(X)
    return reducer, X_reduced


def fit_pca(
    X: np.ndarray,
    n_components: int = 50,
    seed: int = 42,
) -> Tuple[Any, np.ndarray]:
    """
    Fit standard PCA. Preferred for dense embeddings (ESM, ProtT5, etc.).
    Input X must be dense. Raises if sparse input is provided.
    Returns (fitted_reducer, X_reduced).
    """
    from sklearn.decomposition import PCA

    if sp.issparse(X):
        raise ValueError(
            "PCA requires a dense input array. "
            "Use fit_truncated_svd() for sparse matrices (one-hot, k-mer)."
        )
    n_components = min(n_components, X.shape[1], X.shape[0] - 1)
    reducer = PCA(n_components=n_components, random_state=seed)
    X_reduced = reducer.fit_transform(X)
    return reducer, X_reduced


def fit_incremental_pca(
    X: np.ndarray,
    n_components: int = 50,
    batch_size: int = 1000,
) -> Tuple[Any, np.ndarray]:
    """
    Fit IncrementalPCA in minibatches. Use when dense X does not fit in RAM.
    Returns (fitted_reducer, X_reduced).
    """
    from sklearn.decomposition import IncrementalPCA

    n_components = min(n_components, X.shape[1])
    reducer = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    for start in range(0, X.shape[0], batch_size):
        reducer.partial_fit(X[start : start + batch_size])
    X_reduced = np.vstack([
        reducer.transform(X[start : start + batch_size])
        for start in range(0, X.shape[0], batch_size)
    ])
    return reducer, X_reduced


def transform_linear(reducer, X) -> np.ndarray:
    """Transform new data using a previously fitted linear reducer."""
    return reducer.transform(X)
