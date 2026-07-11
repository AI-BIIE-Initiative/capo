"""
Pre-reduction step before clustering.

Maps the feature matrix to a compact analysis space before applying a clusterer.
The correct choice depends on input type:
  - Sparse (one-hot, k-mer) → truncated_svd  (avoids dense centering)
  - Dense (ESM embeddings)  → pca
  - Already reduced         → none

WARNING: Do not cluster on t-SNE output — see references/pitfalls.md.
WARNING: Clustering on UMAP 2D output distorts density. Use umap_intermediate
         (10–50 dims) only if explicitly requested, never as a silent default.
"""
import numpy as np
import scipy.sparse as sp


def reduce_for_clustering(X, cfg) -> np.ndarray:
    """
    Apply pre-reduction based on cfg.pre_reducer.
    Returns a dense numpy array ready for a clustering algorithm.

    cfg.pre_reducer options:
        "none"             — pass through (densify if sparse)
        "truncated_svd"    — TruncatedSVD, sparse-safe
        "pca"              — standard PCA, dense input only
        "umap_intermediate"— UMAP to n_components > 2, explicit opt-in only
    """
    if cfg.pre_reducer == "none":
        return X.toarray() if sp.issparse(X) else np.asarray(X, dtype=np.float32)

    if cfg.pre_reducer == "truncated_svd":
        from sklearn.decomposition import TruncatedSVD
        n = min(cfg.pre_reducer_n, X.shape[1] - 1)
        reducer = TruncatedSVD(n_components=n, random_state=cfg.seed)
        return reducer.fit_transform(X)

    if cfg.pre_reducer == "pca":
        from sklearn.decomposition import PCA
        if sp.issparse(X):
            raise ValueError(
                "PCA requires dense input. Use pre_reducer='truncated_svd' for sparse data."
            )
        n = min(cfg.pre_reducer_n, X.shape[1], X.shape[0] - 1)
        reducer = PCA(n_components=n, random_state=cfg.seed)
        return reducer.fit_transform(X)

    if cfg.pre_reducer == "umap_intermediate":
        from umap import UMAP
        # Use more dimensions than 2 — this is not a visualization step
        n_components = cfg.algorithm_params.get("umap_n_components", 10)
        X_dense = X.toarray() if sp.issparse(X) else X
        print(
            f"INFO: Clustering on UMAP intermediate space ({n_components} dims). "
            "Verify this is intentional — density and distances are affected by UMAP hyperparameters."
        )
        reducer = UMAP(n_components=n_components, random_state=cfg.seed)
        return reducer.fit_transform(X_dense)

    raise ValueError(
        f"Unknown pre_reducer: {cfg.pre_reducer!r}. "
        "Choose: pca | truncated_svd | none | umap_intermediate"
    )
