from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DimRedConfig:
    # I/O
    input_path: str = ""
    output_dir: str = "outputs/dimred/"

    # Column mapping
    sequence_col: str = "sequence"
    id_col: str = "id"
    metadata_cols: list = field(default_factory=list)

    # Representation
    feature_type: str = "onehot_sparse"   # onehot_sparse | onehot_dense | kmer | embedding | precomputed
    max_len: Optional[int] = None
    kmer_k: int = 3
    embedding_model: str = "esm2_t6"      # esm2_t6 | esm2_t12 | esm2_t30 | esm2_t33 | esm2_t36
    embedding_layer: Optional[int] = None  # None = last layer for the model
    embedding_batch_size: int = 32

    # Linear pre-reducer (analysis space)
    pre_reducer: str = "truncated_svd"    # pca | incremental_pca | truncated_svd | none
    pre_reducer_n: int = 50
    incremental_pca_batch_size: int = 1000

    # Nonlinear visual reducer
    visual_reducer: str = "umap"          # umap | tsne | none
    visual_reducer_params: dict = field(default_factory=lambda: {
        "n_neighbors": 30, "min_dist": 0.1, "metric": "cosine"
    })
    n_visual_dims: int = 2

    # Sampling (before nonlinear step)
    sample_strategy: str = "stratified"  # random | stratified | none
    sample_n: Optional[int] = 20_000
    sample_col: Optional[str] = None     # column to stratify sampling by

    # Reproducibility
    seed: int = 42
