from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClusterConfig:
    # I/O
    input_path: str = ""           # pre-reduced .npy or raw feature file
    metadata_path: str = ""        # CSV/Parquet with sequence metadata
    output_dir: str = "outputs/clustering/"

    # Column mapping
    id_col: str = "id"
    seq_col: str = "sequence"
    metadata_cols: list = field(default_factory=list)

    # Pre-reduction (applied before clustering, if input is not already reduced)
    pre_reducer: str = "none"      # pca | truncated_svd | none | umap_intermediate
    pre_reducer_n: int = 50

    # Clustering algorithm
    algorithm: str = "hdbscan"     # hdbscan | kmeans | minibatch_kmeans | dbscan | agglomerative | gmm
    algorithm_params: dict = field(default_factory=dict)

    # Cluster-aware split assignment
    split_strategy: str = "whole_cluster"  # whole_cluster | stratified_cluster | none
    split_ratios: tuple = (0.8, 0.1, 0.1)
    stratify_col: Optional[str] = None     # column to balance across splits

    # Reproducibility
    seed: int = 42
