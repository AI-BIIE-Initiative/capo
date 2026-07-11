"""
CLI entry point for protein sequence clustering.

Usage:
    python run_clustering.py --config config.yaml

Config file keys match ClusterConfig fields (see scripts/config_schema.py).

Minimal config example (HDBSCAN on pre-reduced features from dim-red skill):
    input_path: outputs/dimred/pre_reduced.npy
    metadata_path: data/sequences.csv
    id_col: id
    seq_col: sequence
    metadata_cols: [species, binding_label]
    algorithm: hdbscan
    algorithm_params:
      min_cluster_size: 50
      min_samples: 10
    output_dir: outputs/clustering/

Minimal config example (MiniBatchKMeans with K sweep):
    input_path: outputs/dimred/pre_reduced.npy
    metadata_path: data/sequences.csv
    algorithm: minibatch_kmeans
    algorithm_params:
      n_clusters: 30
    output_dir: outputs/clustering/
"""
import argparse
import time
from pathlib import Path

import yaml
import numpy as np
import pandas as pd


def main(cfg_path: str) -> None:
    from scripts.config_schema import ClusterConfig
    from scripts.io_utils import load_array, load_dataframe
    from scripts.precluster_reduction import reduce_for_clustering
    from scripts.clusterers import fit_clusterer
    from scripts.evaluate_clusters import compute_cluster_metrics
    from scripts.cluster_profiles import profile_clusters
    from scripts.cluster_splits import assign_cluster_splits, check_cluster_leakage
    from scripts.export_outputs import save_clustering

    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    cfg = ClusterConfig(**raw)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Load feature matrix
    if cfg.input_path.endswith(".npy"):
        X = load_array(cfg.input_path)
    else:
        df_feat = load_dataframe(cfg.input_path)
        skip_cols = {cfg.id_col, cfg.seq_col} | set(cfg.metadata_cols)
        feat_cols = [c for c in df_feat.columns if c not in skip_cols]
        X = df_feat[feat_cols].values

    print(f"Feature matrix shape: {X.shape}")

    # 2. Load metadata
    df = load_dataframe(cfg.metadata_path) if cfg.metadata_path else pd.DataFrame(
        {cfg.id_col: range(X.shape[0])}
    )

    # 3. Pre-reduce if needed
    X_cluster = reduce_for_clustering(X, cfg)
    if X_cluster.shape != X.shape:
        print(f"Pre-reduced to shape: {X_cluster.shape}")

    # 4. Cluster
    labels, model = fit_clusterer(X_cluster, cfg)
    df["cluster"] = labels
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"Clusters: {n_clusters} | Noise points: {n_noise} ({n_noise / len(labels):.1%})")

    # 5. Evaluate
    metrics = compute_cluster_metrics(X_cluster, labels, seed=cfg.seed)
    metrics["runtime_s"] = round(time.time() - t0, 1)
    print(
        f"Silhouette: {metrics.get('silhouette')} | "
        f"Davies-Bouldin: {metrics.get('davies_bouldin')} | "
        f"Calinski-Harabasz: {metrics.get('calinski_harabasz')}"
    )
    for w in metrics.get("warnings", []):
        print(f"WARNING: {w}")

    # 6. Biological profiling
    label_cols = [c for c in cfg.metadata_cols if c in df.columns]
    profiles_dir = f"{cfg.output_dir}/cluster_profiles"
    profiles = profile_clusters(
        df,
        cluster_col="cluster",
        label_cols=label_cols,
        seq_col=cfg.seq_col if cfg.seq_col in df.columns else None,
        out_dir=profiles_dir,
    )

    # 7. Cluster-aware splits
    if cfg.split_strategy != "none":
        df = assign_cluster_splits(
            df,
            cluster_col="cluster",
            ratios=tuple(cfg.split_ratios),
            seed=cfg.seed,
            stratify_col=cfg.stratify_col,
        )
        if cfg.seq_col in df.columns:
            leakage = check_cluster_leakage(df, seq_col=cfg.seq_col, cluster_col="cluster")
            metrics["leakage"] = leakage
            if leakage["sequences_in_multiple_splits"] > 0 or leakage["clusters_spanning_splits"] > 0:
                print(f"WARNING: Leakage detected — {leakage}")
        print(f"Split counts: {df['split'].value_counts().to_dict()}")

    # 8. Plot (requires 2D coords from dimensionality-reduction skill)
    coord_candidates = [
        Path(cfg.output_dir).parent / "dimred" / "reduced_coordinates.parquet",
        Path("outputs/dimred/reduced_coordinates.parquet"),
    ]
    coord_path = next((p for p in coord_candidates if p.exists()), None)

    if coord_path:
        from scripts.plotting import plot_cluster_map, plot_cluster_profiles
        coords = pd.read_parquet(coord_path)
        plots_dir = f"{cfg.output_dir}/cluster_plots"
        plot_cluster_map(
            coords, df, cluster_col="cluster",
            out_path=f"{plots_dir}/cluster_map.png",
        )
        for col in label_cols:
            plot_cluster_profiles(
                profiles, label_col=col,
                out_path=f"{plots_dir}/profiles_{col}.png",
            )
        print(f"Saved cluster plots to {plots_dir}")
    else:
        print(
            "2D coordinates not found. Run dimensionality-reduction skill first "
            "to generate reduced_coordinates.parquet for plotting."
        )

    # 9. Export
    save_clustering(df, profiles, metrics, cfg)
    print(f"Done. Outputs: {cfg.output_dir} | Runtime: {metrics['runtime_s']}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clustering on protein sequence features.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
