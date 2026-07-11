"""
CLI entry point for dimensionality reduction.

Usage:
    python run_dimensionality_reduction.py --config config.yaml

Config file keys match DimRedConfig fields (see scripts/config_schema.py).

Minimal config example (sparse one-hot → UMAP):
    input_path: data/sequences.csv
    sequence_col: sequence
    id_col: id
    metadata_cols: [species, binding_label]
    feature_type: onehot_sparse
    pre_reducer: truncated_svd
    visual_reducer: umap
    output_dir: outputs/dimred/

Minimal config example (ESM embeddings → UMAP):
    input_path: data/sequences.csv
    feature_type: embedding
    embedding_model: esm2_t6
    pre_reducer: pca
    visual_reducer: umap
    output_dir: outputs/dimred/
"""
import argparse
import time
from pathlib import Path

import yaml
import numpy as np
import pandas as pd


def main(cfg_path: str) -> None:
    from scripts.config_schema import DimRedConfig
    from scripts.io_utils import load_dataframe, load_array, save_array, validate_columns
    from scripts.feature_builders import build_onehot_sparse, build_onehot_dense, build_kmer_matrix
    from scripts.embedding_backends import compute_esm_embeddings
    from scripts.reducers_linear import fit_truncated_svd, fit_pca, fit_incremental_pca
    from scripts.reducers_nonlinear import run_umap, run_tsne
    from scripts.sampling import stratified_sample, random_sample
    from scripts.evaluate_reduction import eval_linear_reduction, eval_neighborhood
    from scripts.plotting import scatter_2d_multi
    from scripts.export_outputs import save_reduction

    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    cfg = DimRedConfig(**raw)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Load
    df = load_dataframe(cfg.input_path)
    validate_columns(df, required=[cfg.sequence_col, cfg.id_col])
    print(f"Loaded {len(df):,} rows from {cfg.input_path}")

    # 2. Build representation
    ft = cfg.feature_type
    if ft == "onehot_sparse":
        X = build_onehot_sparse(df[cfg.sequence_col], max_len=cfg.max_len)
        default_pre = "truncated_svd"
    elif ft == "onehot_dense":
        X = build_onehot_dense(df[cfg.sequence_col], max_len=cfg.max_len)
        default_pre = "pca"
    elif ft == "kmer":
        X = build_kmer_matrix(df[cfg.sequence_col], k=cfg.kmer_k)
        default_pre = "truncated_svd"
    elif ft in ("embedding", "esm"):
        X = compute_esm_embeddings(
            df[cfg.sequence_col],
            model_name=cfg.embedding_model,
            layer=cfg.embedding_layer,
            batch_size=cfg.embedding_batch_size,
        )
        default_pre = "pca"
    elif ft == "precomputed":
        X = load_array(cfg.input_path)
        default_pre = cfg.pre_reducer
    else:
        raise ValueError(f"Unknown feature_type: '{ft}'. Choose: onehot_sparse | onehot_dense | kmer | embedding | precomputed")

    print(f"Feature matrix shape: {X.shape}")

    # 3. Linear pre-reduce
    pre_reducer_name = cfg.pre_reducer
    report: dict = {"feature_type": ft, "n_samples": len(df), "seed": cfg.seed}
    linear_reducer = None

    if pre_reducer_name == "truncated_svd":
        linear_reducer, X_pre = fit_truncated_svd(X, cfg.pre_reducer_n, cfg.seed)
    elif pre_reducer_name == "pca":
        linear_reducer, X_pre = fit_pca(X, cfg.pre_reducer_n, cfg.seed)
    elif pre_reducer_name == "incremental_pca":
        linear_reducer, X_pre = fit_incremental_pca(X, cfg.pre_reducer_n, cfg.incremental_pca_batch_size)
    elif pre_reducer_name == "none":
        X_pre = X.toarray() if hasattr(X, "toarray") else X
    else:
        raise ValueError(f"Unknown pre_reducer: '{pre_reducer_name}'")

    if linear_reducer is not None:
        report["linear"] = eval_linear_reduction(linear_reducer)
        print(f"Pre-reduced to {X_pre.shape[1]} dims | "
              f"explained variance: {report['linear']['total_explained_variance']:.3f}")

    save_array(X_pre, f"{cfg.output_dir}/pre_reduced.npy")

    # 4. Sample for nonlinear step
    n_sample = cfg.sample_n
    if n_sample and n_sample < len(df):
        if cfg.sample_strategy == "stratified" and cfg.sample_col and cfg.sample_col in df.columns:
            idx = stratified_sample(df, cfg.sample_col, n_sample, cfg.seed)
        else:
            idx = random_sample(df, n_sample, cfg.seed)
        X_sample = X_pre[idx]
        df_sample = df.iloc[idx].reset_index(drop=True)
        print(f"Sampled {len(idx):,} rows for visual reduction")
    else:
        X_sample = X_pre
        df_sample = df.reset_index(drop=True)

    # 5. Nonlinear visual reduction
    umap_model = None
    vr = cfg.visual_reducer
    if vr == "umap":
        coords_2d, umap_model = run_umap(
            X_sample, n_components=cfg.n_visual_dims, seed=cfg.seed,
            **cfg.visual_reducer_params,
        )
    elif vr == "tsne":
        coords_2d = run_tsne(
            X_sample, n_components=cfg.n_visual_dims, seed=cfg.seed,
        )
    elif vr == "none":
        print("No visual reducer specified. Saving pre-reduced matrix only.")
        return
    else:
        raise ValueError(f"Unknown visual_reducer: '{vr}'")

    # 6. Evaluate
    nb = eval_neighborhood(X_sample, coords_2d, seed=cfg.seed)
    report["neighborhood"] = nb
    report["runtime_s"] = round(time.time() - t0, 1)
    print(f"Trustworthiness: {nb['trustworthiness']:.3f} | Runtime: {report['runtime_s']}s")

    # 7. Plot
    plot_cols = [c for c in cfg.metadata_cols if c in df_sample.columns]
    if plot_cols:
        scatter_2d_multi(coords_2d, df_sample, color_cols=plot_cols, out_dir=cfg.output_dir)
        print(f"Saved {len(plot_cols)} scatter plot(s) to {cfg.output_dir}")

    # 8. Export all outputs
    save_reduction(coords_2d, df_sample, linear_reducer, report, cfg, umap_model=umap_model)
    print(f"Done. Outputs: {cfg.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dimensionality reduction on protein sequences.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
