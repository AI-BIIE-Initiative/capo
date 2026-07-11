"""
Standardised output writer for dimensionality reduction runs.
Saves coordinates, fitted models, evaluation report, and config snapshot.
"""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


def save_reduction(
    coords_2d: np.ndarray,
    df: pd.DataFrame,
    linear_reducer,
    report: dict,
    cfg,
    umap_model=None,
) -> None:
    """
    Save all dimensionality reduction outputs to cfg.output_dir.

    Outputs:
        reduced_coordinates.parquet  — 2D coords + id + all metadata_cols
        linear_reducer.joblib        — fitted PCA or TruncatedSVD
        umap_model.joblib            — fitted UMAP (supports transform on new data)
        reduction_report.json        — explained variance, trustworthiness, runtime, seed
        run_config.yaml              — full config snapshot
    """
    import yaml
    from dataclasses import asdict, fields

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 2D coordinates with metadata
    coord_df = pd.DataFrame(
        coords_2d,
        columns=[f"dim_{i}" for i in range(coords_2d.shape[1])],
    )
    for col in [cfg.id_col] + list(cfg.metadata_cols):
        if col in df.columns:
            coord_df[col] = df[col].values[: len(coord_df)]
    coord_df.to_parquet(out / "reduced_coordinates.parquet", index=False)

    # Fitted models
    if linear_reducer is not None:
        joblib.dump(linear_reducer, out / "linear_reducer.joblib")
    if umap_model is not None:
        joblib.dump(umap_model, out / "umap_model.joblib")

    # Evaluation report
    with open(out / "reduction_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Config snapshot
    try:
        cfg_dict = asdict(cfg)
    except Exception:
        cfg_dict = {f: getattr(cfg, f, None) for f in dir(cfg) if not f.startswith("_")}
    with open(out / "run_config.yaml", "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)
