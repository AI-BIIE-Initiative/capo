"""
Standardised output writer for clustering runs.
"""
import json
import pandas as pd
from pathlib import Path


def save_clustering(
    df: pd.DataFrame,
    profiles: dict,
    metrics: dict,
    cfg,
) -> None:
    """
    Save all clustering outputs to cfg.output_dir.

    Outputs:
        cluster_assignments.csv  — id, cluster, split + metadata_cols
        split_assignments.csv    — id, split
        cluster_metrics.json     — evaluation metrics
        run_config.yaml          — config snapshot
    """
    import yaml
    from dataclasses import asdict

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Cluster + split assignments
    keep = [cfg.id_col, "cluster"]
    if "split" in df.columns:
        keep.append("split")
    keep += [c for c in cfg.metadata_cols if c in df.columns]
    df[[c for c in keep if c in df.columns]].to_csv(out / "cluster_assignments.csv", index=False)

    if "split" in df.columns:
        split_cols = [cfg.id_col, "split"]
        df[[c for c in split_cols if c in df.columns]].to_csv(
            out / "split_assignments.csv", index=False
        )

    # Metrics
    with open(out / "cluster_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Config snapshot
    try:
        cfg_dict = asdict(cfg)
    except Exception:
        cfg_dict = vars(cfg)
    with open(out / "run_config.yaml", "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)
