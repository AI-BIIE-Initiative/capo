import numpy as np
import pandas as pd
from pathlib import Path


def load_dataframe(path: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    elif path.suffix in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    elif path.suffix in (".feather", ".fth"):
        return pd.read_feather(path)
    else:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. Supported: .csv, .parquet, .tsv, .feather"
        )


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    p = Path(path)
    if p.suffix in (".parquet", ".pq"):
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)


def load_array(path: str) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        return data[list(data.keys())[0]]
    else:
        raise ValueError(f"Unsupported array format: {path.suffix}. Use .npy or .npz")


def save_array(arr: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {list(df.columns)}"
        )
