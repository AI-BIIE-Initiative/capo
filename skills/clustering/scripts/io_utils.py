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
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_array(path: str) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        return data[list(data.keys())[0]]
    raise ValueError(f"Unsupported array format: {path.suffix}. Use .npy or .npz")
