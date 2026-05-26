"""
data_loader.py
──────────────
Dataset loading and preprocessing driven entirely by config.yaml entries.
Returns ready-to-use (X, y, views, encoder) tuples.
Compatible with Python 3.12 / scikit-learn 1.6.x.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_dataset(
    dataset_cfg: dict,
) -> tuple[np.ndarray, np.ndarray, list[list[int]], LabelEncoder]:
    """
    Load a single dataset from a config entry.

    Parameters
    ----------
    dataset_cfg : dict with keys:
        path       – path to CSV file
        label_col  – name of the target column
        drop_cols  – list of columns to drop (e.g. user IDs)
        views      – list of [start, end] pairs (end-exclusive index ranges)

    Returns
    -------
    X       : float64 numpy array  (n_samples, n_features)
    y       : int numpy array      (n_samples,)
    views   : list of lists of column indices, one per view
    encoder : fitted LabelEncoder  (use .classes_ for original labels)
    """
    
    path      = dataset_cfg["path"]
    label_col = dataset_cfg["label_col"]
    drop_cols = dataset_cfg.get("drop_cols", [])
    view_cfg  = dataset_cfg["views"]          # list of [start, end] pairs

    df = pd.read_csv(path)

    # Drop auxiliary columns
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Separate features and labels
    y_raw = df[label_col].values
    X_df  = df.drop(columns=[label_col]).astype(float)
    X     = X_df.values

    # Encode labels
    encoder = LabelEncoder()
    y       = encoder.fit_transform(y_raw)

    # Build view index lists from [start, end) ranges
    views: list[list[int]] = [list(range(start, end)) for start, end in view_cfg]

    return X, y, views, encoder
