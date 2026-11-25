"""
Canonical Group-Disjoint 4-Way Partition Generator (D-15).
Splits dataset into:
- TRAIN: 60% (Specialist training)
- CAL_ROUTER: 10% (Advantage router training / calibration)
- DEV_POLICY: 10% (Operating point, threshold, integration rule selection)
- TEST: 20% (Final single-touch evaluation)
"""

from __future__ import annotations

import hashlib
from typing import Dict, Tuple
import numpy as np
import pandas as pd


def create_group_disjoint_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.60,
    cal_router_ratio: float = 0.10,
    dev_policy_ratio: float = 0.10,
    test_ratio: float = 0.20,
) -> pd.DataFrame:
    """
    Creates deterministic group-disjoint split where identical canonical URLs
    (and their hashes) NEVER cross split boundaries.
    """
    assert abs(train_ratio + cal_router_ratio + dev_policy_ratio + test_ratio - 1.0) < 1e-6
    
    d = df.copy().reset_index(drop=True)
    if "sample_id" not in d.columns:
        d["sample_id"] = [f"SMP_{i:06d}" for i in range(len(d))]

    # Get unique groups
    rng = np.random.RandomState(seed)
    
    # Stratify by class label at group level
    groups = d[["url_hash", "y"]].drop_duplicates().copy()
    
    splits = []
    for label_val in (0, 1):
        sub_groups = list(groups[groups["y"] == label_val]["url_hash"].values)
        rng.shuffle(sub_groups)
        n_sub = len(sub_groups)
        
        n_train = int(round(n_sub * train_ratio))
        n_cal = int(round(n_sub * cal_router_ratio))
        n_dev = int(round(n_sub * dev_policy_ratio))
        n_test = n_sub - (n_train + n_cal + n_dev)
        
        g_train = sub_groups[:n_train]
        g_cal = sub_groups[n_train:n_train + n_cal]
        g_dev = sub_groups[n_train + n_cal:n_train + n_cal + n_dev]
        g_test = sub_groups[n_train + n_cal + n_dev:]
        
        for g in g_train:
            splits.append({"url_hash": g, "split": "TRAIN"})
        for g in g_cal:
            splits.append({"url_hash": g, "split": "CAL_ROUTER"})
        for g in g_dev:
            splits.append({"url_hash": g, "split": "DEV_POLICY"})
        for g in g_test:
            splits.append({"url_hash": g, "split": "TEST"})

    split_map = pd.DataFrame(splits)
    manifest = d.merge(split_map, on="url_hash", how="inner")
    
    return manifest


def compute_manifest_hash(manifest_df: pd.DataFrame) -> str:
    """Compute SHA-256 digest of the sample_id + split pairs."""
    s = manifest_df[["sample_id", "split"]].sort_values("sample_id").to_csv(index=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def materialize_split(manifest_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract partitions from manifest DataFrame."""
    train = manifest_df[manifest_df["split"] == "TRAIN"].copy().reset_index(drop=True)
    cal_router = manifest_df[manifest_df["split"] == "CAL_ROUTER"].copy().reset_index(drop=True)
    dev_policy = manifest_df[manifest_df["split"] == "DEV_POLICY"].copy().reset_index(drop=True)
    test = manifest_df[manifest_df["split"] == "TEST"].copy().reset_index(drop=True)
    return train, cal_router, dev_policy, test
