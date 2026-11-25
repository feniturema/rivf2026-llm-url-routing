"""
Dataset loader and stratum selector.
Handles PhiUSIIL and Dataset 2 (PhishTank/Tranco) raw inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple
import pandas as pd

from .canonical import (
    canonicalise_url,
    compute_url_hash,
    is_s_ctrl,
    is_s_strict,
)

RAW_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


def load_dataset(
    dataset_name: Literal["PhiUSIIL", "Dataset2"],
    data_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """
    Load and standardise raw dataset into canonical format:
    Returns DataFrame with columns:
      - 'raw_url': original string
      - 'canonical_url': normalised string
      - 'url_hash': SHA-256 of canonical URL
      - 'y': 1 = phishing, 0 = legitimate
    """
    if dataset_name == "PhiUSIIL":
        p = data_dir / "PhiUSIIL_Phishing_URL_Dataset.csv"
        if not p.exists():
            raise FileNotFoundError(f"PhiUSIIL dataset not found at: {p}")
        df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
        # UCI encoding in PhiUSIIL is 1=legit, 0=phish -> project y: 1=phish, 0=legit
        df["y"] = 1 - df["label"].astype(int)
        df["raw_url"] = df["URL"].astype(str)
        
    elif dataset_name == "Dataset2":
        p = data_dir / "df_balanced.csv"
        if not p.exists():
            raise FileNotFoundError(f"Dataset 2 not found at: {p}")
        df = pd.read_csv(p)
        df["y"] = df["label"].astype(int)
        df["raw_url"] = df["URL"].astype(str)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Standardise
    df["canonical_url"] = df["raw_url"].map(canonicalise_url)
    df["url_hash"] = df["canonical_url"].map(compute_url_hash)
    
    return df[["raw_url", "canonical_url", "url_hash", "y"]].copy()


def get_stratum(
    df: pd.DataFrame,
    stratum: Literal["S-STRICT", "S-CTRL", "S-FULL"],
    balance_classes: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Filter dataset according to defined evaluation stratum.
    - S-STRICT (D-15): Strict HTTPS bare hostname without query/frag/params/ports.
    - S-CTRL: HTTPS scheme and empty path.
    - S-FULL: Uncontrolled full corpus.
    """
    d = df.copy()
    
    if stratum == "S-STRICT":
        mask = d["raw_url"].map(is_s_strict)
        d = d[mask].copy()
    elif stratum == "S-CTRL":
        mask = d["raw_url"].map(is_s_ctrl)
        d = d[mask].copy()
    elif stratum == "S-FULL":
        pass
    else:
        raise ValueError(f"Unknown stratum: {stratum}")

    # Drop conflicting labels within the same canonical group
    group_labels = d.groupby("url_hash")["y"].nunique()
    conflicting_hashes = set(group_labels[group_labels > 1].index)
    if conflicting_hashes:
        d = d[~d["url_hash"].isin(conflicting_hashes)].copy()

    # Deduplicate: keep one representative per canonical group
    d = d.drop_duplicates(subset=["url_hash"]).copy().reset_index(drop=True)

    # Class balance if requested
    if balance_classes:
        legit = d[d["y"] == 0]
        phish = d[d["y"] == 1]
        n_balanced = min(len(legit), len(phish))
        
        legit_sub = legit.sample(n=n_balanced, random_state=seed)
        phish_sub = phish.sample(n=n_balanced, random_state=seed)
        
        d = pd.concat([legit_sub, phish_sub]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return d
