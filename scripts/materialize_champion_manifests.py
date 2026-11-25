#!/usr/bin/env python3
"""
Materialize Fixed Group-Disjoint 60/10/10/20 Manifests for Champion Datasets:
1. PWD2021 (Primary Workload - IEEE Access 2022)
2. URL-Phish v1 (Modern Workload - Data in Brief 2025)
3. EBBU 2017 (Classic Workload - ESWA 2019)
Also generates the unified 300-row LLM Pilot manifest across champion datasets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.canonical import canonicalise_url, compute_url_hash
from src.data.split import create_group_disjoint_split, materialize_split, compute_manifest_hash
from src.specialist.char_model import CharTFIDFSpecialist
from src.specialist.calibration import IsotonicCalibrator

DATA_DIR = _REPO_ROOT / "data" / "candidates"
MANIFEST_DIR = _REPO_ROOT / "data" / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


def process_dataset(df_raw: pd.DataFrame, dataset_id: str, balance: bool = True, seed: int = 42):
    print(f"\nProcessing {dataset_id}...")
    df = df_raw.copy()
    df["canonical_url"] = df["raw_url"].astype(str).map(canonicalise_url)
    df["url_hash"] = df["canonical_url"].map(compute_url_hash)
    
    # Drop conflicts
    conflicts = set(df.groupby("url_hash")["y"].nunique().loc[lambda x: x > 1].index)
    if conflicts:
        df = df[~df["url_hash"].isin(conflicts)].copy()
        
    df = df.drop_duplicates(subset=["url_hash"]).copy().reset_index(drop=True)
    
    if balance:
        legit = df[df["y"] == 0]
        phish = df[df["y"] == 1]
        n_min = min(len(legit), len(phish))
        df = pd.concat([
            legit.sample(n=n_min, random_state=seed),
            phish.sample(n=n_min, random_state=seed)
        ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        
    df["sample_id"] = [f"{dataset_id}_{i:06d}" for i in range(len(df))]
    manifest = create_group_disjoint_split(df, seed=seed)
    
    # Compute manifest hash
    m_hash = compute_manifest_hash(manifest)
    manifest["manifest_hash"] = m_hash
    manifest["dataset"] = dataset_id
    
    train, cal, dev, test = materialize_split(manifest)
    print(f"  Total: {len(manifest)} | Train: {len(train)}, Cal: {len(cal)}, Dev: {len(dev)}, Test: {len(test)}")
    print(f"  Manifest Hash: {m_hash[:16]}...")
    
    # Fit specialist
    model = CharTFIDFSpecialist(ngram_range=(3, 5), min_df=3, seed=seed)
    model.fit(train["canonical_url"], train["y"])
    
    calibrator = IsotonicCalibrator()
    p_cal_uncal = model.predict_proba(cal["canonical_url"])
    calibrator.fit(p_cal_uncal, cal["y"].values)
    
    # Attach specialist predictions across all splits
    p_uncal = model.predict_proba(manifest["canonical_url"])
    p_cal = calibrator.calibrate(p_uncal)
    manifest["specialist_prob_uncal"] = p_uncal
    manifest["specialist_prob_cal"] = p_cal
    manifest["specialist_pred"] = (p_cal >= 0.5).astype(int)
    manifest["canonical_url_sha256"] = manifest["url_hash"]
    
    out_csv = MANIFEST_DIR / f"{dataset_id}_manifest_seed{seed}.csv"
    manifest.to_csv(out_csv, index=False)
    print(f"  Saved to {out_csv}")
    return manifest


def main():
    if not DATA_DIR.exists():
        raise SystemExit(
            f"Raw source datasets not found at {DATA_DIR}. "
            "Pre-materialized manifests are already provided in data/manifests/; "
            "to rebuild from scratch, run `python scripts/download_and_probe_candidate_datasets.py` first."
        )
    # 1. PWD2021
    p_sql = DATA_DIR / "pwd2021_index.sql"
    import re
    sql_text = p_sql.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"\((\d+),\s*'([^']+)',\s*'([^']*)',\s*([01]),", sql_text)
    df_pwd = pd.DataFrame([{"raw_url": m[1], "y": int(m[3])} for m in matches])
    m_pwd = process_dataset(df_pwd, "PWD2021")

    # 2. URL-Phish v1
    p_url = DATA_DIR / "url_phish_v1.csv"
    df_url = pd.read_csv(p_url, low_memory=False).dropna(subset=["url", "label"])
    df_url = pd.DataFrame({"raw_url": df_url["url"], "y": df_url["label"].astype(int)})
    m_url = process_dataset(df_url, "URLPhish2025")

    # 3. EBBU 2017
    p_ebbu = DATA_DIR / "ebbu2017.csv"
    df_ebbu = pd.read_csv(p_ebbu).dropna(subset=["raw_url", "y"])
    m_ebbu = process_dataset(df_ebbu, "EBBU2017")

    # Generate Champion 300-Row Stratified Pilot Manifest:
    # 150 rows from PWD2021 TEST + 75 rows from URLPhish TEST + 75 rows from EBBU TEST
    print("\nGenerating 300-row Champion Pilot Manifest...")
    p_test = m_pwd[m_pwd["split"] == "TEST"]
    u_test = m_url[m_url["split"] == "TEST"]
    e_test = m_ebbu[m_ebbu["split"] == "TEST"]

    sub_pwd = pd.concat([
        p_test[p_test["y"] == 0].sample(n=75, random_state=42),
        p_test[p_test["y"] == 1].sample(n=75, random_state=42),
    ])
    sub_url = pd.concat([
        u_test[u_test["y"] == 0].sample(n=38, random_state=42),
        u_test[u_test["y"] == 1].sample(n=37, random_state=42),
    ])
    sub_ebbu = pd.concat([
        e_test[e_test["y"] == 0].sample(n=37, random_state=42),
        e_test[e_test["y"] == 1].sample(n=38, random_state=42),
    ])

    pilot_300 = pd.concat([sub_pwd, sub_url, sub_ebbu]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    pilot_300["pilot_id"] = [f"PILOT_{i:04d}" for i in range(len(pilot_300))]
    
    pilot_path = Path("data/champion_pilot_manifest_300.csv")
    pilot_300.to_csv(pilot_path, index=False)
    print(f"Saved 300-row Champion Pilot Manifest to {pilot_path} ({len(pilot_300)} rows).")


if __name__ == "__main__":
    main()
