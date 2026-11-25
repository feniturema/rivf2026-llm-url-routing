#!/usr/bin/env python3
"""
Generate and Freeze Dedicated Experiment Cohort Manifests for E3, E4, and E9.
Ensures 100% partition purity:
- E3: Pure TEST partitions (26,886 rows across PWD2021, URLPhish2025, EBBU2017)
- E4: Pure CAL_ROUTER partitions (13,446 rows)
- E9: Deterministic, balanced shared TEST cohort (600 rows: 200 per dataset)
Produces results/manifests/cohort_manifest_freeze.json with SHA-256 seals.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd


def compute_sha256(filepath: Path) -> str:
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


def main():
    manifest_dir = Path("data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_json_dir = Path("results/manifests")
    out_json_dir.mkdir(parents=True, exist_ok=True)

    master_datasets = {
        "PWD2021": manifest_dir / "PWD2021_manifest_seed42.csv",
        "URLPhish2025": manifest_dir / "URLPhish2025_manifest_seed42.csv",
        "EBBU2017": manifest_dir / "EBBU2017_manifest_seed42.csv",
    }

    cohort_registry = {}

    e3_dfs = []
    e4_dfs = []
    e9_samples_600 = []
    e9_samples_3000 = []

    for name, p in master_datasets.items():
        if not p.exists():
            raise FileNotFoundError(f"Master manifest missing: {p}")
        df = pd.read_csv(p)

        # 1. E3 TEST Manifest
        df_test = df[df["split"] == "TEST"].copy().reset_index(drop=True)
        if len(df_test) == 0 or not (df_test["split"] == "TEST").all():
            raise ValueError(f"Corrupt TEST split in {name}")
        e3_path = manifest_dir / f"e3_test_{name.lower()}.csv"
        df_test.to_csv(e3_path, index=False)
        e3_dfs.append(df_test)

        cohort_registry[e3_path.name] = {
            "experiment": "E3",
            "dataset": name,
            "split": "TEST",
            "total_rows": len(df_test),
            "y_0_legit": int((df_test["y"] == 0).sum()),
            "y_1_phish": int((df_test["y"] == 1).sum()),
            "sha256": compute_sha256(e3_path),
        }

        # 2. E4 CAL_ROUTER Manifest
        df_cal = df[df["split"] == "CAL_ROUTER"].copy().reset_index(drop=True)
        if len(df_cal) == 0 or not (df_cal["split"] == "CAL_ROUTER").all():
            raise ValueError(f"Corrupt CAL_ROUTER split in {name}")
        e4_path = manifest_dir / f"e4_cal_{name.lower()}.csv"
        df_cal.to_csv(e4_path, index=False)
        e4_dfs.append(df_cal)

        cohort_registry[e4_path.name] = {
            "experiment": "E4",
            "dataset": name,
            "split": "CAL_ROUTER",
            "total_rows": len(df_cal),
            "y_0_legit": int((df_cal["y"] == 0).sum()),
            "y_1_phish": int((df_cal["y"] == 1).sum()),
            "sha256": compute_sha256(e4_path),
        }

        # 3. E9 Shared Cohort Samples (seed=42)
        # 3a. 600-sample pilot subset (100 legit, 100 phish)
        df_test_legit_100 = df_test[df_test["y"] == 0].sample(n=100, random_state=42)
        df_test_phish_100 = df_test[df_test["y"] == 1].sample(n=100, random_state=42)
        e9_subset_600 = pd.concat([df_test_legit_100, df_test_phish_100], ignore_index=True)
        e9_samples_600.append(e9_subset_600)

        # 3b. 3000-sample main benchmark cohort (500 legit, 500 phish per dataset)
        df_test_legit_500 = df_test[df_test["y"] == 0].sample(n=500, random_state=42)
        df_test_phish_500 = df_test[df_test["y"] == 1].sample(n=500, random_state=42)
        e9_subset_3000 = pd.concat([df_test_legit_500, df_test_phish_500], ignore_index=True)
        e9_samples_3000.append(e9_subset_3000)

    # Combined E3 All TEST
    df_e3_all = pd.concat(e3_dfs, ignore_index=True)
    e3_all_path = manifest_dir / "e3_test_all_26886.csv"
    df_e3_all.to_csv(e3_all_path, index=False)
    cohort_registry[e3_all_path.name] = {
        "experiment": "E3",
        "dataset": "ALL_3_CORPORA",
        "split": "TEST",
        "total_rows": len(df_e3_all),
        "y_0_legit": int((df_e3_all["y"] == 0).sum()),
        "y_1_phish": int((df_e3_all["y"] == 1).sum()),
        "sha256": compute_sha256(e3_all_path),
    }

    # Combined E4 All CAL_ROUTER
    df_e4_all = pd.concat(e4_dfs, ignore_index=True)
    e4_all_path = manifest_dir / "e4_cal_all_13446.csv"
    df_e4_all.to_csv(e4_all_path, index=False)
    cohort_registry[e4_all_path.name] = {
        "experiment": "E4",
        "dataset": "ALL_3_CORPORA",
        "split": "CAL_ROUTER",
        "total_rows": len(df_e4_all),
        "y_0_legit": int((df_e4_all["y"] == 0).sum()),
        "y_1_phish": int((df_e4_all["y"] == 1).sum()),
        "sha256": compute_sha256(e4_all_path),
    }

    # Combined E9 Shared TEST Cohort (600 rows)
    df_e9_600 = pd.concat(e9_samples_600, ignore_index=True)
    e9_path_600 = manifest_dir / "e9_shared_test_cohort_600.csv"
    df_e9_600.to_csv(e9_path_600, index=False)
    cohort_registry[e9_path_600.name] = {
        "experiment": "E9",
        "dataset": "ALL_3_CORPORA",
        "split": "TEST",
        "total_rows": len(df_e9_600),
        "y_0_legit": int((df_e9_600["y"] == 0).sum()),
        "y_1_phish": int((df_e9_600["y"] == 1).sum()),
        "sha256": compute_sha256(e9_path_600),
    }

    # Combined E9 Shared TEST Cohort (3,000 rows)
    df_e9_3000 = pd.concat(e9_samples_3000, ignore_index=True)
    e9_path_3000 = manifest_dir / "e9_shared_test_cohort_3000.csv"
    df_e9_3000.to_csv(e9_path_3000, index=False)
    cohort_registry[e9_path_3000.name] = {
        "experiment": "E9",
        "dataset": "ALL_3_CORPORA",
        "split": "TEST",
        "total_rows": len(df_e9_3000),
        "y_0_legit": int((df_e9_3000["y"] == 0).sum()),
        "y_1_phish": int((df_e9_3000["y"] == 1).sum()),
        "sha256": compute_sha256(e9_path_3000),
    }

    freeze_payload = {
        "timestamp": "2026-08-28T01:21:00Z",
        "protocol": "S-STRICT Seed 42 Fixed Partition Cohorts",
        "cohorts": cohort_registry,
    }

    freeze_path = out_json_dir / "cohort_manifest_freeze.json"
    with open(freeze_path, "w", encoding="utf-8") as f:
        json.dump(freeze_payload, f, indent=2)

    print("=" * 80)
    print("  COHORT MANIFEST GENERATION & FREEZE COMPLETE")
    print("=" * 80)
    for fname, meta in cohort_registry.items():
        print(f"  {fname:<32} Split: {meta['split']:<12} Rows: {meta['total_rows']:<6} SHA-256: {meta['sha256'][:16]}...")
    print("=" * 80)
    print(f"Sealed metadata to: {freeze_path}")

if __name__ == '__main__':
    main()
