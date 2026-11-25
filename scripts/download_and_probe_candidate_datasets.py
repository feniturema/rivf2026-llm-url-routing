#!/usr/bin/env python3
"""
Automated Downloader, Standardizer, and Headroom Probe for Candidate Phishing Benchmarks.
Downloads:
1. URL-Phish v1 (Mendeley 2025, Data in Brief)
2. LegitPhish (Mendeley 2025, Data in Brief)
3. EBBU 2017 (GitHub raw JSONs)
4. Hannousse 2021 (Mendeley Data)
5. PhishStorm (Aalto University)
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
import io
from pathlib import Path
import httpx
import numpy as np
import pandas as pd
from urllib.parse import urlparse

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.canonical import canonicalise_url, compute_url_hash
from src.data.split import create_group_disjoint_split, materialize_split
from src.specialist.char_model import CharTFIDFSpecialist
from src.specialist.calibration import IsotonicCalibrator
from src.evaluation.metrics import compute_classification_metrics

DATA_DIR = _REPO_ROOT / "data" / "candidates"
PROCESSED_DIR = _REPO_ROOT / "data" / "processed_candidates"
RESULTS_DIR = _REPO_ROOT / "results" / "candidate_probes"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def download_mendeley_direct(url: str, target_name: str) -> Path:
    target_file = DATA_DIR / target_name
    if target_file.exists() and target_file.stat().st_size > 1000:
        print(f"[CACHE] {target_name} already exists ({target_file.stat().st_size / 1e6:.2f} MB).")
        return target_file

    print(f"[DOWNLOADING] {target_name} from {url}...")
    with httpx.Client(timeout=120.0, follow_redirects=True, headers=HEADERS) as client:
        resp = client.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"Download failed: HTTP {resp.status_code}")
        target_file.write_bytes(resp.content)
        print(f"[SUCCESS] Saved {target_name} ({len(resp.content)/1e6:.2f} MB).")
        return target_file


def download_ebbu2017() -> Path:
    """Download EBBU 2017 dataset from GitHub raw JSONs."""
    target_csv = DATA_DIR / "ebbu2017.csv"
    if target_csv.exists() and target_csv.stat().st_size > 1000:
        print(f"[CACHE] ebbu2017.csv already exists.")
        return target_csv

    print("[DOWNLOADING] EBBU 2017 from GitHub...")
    base_url = "https://raw.githubusercontent.com/fcdalgic/GramBeddings/master/data/ebubekir"
    with httpx.Client(timeout=60.0, follow_redirects=True, headers=HEADERS) as client:
        r_legit = client.get(f"{base_url}/data_legitimate_36400.json")
        r_phish = client.get(f"{base_url}/data_phishing_37175.json")
        
        if r_legit.status_code != 200 or r_phish.status_code != 200:
            # Fallback repo
            base_url2 = "https://raw.githubusercontent.com/ebubekirbbr/pdd/master/input"
            r_legit = client.get(f"{base_url2}/data_legitimate_36400.json")
            r_phish = client.get(f"{base_url2}/data_phishing_37175.json")
            
        if r_legit.status_code != 200 or r_phish.status_code != 200:
            raise RuntimeError("Failed to download EBBU 2017 JSON files.")
            
        legit_data = r_legit.json()
        phish_data = r_phish.json()
        
        rows = []
        for item in legit_data:
            u = item if isinstance(item, str) else item.get("url", "")
            if u:
                rows.append({"raw_url": u, "y": 0})
        for item in phish_data:
            u = item if isinstance(item, str) else item.get("url", "")
            if u:
                rows.append({"raw_url": u, "y": 1})
                
        df = pd.DataFrame(rows)
        df.to_csv(target_csv, index=False)
        print(f"[SUCCESS] Built EBBU 2017 CSV ({len(df)} rows).")
        return target_csv


def download_phishstorm() -> Path:
    """Download PhishStorm from Aalto research repository."""
    target_csv = DATA_DIR / "phishstorm.csv"
    if target_csv.exists() and target_csv.stat().st_size > 1000:
        print(f"[CACHE] phishstorm.csv already exists.")
        return target_csv

    print("[DOWNLOADING] PhishStorm from Aalto repository...")
    url = "https://research.aalto.fi/files/24342/urlset.csv.zip"
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True, headers=HEADERS) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    for filename in z.namelist():
                        if filename.endswith(".csv"):
                            with z.open(filename) as f:
                                df = pd.read_csv(f, encoding="latin1", on_bad_lines="skip")
                                # Find URL column and label column
                                df.rename(columns={df.columns[0]: "raw_url", df.columns[-1]: "y"}, inplace=True)
                                df = df[["raw_url", "y"]].dropna()
                                df["y"] = df["y"].astype(int)
                                df.to_csv(target_csv, index=False)
                                print(f"[SUCCESS] Saved PhishStorm ({len(df)} rows).")
                                return target_csv
    except Exception as e:
        print(f"[WARNING] Aalto PhishStorm download direct failed: {e}")
    return None


def probe_dataset(df_raw: pd.DataFrame, dataset_name: str) -> dict:
    """Standardize, split 60/10/10/20, train specialist & calibrator, probe headroom."""
    print(f"\n=======================================================")
    print(f"  Probing Dataset: {dataset_name}")
    print(f"=======================================================")
    
    # 1. Canonicalise
    df = df_raw.copy()
    df["canonical_url"] = df["raw_url"].astype(str).map(canonicalise_url)
    df["url_hash"] = df["canonical_url"].map(compute_url_hash)
    
    # Drop contradictory labels within group
    label_nunique = df.groupby("url_hash")["y"].nunique()
    conflicts = set(label_nunique[label_nunique > 1].index)
    if conflicts:
        df = df[~df["url_hash"].isin(conflicts)].copy()
        
    # Deduplicate canonical
    df = df.drop_duplicates(subset=["url_hash"]).copy().reset_index(drop=True)
    n_unique = len(df)
    
    # Dataset statistics
    n_phish = int((df["y"] == 1).sum())
    n_legit = int((df["y"] == 0).sum())
    
    # Structural features
    has_path_count = sum(len(urlparse(u).path.strip("/")) > 0 for u in df["raw_url"])
    is_https_count = sum(u.lower().startswith("https://") for u in df["raw_url"])
    
    # Balance classes for evaluation if needed, or evaluate imbalanced
    legit_df = df[df["y"] == 0]
    phish_df = df[df["y"] == 1]
    n_min = min(len(legit_df), len(phish_df))
    
    balanced_df = pd.concat([
        legit_df.sample(n=n_min, random_state=42),
        phish_df.sample(n=n_min, random_state=42)
    ]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Split 60/10/10/20
    manifest = create_group_disjoint_split(balanced_df, seed=42)
    train_df, cal_df, dev_df, test_df = materialize_split(manifest)
    
    # Train Specialist
    model = CharTFIDFSpecialist(ngram_range=(3, 5), min_df=3, seed=42)
    model.fit(train_df["canonical_url"], train_df["y"])
    
    # Calibrate on CAL partition
    p_cal_uncal = model.predict_proba(cal_df["canonical_url"])
    calibrator = IsotonicCalibrator()
    calibrator.fit(p_cal_uncal, cal_df["y"].values)
    
    # Evaluate on TEST partition
    y_test = test_df["y"].values
    p_test_uncal = model.predict_proba(test_df["canonical_url"])
    p_test_cal = calibrator.calibrate(p_test_uncal)
    y_pred = (p_test_cal >= 0.5).astype(int)
    
    metrics = compute_classification_metrics(y_test, y_pred)
    errors = (y_pred != y_test)
    n_errors = int(np.sum(errors))
    n_test = len(y_test)
    error_rate = n_errors / n_test
    
    # High-confidence error analysis
    high_conf_040 = errors & (np.abs(p_test_cal - 0.5) >= 0.40)
    high_conf_035 = errors & (np.abs(p_test_cal - 0.5) >= 0.35)
    
    n_high_conf_040 = int(np.sum(high_conf_040))
    n_high_conf_035 = int(np.sum(high_conf_035))
    
    pct_hc_040 = (n_high_conf_040 / n_errors * 100.0) if n_errors > 0 else 0.0
    pct_hc_035 = (n_high_conf_035 / n_errors * 100.0) if n_errors > 0 else 0.0
    
    report = {
        "dataset_name": dataset_name,
        "total_raw_rows": len(df_raw),
        "unique_canonical_rows": n_unique,
        "n_legitimate": n_legit,
        "n_phishing": n_phish,
        "imbalance_ratio": f"{n_legit}:{n_phish}",
        "has_path_pct": f"{has_path_count / n_unique * 100:.1f}%",
        "is_https_pct": f"{is_https_count / n_unique * 100:.1f}%",
        "balanced_eval_size": len(balanced_df),
        "test_size": n_test,
        "specialist_f1": metrics["f1"],
        "specialist_mcc": metrics["mcc"],
        "specialist_accuracy": metrics["accuracy"],
        "test_errors": n_errors,
        "error_rate_pct": f"{error_rate * 100:.2f}%",
        "high_conf_errors_040": n_high_conf_040,
        "high_conf_pct_040": f"{pct_hc_040:.1f}%",
        "high_conf_errors_035": n_high_conf_035,
        "high_conf_pct_035": f"{pct_hc_035:.1f}%",
    }
    
    print(f"  Test Size:            {n_test}")
    print(f"  Specialist F1:        {metrics['f1']:.4f}")
    print(f"  Specialist MCC:       {metrics['mcc']:.4f}")
    print(f"  Total Test Errors:    {n_errors} ({error_rate*100:.2f}%)")
    print(f"  High-Conf Errors 0.40:{n_high_conf_040} / {n_errors} ({pct_hc_040:.1f}%)")
    print(f"  High-Conf Errors 0.35:{n_high_conf_035} / {n_errors} ({pct_hc_035:.1f}%)")
    
    return report


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    
    # 1. URL-Phish v1 (2025, Data in Brief)
    try:
        url1 = "https://data.mendeley.com/public-files/datasets/65z9twcx3r/files/0e9c55e4-9adb-43f5-8403-1bbd143ebdb6/file_downloaded"
        p1 = download_mendeley_direct(url1, "url_phish_v1.csv")
        df1 = pd.read_csv(p1, low_memory=False)
        url_col = "url" if "url" in df1.columns else df1.columns[0]
        label_col = "label" if "label" in df1.columns else df1.columns[-1]
        df1 = df1.dropna(subset=[url_col, label_col])
        df1_clean = pd.DataFrame({"raw_url": df1[url_col], "y": df1[label_col].astype(int)})
        reports.append(probe_dataset(df1_clean, "URL-Phish v1 (2025, Data in Brief)"))
    except Exception as e:
        print(f"[ERROR] URL-Phish probe failed: {e}")

    # 2. LegitPhish (2025, Data in Brief)
    try:
        url2 = "https://data.mendeley.com/public-files/datasets/hx4m73v2sf/files/0ed51e4d-d160-4047-b650-3f0801eea4aa/file_downloaded"
        p2 = download_mendeley_direct(url2, "legit_phish.csv")
        df2 = pd.read_csv(p2, low_memory=False)
        url_col = "URL" if "URL" in df2.columns else "url"
        label_col = "ClassLabel" if "ClassLabel" in df2.columns else "label"
        df2 = df2.dropna(subset=[url_col, label_col])
        # In LegitPhish: 0 = Phishing, 1 = Legitimate -> project y: 1 = Phish, 0 = Legit
        df2_clean = pd.DataFrame({"raw_url": df2[url_col], "y": 1 - df2[label_col].astype(int)})
        reports.append(probe_dataset(df2_clean, "LegitPhish (2025, Data in Brief)"))
    except Exception as e:
        print(f"[ERROR] LegitPhish probe failed: {e}")

    # 3. EBBU 2017 (ESWA)
    try:
        p3 = download_ebbu2017()
        if p3:
            df3 = pd.read_csv(p3)
            reports.append(probe_dataset(df3, "EBBU Phishing 2017 (ESWA)"))
    except Exception as e:
        print(f"[ERROR] EBBU 2017 probe failed: {e}")

    # 4. Hannousse 2021 (Dataset B, EAAI)
    try:
        url4 = "https://data.mendeley.com/public-files/datasets/c2gw7fy2j4/files/575316f4-ee1d-453e-a04f-7b950915b61b/file_downloaded"
        p4 = download_mendeley_direct(url4, "hannousse2021_b.csv")
        df4 = pd.read_csv(p4, low_memory=False)
        url_col = "url" if "url" in df4.columns else df4.columns[0]
        label_col = "status" if "status" in df4.columns else ("phishing" if "phishing" in df4.columns else df4.columns[-1])
        df4 = df4.dropna(subset=[url_col, label_col])
        y_map = df4[label_col].astype(str).map(lambda x: 1 if x.strip().lower() in ("phishing", "1", "phish") else 0)
        df4_clean = pd.DataFrame({"raw_url": df4[url_col], "y": y_map})
        reports.append(probe_dataset(df4_clean, "Hannousse Mendeley 2021 (EAAI)"))
    except Exception as e:
        print(f"[ERROR] Hannousse 2021 probe failed: {e}")

    # 5. PWD2021 (Mendeley, IEEE Access)
    try:
        url5 = "https://data.mendeley.com/public-files/datasets/n96ncsr5g4/files/dac80106-cc68-43c3-8810-96408c09fbbc/file_downloaded"
        p5 = download_mendeley_direct(url5, "pwd2021_index.sql")
        import re
        sql_text = p5.read_text(encoding="utf-8", errors="ignore")
        matches = re.findall(r"\((\d+),\s*'([^']+)',\s*'([^']*)',\s*([01]),", sql_text)
        if matches:
            df5_clean = pd.DataFrame([{"raw_url": m[1], "y": int(m[3])} for m in matches])
            reports.append(probe_dataset(df5_clean, "PWD2021 (IEEE Access 2022)"))
    except Exception as e:
        print(f"[ERROR] PWD2021 probe failed: {e}")

    # 6. PhishStorm
    try:
        p5 = download_phishstorm()
        if p5:
            df5 = pd.read_csv(p5, encoding="utf-8")
            reports.append(probe_dataset(df5, "PhishStorm 2014"))
    except Exception as e:
        print(f"[ERROR] PhishStorm probe failed: {e}")

    # Output Summary Table
    print("\n\n=======================================================")
    print("      CANDIDATE DATASET HEADROOM & AUDIT SUMMARY")
    print("=======================================================")
    summary_df = pd.DataFrame(reports)
    print(summary_df[[
        "dataset_name", "unique_canonical_rows", "balanced_eval_size",
        "specialist_f1", "test_errors", "error_rate_pct", "high_conf_pct_040"
    ]].to_string(index=False))

    out_json = RESULTS_DIR / "candidate_probe_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    print(f"\nFull probe results saved to {out_json}")


if __name__ == "__main__":
    main()
