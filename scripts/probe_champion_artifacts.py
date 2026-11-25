#!/usr/bin/env python3
"""
Empirical Artifact & Lineage Probe across Champion Evaluation Subsets (P-28 / P-29 / P-34).
Measures on balanced test manifests:
1. Path single-feature rule strength (|acc - 0.5|)
2. Scheme (HTTP/HTTPS) single-feature rule strength (|acc - 0.5|)
3. TLD categorical baseline accuracy and F1 alongside Majority Class Baseline (0.50)
4. Cross-dataset canonical URL intersection matrix
Saves verified JSON/CSV artifacts to results/artifacts/
"""

import json
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

MANIFEST_DIR = Path("data/manifests")
OUT_DIR = Path("results/artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset_manifest(name: str) -> pd.DataFrame:
    p = MANIFEST_DIR / f"{name}_manifest_seed42.csv"
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    df = pd.read_csv(p)
    # Filter to balanced TEST partition
    df_test = df[df["split"] == "TEST"].copy().reset_index(drop=True)
    return df_test


def probe_dataset(df: pd.DataFrame, name: str) -> dict:
    parsed = df["canonical_url"].map(urlparse)
    
    has_path = parsed.map(lambda p: bool(p.path and p.path != "/")).values
    is_https = parsed.map(lambda p: p.scheme == "https").values
    tlds = parsed.map(lambda p: (p.netloc.split(".")[-1].lower() if "." in p.netloc else "none")).values
    y = df["y"].values
    
    n_total = len(df)
    n_phish = int(np.sum(y == 1))
    n_legit = int(np.sum(y == 0))
    majority_class_baseline = max(n_phish, n_legit) / n_total if n_total > 0 else 0.50

    # 1. Path metrics
    acc_path_fwd = float(np.mean(has_path == y))
    acc_path_rev = float(np.mean((~has_path) == y))
    path_best_acc = max(acc_path_fwd, acc_path_rev)
    path_bias_delta = abs(path_best_acc - 0.5)

    phish_path_pct = float(np.mean(has_path[y == 1])) * 100.0 if n_phish > 0 else 0.0
    legit_path_pct = float(np.mean(has_path[y == 0])) * 100.0 if n_legit > 0 else 0.0
    
    # 2. Scheme metrics
    acc_scheme_fwd = float(np.mean((~is_https) == y))
    acc_scheme_rev = float(np.mean(is_https == y))
    scheme_best_acc = max(acc_scheme_fwd, acc_scheme_rev)
    scheme_bias_delta = abs(scheme_best_acc - 0.5)

    phish_https_pct = float(np.mean(is_https[y == 1])) * 100.0 if n_phish > 0 else 0.0
    legit_https_pct = float(np.mean(is_https[y == 0])) * 100.0 if n_legit > 0 else 0.0
    
    # 3. TLD-only categorical baseline (5-fold CV)
    vec = CountVectorizer(binary=True)
    X_tld = vec.fit_transform(tlds)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tld_accs = []
    tld_f1s = []
    for tr, ts in skf.split(X_tld, y):
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X_tld[tr], y[tr])
        preds = clf.predict(X_tld[ts])
        tld_accs.append(accuracy_score(y[ts], preds))
        tld_f1s.append(f1_score(y[ts], preds, zero_division=0))
    tld_mean_acc = float(np.mean(tld_accs))
    tld_mean_f1 = float(np.mean(tld_f1s))

    return {
        "dataset": name,
        "eval_split": "TEST (Balanced)",
        "total_samples": n_total,
        "phishing_count": n_phish,
        "legit_count": n_legit,
        "majority_class_baseline": majority_class_baseline,
        "path_forward_acc": acc_path_fwd,
        "path_reversed_acc": acc_path_rev,
        "path_strongest_acc": path_best_acc,
        "path_bias_abs_delta": path_bias_delta,
        "phish_has_path_pct": phish_path_pct,
        "legit_has_path_pct": legit_path_pct,
        "scheme_forward_acc": acc_scheme_fwd,
        "scheme_reversed_acc": acc_scheme_rev,
        "scheme_strongest_acc": scheme_best_acc,
        "scheme_bias_abs_delta": scheme_bias_delta,
        "phish_https_pct": phish_https_pct,
        "legit_https_pct": legit_https_pct,
        "tld_cv_accuracy": tld_mean_acc,
        "tld_cv_f1": tld_mean_f1,
    }


def main():
    print("================================================================================")
    print("  Artifact & Structural Bias Probe on Balanced TEST Workloads")
    print("================================================================================")
    datasets = {}
    reports = []
    
    for ds_name in ["PWD2021", "URLPhish2025", "EBBU2017"]:
        df = load_dataset_manifest(ds_name)
        datasets[ds_name] = df
        res = probe_dataset(df, ds_name)
        reports.append(res)
        
    df_rep = pd.DataFrame(reports)
    print(df_rep[["dataset", "total_samples", "majority_class_baseline", "path_strongest_acc", "path_bias_abs_delta", "scheme_strongest_acc", "scheme_bias_abs_delta", "tld_cv_accuracy", "tld_cv_f1"]].to_string(index=False))

    json_path = OUT_DIR / "champion_artifact_probe.json"
    csv_path = OUT_DIR / "champion_artifact_probe.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    df_rep.to_csv(csv_path, index=False)
    print(f"\nSaved artifact probe to {json_path} and {csv_path}")

    # Overlap Matrix across Full Manifests
    print("\n================================================================================")
    print("  Cross-Dataset Canonical URL Overlap Matrix")
    print("================================================================================")
    names = ["PWD2021", "URLPhish2025", "EBBU2017"]
    full_dfs = {n: pd.read_csv(MANIFEST_DIR / f"{n}_manifest_seed42.csv") for n in names}
    matrix_dict = {}
    for n1 in names:
        s1 = set(full_dfs[n1]["url_hash"])
        matrix_dict[n1] = {}
        for n2 in names:
            s2 = set(full_dfs[n2]["url_hash"])
            overlap = len(s1.intersection(s2))
            matrix_dict[n1][n2] = overlap

    df_mat = pd.DataFrame(matrix_dict)
    print(df_mat)

    lineage_path = OUT_DIR / "dataset_lineage_matrix.json"
    with open(lineage_path, "w", encoding="utf-8") as f:
        json.dump(matrix_dict, f, indent=2)
    print(f"Saved lineage matrix to {lineage_path}")


if __name__ == "__main__":
    main()
