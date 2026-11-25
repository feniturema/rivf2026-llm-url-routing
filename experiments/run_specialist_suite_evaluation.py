#!/usr/bin/env python3
"""
Comprehensive Traditional Specialist Baseline Benchmark across Champion Datasets.
Tests multiple traditional architectures across 10 random seeds:
1. Char 3-5g TF-IDF + Logistic Regression (Primary Inductive Baseline)
2. Char 3-5g TF-IDF + SGD (Linear SVM / Modified Huber)
3. Char 3-5g TF-IDF + Multinomial Naive Bayes
4. Lexical Feature Extractor + XGBoost Classifier
5. Lexical Feature Extractor + Random Forest Classifier
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, brier_score_loss

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.canonical import canonicalise_url, compute_url_hash
from src.data.split import create_group_disjoint_split, materialize_split
from src.specialist.lexical_model import extract_lexical_features
from src.specialist.calibration import IsotonicCalibrator

DATA_DIR = _REPO_ROOT / "data" / "candidates"
RESULTS_DIR = _REPO_ROOT / "results" / "specialist_baselines"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask] == (probs[mask] >= 0.5).astype(int))
            bin_conf = np.mean(np.maximum(probs[mask], 1.0 - probs[mask]))
            ece += (np.sum(mask) / len(probs)) * np.abs(bin_acc - bin_conf)
    return float(ece)


def load_dataset_clean(name: str) -> pd.DataFrame:
    if name == "PWD2021":
        p_sql = DATA_DIR / "pwd2021_index.sql"
        import re
        sql_text = p_sql.read_text(encoding="utf-8", errors="ignore")
        matches = re.findall(r"\((\d+),\s*'([^']+)',\s*'([^']*)',\s*([01]),", sql_text)
        df = pd.DataFrame([{"raw_url": m[1], "y": int(m[3])} for m in matches])
    elif name == "URLPhish2025":
        p_url = DATA_DIR / "url_phish_v1.csv"
        df_url = pd.read_csv(p_url, low_memory=False).dropna(subset=["url", "label"])
        df = pd.DataFrame({"raw_url": df_url["url"], "y": df_url["label"].astype(int)})
    elif name == "EBBU2017":
        p_ebbu = DATA_DIR / "ebbu2017.csv"
        df = pd.read_csv(p_ebbu).dropna(subset=["raw_url", "y"])
    else:
        raise ValueError(f"Unknown dataset: {name}")

    df["canonical_url"] = df["raw_url"].astype(str).map(canonicalise_url)
    df["url_hash"] = df["canonical_url"].map(compute_url_hash)
    
    # Drop conflicts
    conflicts = set(df.groupby("url_hash")["y"].nunique().loc[lambda x: x > 1].index)
    if conflicts:
        df = df[~df["url_hash"].isin(conflicts)].copy()
        
    df = df.drop_duplicates(subset=["url_hash"]).copy().reset_index(drop=True)
    return df


def benchmark_dataset_multiseed(dataset_name: str) -> dict:
    print(f"\n=======================================================")
    print(f"  Multi-Seed Specialist Evaluation: {dataset_name}")
    print(f"=======================================================")
    df_clean = load_dataset_clean(dataset_name)
    
    # Balance
    legit = df_clean[df_clean["y"] == 0]
    phish = df_clean[df_clean["y"] == 1]
    n_min = min(len(legit), len(phish))
    
    print(f"Clean Unique URLs: {len(df_clean)} | Class Balanced: {n_min * 2}")

    models_metrics = {
        "Char_TFIDF_LogisticRegression": [],
        "Char_TFIDF_LinearSVM": [],
        "Char_TFIDF_NaiveBayes": [],
        "Lexical_XGBoost": [],
        "Lexical_RandomForest": [],
    }

    headroom_stats = []

    for seed in SEEDS:
        df_balanced = pd.concat([
            legit.sample(n=n_min, random_state=seed),
            phish.sample(n=n_min, random_state=seed)
        ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

        manifest = create_group_disjoint_split(df_balanced, seed=seed)
        train, cal, dev, test = materialize_split(manifest)
        
        y_train, y_cal, y_test = train["y"].values, cal["y"].values, test["y"].values
        urls_train, urls_cal, urls_test = train["canonical_url"], cal["canonical_url"], test["canonical_url"]

        # 1. Char TF-IDF Features
        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, max_features=50000)
        X_train_char = vec.fit_transform(urls_train)
        X_cal_char = vec.transform(urls_cal)
        X_test_char = vec.transform(urls_test)

        # 2. Lexical Features
        X_train_lex = extract_lexical_features(urls_train)
        X_cal_lex = extract_lexical_features(urls_cal)
        X_test_lex = extract_lexical_features(urls_test)

        # Model 1: Char-TFIDF + Logistic Regression
        lr = LogisticRegression(C=1.0, max_iter=500, random_state=seed, solver="lbfgs")
        lr.fit(X_train_char, y_train)
        p_cal_lr = lr.predict_proba(X_cal_char)[:, 1]
        p_test_lr = lr.predict_proba(X_test_char)[:, 1]
        
        iso = IsotonicCalibrator().fit(p_cal_lr, y_cal)
        p_test_lr_cal = iso.calibrate(p_test_lr)
        y_pred_lr = (p_test_lr_cal >= 0.5).astype(int)
        
        f1_lr = f1_score(y_test, y_pred_lr)
        mcc_lr = matthews_corrcoef(y_test, y_pred_lr)
        acc_lr = accuracy_score(y_test, y_pred_lr)
        brier_lr = brier_score_loss(y_test, p_test_lr_cal)
        ece_lr = compute_ece(p_test_lr_cal, y_test)
        
        models_metrics["Char_TFIDF_LogisticRegression"].append({
            "f1": f1_lr, "mcc": mcc_lr, "accuracy": acc_lr, "brier": brier_lr, "ece": ece_lr
        })

        # Headroom breakdown on primary model
        errors = (y_pred_lr != y_test)
        n_errors = int(np.sum(errors))
        n_high_conf_040 = int(np.sum(errors & (np.abs(p_test_lr_cal - 0.5) >= 0.40)))
        n_uncert_040 = n_errors - n_high_conf_040
        
        headroom_stats.append({
            "seed": seed,
            "test_size": len(y_test),
            "errors": n_errors,
            "error_rate": n_errors / len(y_test),
            "high_conf_errors": n_high_conf_040,
            "uncertain_errors": n_uncert_040,
            "uncertain_pct": (n_uncert_040 / n_errors * 100.0) if n_errors > 0 else 0.0,
        })

        # Model 2: Char-TFIDF + Linear SVM (Modified Huber loss for probabilities)
        svm = SGDClassifier(loss="modified_huber", max_iter=1000, random_state=seed)
        svm.fit(X_train_char, y_train)
        p_test_svm = svm.predict_proba(X_test_char)[:, 1]
        y_pred_svm = (p_test_svm >= 0.5).astype(int)
        models_metrics["Char_TFIDF_LinearSVM"].append({
            "f1": f1_score(y_test, y_pred_svm),
            "mcc": matthews_corrcoef(y_test, y_pred_svm),
            "accuracy": accuracy_score(y_test, y_pred_svm)
        })

        # Model 3: Char-TFIDF + Naive Bayes
        nb = MultinomialNB()
        nb.fit(X_train_char, y_train)
        p_test_nb = nb.predict_proba(X_test_char)[:, 1]
        y_pred_nb = (p_test_nb >= 0.5).astype(int)
        models_metrics["Char_TFIDF_NaiveBayes"].append({
            "f1": f1_score(y_test, y_pred_nb),
            "mcc": matthews_corrcoef(y_test, y_pred_nb),
            "accuracy": accuracy_score(y_test, y_pred_nb)
        })

        # Model 4: Lexical + XGBoost
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=seed, n_jobs=-1, eval_metric="logloss")
        xgb_model.fit(X_train_lex, y_train)
        p_test_xgb = xgb_model.predict_proba(X_test_lex)[:, 1]
        y_pred_xgb = (p_test_xgb >= 0.5).astype(int)
        models_metrics["Lexical_XGBoost"].append({
            "f1": f1_score(y_test, y_pred_xgb),
            "mcc": matthews_corrcoef(y_test, y_pred_xgb),
            "accuracy": accuracy_score(y_test, y_pred_xgb)
        })

        # Model 5: Lexical + Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
        rf.fit(X_train_lex, y_train)
        p_test_rf = rf.predict_proba(X_test_lex)[:, 1]
        y_pred_rf = (p_test_rf >= 0.5).astype(int)
        models_metrics["Lexical_RandomForest"].append({
            "f1": f1_score(y_test, y_pred_rf),
            "mcc": matthews_corrcoef(y_test, y_pred_rf),
            "accuracy": accuracy_score(y_test, y_pred_rf)
        })

    # Summarize models
    summary = {}
    print(f"\n--- 10-Seed Performance Summary for {dataset_name} ---")
    print(f"{'Model Architecture':<35} | {'F1 Score (Mean ± Std)':<22} | {'MCC (Mean ± Std)':<20} | {'Accuracy':<15}")
    print("-" * 100)
    for model_name, metric_list in models_metrics.items():
        f1s = [m["f1"] for m in metric_list]
        mccs = [m["mcc"] for m in metric_list]
        accs = [m["accuracy"] for m in metric_list]
        
        f1_str = f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}"
        mcc_str = f"{np.mean(mccs):.4f} ± {np.std(mccs):.4f}"
        acc_str = f"{np.mean(accs):.4f} ± {np.std(accs):.4f}"
        print(f"{model_name:<35} | {f1_str:<22} | {mcc_str:<20} | {acc_str:<15}")
        
        summary[model_name] = {
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "mcc_mean": float(np.mean(mccs)), "mcc_std": float(np.std(mccs)),
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        }

    # Summarize Headroom
    err_means = [h["errors"] for h in headroom_stats]
    err_rates = [h["error_rate"] * 100.0 for h in headroom_stats]
    unc_pcts = [h["uncertain_pct"] for h in headroom_stats]
    
    print(f"\n--- Headroom & Uncertainty Zone Breakdown (Primary Char-LR) ---")
    print(f"  Test Errors per Seed:     {np.mean(err_means):.1f} ± {np.std(err_means):.1f} ({np.mean(err_rates):.2f}% error rate)")
    print(f"  Errors in Uncertainty Zone: {np.mean(unc_pcts):.1f}% ± {np.std(unc_pcts):.1f}% (Rescuable by LLM)")
    print(f"  High-Confidence Blindspots: {100.0 - np.mean(unc_pcts):.1f}% (Unrescuable)")

    return {
        "dataset": dataset_name,
        "models": summary,
        "headroom": {
            "avg_errors": float(np.mean(err_means)),
            "avg_error_rate_pct": float(np.mean(err_rates)),
            "avg_uncertain_errors_pct": float(np.mean(unc_pcts)),
        }
    }


def main():
    if not DATA_DIR.exists():
        raise SystemExit(
            f"Raw source datasets not found at {DATA_DIR}. "
            "Pre-computed 10-seed results are in results/specialist_baselines/traditional_specialists_10seeds.json; "
            "to re-run from raw corpora, run `python scripts/download_and_probe_candidate_datasets.py` first."
        )
    results = {}
    for ds in ["PWD2021", "URLPhish2025", "EBBU2017"]:
        results[ds] = benchmark_dataset_multiseed(ds)
        
    out_file = RESULTS_DIR / "traditional_specialists_10seeds.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[COMPLETE] All 10-seed benchmarks saved to {out_file}")


if __name__ == "__main__":
    main()
