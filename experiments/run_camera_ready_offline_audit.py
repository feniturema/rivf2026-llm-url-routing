#!/usr/bin/env python3
"""
[ARCHIVED PRE-v2 PROTOTYPE — SUPERSEDED BY experiments/reproduce_v2_all.py]
Pre-backfill prototype retained for reference only. Authoritative camera-ready v2 metrics
are produced by `experiments/reproduce_v2_all.py` (eval_v2_main, eval_v2_policies, eval_v2_supplement).
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import binom, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.routing.policies import compute_routing_scores


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    acc = (tp + tn) / max(len(y_true), 1)
    return {
        "n": int(len(y_true)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_2x2(y_true: np.ndarray, y_spec: np.ndarray, y_llm: np.ndarray) -> dict:
    sc = y_spec == y_true
    lc = y_llm == y_true
    a = int(np.sum(sc & lc))
    b = int(np.sum((~sc) & lc))
    c = int(np.sum(sc & (~lc)))
    d = int(np.sum((~sc) & (~lc)))
    n = int(len(y_true))
    e = b + d
    n_ok = a + c
    rho = b / max(e, 1)
    eps = c / max(n_ok, 1)
    eps_star = rho * e / max(n_ok, 1)
    n_disc = b + c
    p_val = float(min(1.0, 2.0 * binom.cdf(min(b, c), n_disc, 0.5))) if n_disc > 0 else 1.0
    return {
        "n": n,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e_errors": e,
        "pi": float(e / max(n, 1)),
        "rho": float(rho),
        "eps": float(eps),
        "eps_star": float(eps_star),
        "margin": float(eps_star - eps),
        "net": int(b - c),
        "mcnemar_p": p_val,
    }


def paired_bootstrap_ci(
    y_true: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """95% paired bootstrap CI for (metric(y_a) - metric(y_b))."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    df1s = np.empty(n_boot, dtype=np.float64)
    daccs = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ya = y_a[idx]
        yb = y_b[idx]

        tp_a = np.sum((yt == 1) & (ya == 1))
        fp_a = np.sum((yt == 0) & (ya == 1))
        fn_a = np.sum((yt == 1) & (ya == 0))
        f1_a = (2.0 * tp_a) / max(2.0 * tp_a + fp_a + fn_a, 1.0)

        tp_b = np.sum((yt == 1) & (yb == 1))
        fp_b = np.sum((yt == 0) & (yb == 1))
        fn_b = np.sum((yt == 1) & (yb == 0))
        f1_b = (2.0 * tp_b) / max(2.0 * tp_b + fp_b + fn_b, 1.0)

        df1s[k] = f1_a - f1_b
        daccs[k] = np.mean(ya == yt) - np.mean(yb == yt)

    return {
        "delta_f1_point": float(compute_metrics(y_true, y_a)["f1"] - compute_metrics(y_true, y_b)["f1"]),
        "delta_f1_ci95_low": float(np.percentile(df1s, 2.5)),
        "delta_f1_ci95_high": float(np.percentile(df1s, 97.5)),
        "delta_acc_point": float(np.mean(y_a == y_true) - np.mean(y_b == y_true)),
        "delta_acc_ci95_low": float(np.percentile(daccs, 2.5)),
        "delta_acc_ci95_high": float(np.percentile(daccs, 97.5)),
    }


def load_cache_dict(paths: list[Path], model_filter: str | None = None) -> dict[str, dict]:
    d: dict[str, dict] = {}
    for p in paths:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if model_filter is not None and rec.get("model") != model_filter:
                    continue
                d[rec["canonical_url_sha256"]] = rec
    return d


def attach_llm(df: pd.DataFrame, cache: dict[str, dict]) -> pd.DataFrame:
    df = df.copy()
    preds, probs, valid = [], [], []
    for sha, sp, sp_prob in zip(
        df["canonical_url_sha256"].values,
        df["specialist_pred"].values,
        df["specialist_prob_cal"].values,
    ):
        rec = cache.get(sha)
        if rec and rec.get("parsed_verdict") in ("phishing", "legitimate"):
            v = 1 if rec["parsed_verdict"] == "phishing" else 0
            p = rec.get("parsed_probability")
            if p is None or (isinstance(p, float) and math.isnan(p)):
                p = 0.75 if v == 1 else 0.25
            preds.append(v)
            probs.append(float(np.clip(p, 0.0, 1.0)))
            valid.append(True)
        else:
            preds.append(int(sp))
            probs.append(float(sp_prob))
            valid.append(False)
    df["llm_pred"] = np.array(preds, dtype=int)
    df["llm_prob"] = np.array(probs, dtype=float)
    df["llm_valid"] = np.array(valid, dtype=bool)
    df["R3_entropy"] = compute_routing_scores(
        "R3_entropy",
        uncal_probs=df["specialist_prob_uncal"].values,
        cal_probs=df["specialist_prob_cal"].values,
    )
    return df


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if np.any(mask):
            ece += (np.sum(mask) / n) * abs(np.mean(y_true[mask]) - np.mean(probs[mask]))
    return float(ece)


def fit_temperature(y_true: np.ndarray, p_uncal: np.ndarray) -> float:
    z = logit(np.clip(p_uncal, 1e-6, 1.0 - 1e-6))

    def nll(T: float) -> float:
        p = np.clip(expit(z / T), 1e-7, 1.0 - 1e-7)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def main() -> None:
    pf = json.load(open("results/policies/policy_freeze.json", "r", encoding="utf-8"))
    tau = float(pf["metadata"]["primary_cutoff_25"])

    # Load E3 (TEST, N=26,886), E4 (CAL_ROUTER, N=13,446), E9 (GPT-5.6 Luna, N=3,000), DEV_POLICY (N=13,446)
    e3_cache = load_cache_dict([Path("results/E3/e3_gemini_test_cache.jsonl")])
    e4_cache = load_cache_dict([Path("results/E4/e4_gemini_cal_cache.jsonl")])
    e9_cache = load_cache_dict([Path("results/E9/e9_openai_luna_cache.jsonl")], model_filter="gpt-5.6-luna")

    df_e3 = attach_llm(pd.read_csv("data/manifests/e3_test_all_26886.csv"), e3_cache)
    df_e4 = attach_llm(pd.read_csv("data/manifests/e4_cal_all_13446.csv"), e4_cache)
    df_e9 = attach_llm(pd.read_csv("data/manifests/e9_shared_test_cohort_3000.csv"), e9_cache)

    dev_frames = []
    for ds in ["EBBU2017", "PWD2021", "URLPhish2025"]:
        mf = pd.read_csv(f"data/manifests/{ds}_manifest_seed42.csv")
        dev_frames.append(mf[mf["split"] == "DEV_POLICY"].copy())
    df_dev = pd.concat(dev_frames, ignore_index=True)
    df_dev["R3_entropy"] = compute_routing_scores(
        "R3_entropy",
        uncal_probs=df_dev["specialist_prob_uncal"].values,
        cal_probs=df_dev["specialist_prob_cal"].values,
    )

    out: dict = {"frozen_tau": tau}

    # =========================================================================
    # 0. Sanity Reconciliation Check
    # =========================================================================
    e3_full = compute_2x2(df_e3["y"].values, df_e3["specialist_pred"].values, df_e3["llm_pred"].values)
    e3_mask = df_e3["R3_entropy"].values >= tau
    e3_top = compute_2x2(
        df_e3["y"].values[e3_mask],
        df_e3["specialist_pred"].values[e3_mask],
        df_e3["llm_pred"].values[e3_mask],
    )
    e9_full = compute_2x2(df_e9["y"].values, df_e9["specialist_pred"].values, df_e9["llm_pred"].values)
    e9_mask = df_e9["R3_entropy"].values >= tau
    e9_top = compute_2x2(
        df_e9["y"].values[e9_mask],
        df_e9["specialist_pred"].values[e9_mask],
        df_e9["llm_pred"].values[e9_mask],
    )
    out["reconciliation"] = {
        "e3_full_net": e3_full["net"],
        "e3_top_net": e3_top["net"],
        "e3_top_esc_count": int(e3_mask.sum()),
        "e9_full_net": e9_full["net"],
        "e9_top_net": e9_top["net"],
        "e9_top_esc_count": int(e9_mask.sum()),
        "e9_valid_llm_frac": float(df_e9["llm_valid"].mean()),
    }

    # =========================================================================
    # 1. Out-of-Sample Breakeven Prediction (E4 -> E3) & Sample-Size Curve
    # =========================================================================
    pred_results = {}
    rng = np.random.default_rng(42)
    sample_sizes = [100, 200, 300, 500, 1000, 2000]
    n_trials = 1000

    for label, g_cal, g_test in [
        ("PWD2021", df_e4[df_e4["dataset"] == "PWD2021"], df_e3[df_e3["dataset"] == "PWD2021"]),
        ("EBBU2017", df_e4[df_e4["dataset"] == "EBBU2017"], df_e3[df_e3["dataset"] == "EBBU2017"]),
        ("URLPhish2025", df_e4[df_e4["dataset"] == "URLPhish2025"], df_e3[df_e3["dataset"] == "URLPhish2025"]),
        ("Pooled", df_e4, df_e3),
    ]:
        cal_2x2 = compute_2x2(g_cal["y"].values, g_cal["specialist_pred"].values, g_cal["llm_pred"].values)
        cal_uncal_pred = (g_cal["specialist_prob_uncal"].values >= 0.5).astype(int)
        cal_uncal_2x2 = compute_2x2(g_cal["y"].values, cal_uncal_pred, g_cal["llm_pred"].values)
        test_2x2 = compute_2x2(g_test["y"].values, g_test["specialist_pred"].values, g_test["llm_pred"].values)
        true_sign = 1 if test_2x2["net"] > 0 else -1

        curve = {}
        y_c = g_cal["y"].values
        s_c = g_cal["specialist_pred"].values
        l_c = g_cal["llm_pred"].values
        n_c = len(g_cal)

        for m in sample_sizes:
            if m > n_c:
                continue
            sign_hits = 0
            abs_err_eps = []
            abs_err_eps_star = []
            abs_err_margin = []
            for _ in range(n_trials):
                idx = rng.choice(n_c, size=m, replace=False)
                sc = s_c[idx] == y_c[idx]
                lc = l_c[idx] == y_c[idx]
                b = np.sum((~sc) & lc)
                c = np.sum(sc & (~lc))
                e = np.sum(~sc)
                n_ok = m - e
                if e == 0:
                    eps_hat = c / max(n_ok, 1)
                    eps_star_hat = 0.0
                else:
                    rho_hat = b / e
                    eps_hat = c / max(n_ok, 1)
                    eps_star_hat = rho_hat * e / max(n_ok, 1)
                margin_hat = eps_star_hat - eps_hat
                pred_sign = 1 if margin_hat > 0 else (-1 if margin_hat < 0 else 0)
                if pred_sign == true_sign:
                    sign_hits += 1
                abs_err_eps.append(abs(eps_hat - test_2x2["eps"]))
                abs_err_eps_star.append(abs(eps_star_hat - test_2x2["eps_star"]))
                abs_err_margin.append(abs(margin_hat - test_2x2["margin"]))

            curve[str(m)] = {
                "sign_accuracy": float(sign_hits / n_trials),
                "mae_eps": float(np.mean(abs_err_eps)),
                "mae_eps_star": float(np.mean(abs_err_eps_star)),
                "mae_margin": float(np.mean(abs_err_margin)),
            }

        pred_results[label] = {
            "e4_full_estimate": {
                "n": cal_2x2["n"],
                "rho": cal_2x2["rho"],
                "eps": cal_2x2["eps"],
                "eps_star": cal_2x2["eps_star"],
                "margin": cal_2x2["margin"],
                "predicted_sign": "+" if cal_2x2["margin"] > 0 else "-",
            },
            "e4_uncal_specialist_estimate": {
                "rho": cal_uncal_2x2["rho"],
                "eps": cal_uncal_2x2["eps"],
                "eps_star": cal_uncal_2x2["eps_star"],
                "margin": cal_uncal_2x2["margin"],
                "predicted_sign": "+" if cal_uncal_2x2["margin"] > 0 else "-",
            },
            "e3_test_truth": {
                "n": test_2x2["n"],
                "rho": test_2x2["rho"],
                "eps": test_2x2["eps"],
                "eps_star": test_2x2["eps_star"],
                "margin": test_2x2["margin"],
                "actual_net": test_2x2["net"],
                "actual_sign": "+" if test_2x2["net"] > 0 else "-",
            },
            "full_e4_sign_correct": (cal_2x2["margin"] > 0) == (test_2x2["net"] > 0),
            "full_e4_uncal_sign_correct": (cal_uncal_2x2["margin"] > 0) == (test_2x2["net"] > 0),
            "abs_err_eps_full_e4": float(abs(cal_2x2["eps"] - test_2x2["eps"])),
            "abs_err_eps_star_full_e4": float(abs(cal_2x2["eps_star"] - test_2x2["eps_star"])),
            "sample_size_curve": curve,
        }

    out["oos_breakeven_prediction"] = pred_results

    # =========================================================================
    # 2. Bootstrap 95% CIs & Per-Corpus F1 at tau = 0.4363
    # =========================================================================
    per_cohort_stats = {}
    for label, df_sub in [
        ("PWD2021", df_e3[df_e3["dataset"] == "PWD2021"]),
        ("EBBU2017", df_e3[df_e3["dataset"] == "EBBU2017"]),
        ("URLPhish2025", df_e3[df_e3["dataset"] == "URLPhish2025"]),
        ("Pooled_E3", df_e3),
        ("E9_GPT56_Pooled", df_e9),
        ("E9_PWD2021", df_e9[df_e9["dataset"] == "PWD2021"]),
        ("E9_EBBU2017", df_e9[df_e9["dataset"] == "EBBU2017"]),
        ("E9_URLPhish2025", df_e9[df_e9["dataset"] == "URLPhish2025"]),
    ]:
        yt = df_sub["y"].values
        ys = df_sub["specialist_pred"].values
        yl = df_sub["llm_pred"].values
        mask = df_sub["R3_entropy"].values >= tau
        yc = ys.copy()
        yc[mask] = yl[mask]

        m_spec = compute_metrics(yt, ys)
        m_tau = compute_metrics(yt, yc)
        m_full = compute_metrics(yt, yl)
        ci_vs_spec = paired_bootstrap_ci(yt, yc, ys, n_boot=1000, seed=42)
        ci_vs_full = paired_bootstrap_ci(yt, yc, yl, n_boot=1000, seed=42)

        per_cohort_stats[label] = {
            "n": int(len(df_sub)),
            "esc_count": int(mask.sum()),
            "esc_pct": float(mask.mean()),
            "specialist_f1": m_spec["f1"],
            "selective_tau_f1": m_tau["f1"],
            "full_llm_f1": m_full["f1"],
            "specialist_acc": m_spec["accuracy"],
            "selective_tau_acc": m_tau["accuracy"],
            "full_llm_acc": m_full["accuracy"],
            "tau_vs_specialist": ci_vs_spec,
            "tau_vs_full": ci_vs_full,
        }

    out["per_cohort_f1_and_ci"] = per_cohort_stats

    # =========================================================================
    # 3. Policy-Space Alternatives (Tuned on E4, Evaluated Out-of-Sample on E3)
    # =========================================================================
    y4 = df_e4["y"].values
    s4 = df_e4["specialist_pred"].values
    p4_s = df_e4["specialist_prob_cal"].values
    l4 = df_e4["llm_pred"].values
    p4_l = df_e4["llm_prob"].values
    m4_tau = df_e4["R3_entropy"].values >= tau

    y3 = df_e3["y"].values
    s3 = df_e3["specialist_pred"].values
    p3_s = df_e3["specialist_prob_cal"].values
    l3 = df_e3["llm_pred"].values
    p3_l = df_e3["llm_prob"].values
    m3_tau = df_e3["R3_entropy"].values >= tau

    policy_space = {}

    # (a) Asymmetric LLM-confidence-gated override: override specialist only when max(p_L, 1-p_L) >= c_L
    conf4_l = np.maximum(p4_l, 1.0 - p4_l)
    conf3_l = np.maximum(p3_l, 1.0 - p3_l)
    for scope, mask4, mask3 in [("full_100pct", np.ones(len(df_e4), dtype=bool), np.ones(len(df_e3), dtype=bool)), ("selective_tau", m4_tau, m3_tau)]:
        best_c, best_f1 = 0.50, -1.0
        for c_cand in np.round(np.linspace(0.50, 0.99, 50), 2):
            ovr = mask4 & (conf4_l >= c_cand)
            yp = s4.copy()
            yp[ovr] = l4[ovr]
            f1 = compute_metrics(y4, yp)["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_c = float(c_cand)
        ovr3 = mask3 & (conf3_l >= best_c)
        yp3 = s3.copy()
        yp3[ovr3] = l3[ovr3]
        t3 = compute_2x2(y3[ovr3], s3[ovr3], l3[ovr3]) if ovr3.any() else {"b": 0, "c": 0, "net": 0}
        m3 = compute_metrics(y3, yp3)
        policy_space[f"llm_conf_gated_{scope}"] = {
            "tuned_c_L_on_E4": best_c,
            "e3_queried_pct": float(mask3.mean()),
            "e3_overridden_pct": float(ovr3.mean()),
            "e3_rescues_b": t3["b"],
            "e3_regressions_c": t3["c"],
            "e3_net": t3["net"],
            "e3_f1": m3["f1"],
            "e3_delta_f1_vs_spec": float(m3["f1"] - per_cohort_stats["Pooled_E3"]["specialist_f1"]),
        }

    # (b) Soft posterior fusion: (1 - alpha) * p_S + alpha * p_L >= 0.5
    for scope, mask4, mask3 in [("full_100pct", np.ones(len(df_e4), dtype=bool), np.ones(len(df_e3), dtype=bool)), ("selective_tau", m4_tau, m3_tau)]:
        best_a, best_f1 = 0.50, -1.0
        for a_cand in np.round(np.linspace(0.05, 1.00, 20), 2):
            pmix = p4_s.copy()
            pmix[mask4] = (1.0 - a_cand) * p4_s[mask4] + a_cand * p4_l[mask4]
            yp = (pmix >= 0.5).astype(int)
            f1 = compute_metrics(y4, yp)["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_a = float(a_cand)
        pmix3 = p3_s.copy()
        pmix3[mask3] = (1.0 - best_a) * p3_s[mask3] + best_a * p3_l[mask3]
        yp3 = s3.copy()
        yp3[mask3] = (pmix3[mask3] >= 0.5).astype(int)
        t3 = compute_2x2(y3, s3, yp3)
        m3 = compute_metrics(y3, yp3)
        policy_space[f"posterior_fusion_{scope}"] = {
            "tuned_alpha_on_E4": best_a,
            "e3_queried_pct": float(mask3.mean()),
            "e3_rescues_b": t3["b"],
            "e3_regressions_c": t3["c"],
            "e3_net": t3["net"],
            "e3_f1": m3["f1"],
            "e3_delta_f1_vs_spec": float(m3["f1"] - per_cohort_stats["Pooled_E3"]["specialist_f1"]),
        }

    # (c) Post-hoc learned deferral gate (Jitkrittum et al. style):
    # Train on E4 discordant pairs (or full E4 estimating P(Cell B | x) and P(Cell C | x))
    def build_gate_features(df_in: pd.DataFrame, include_corpus: bool = False) -> np.ndarray:
        pu = np.clip(df_in["specialist_prob_uncal"].values, 1e-6, 1.0 - 1e-6)
        pc = np.clip(df_in["specialist_prob_cal"].values, 1e-6, 1.0 - 1e-6)
        ent = df_in["R3_entropy"].values
        marg_u = np.abs(2.0 * pu - 1.0)
        marg_c = np.abs(2.0 * pc - 1.0)
        feats = [pu, pc, ent, marg_u, marg_c, np.abs(logit(pu))]
        if include_corpus:
            for ds in ["PWD2021", "EBBU2017", "URLPhish2025"]:
                feats.append((df_in["dataset"].values == ds).astype(float))
        return np.column_stack(feats)

    is_b4 = ((s4 != y4) & (l4 == y4)).astype(int)
    is_c4 = ((s4 == y4) & (l4 != y4)).astype(int)

    for tag, inc_ds in [("learned_gate_corpus_agnostic", False), ("learned_gate_with_corpus", True)]:
        X4 = build_gate_features(df_e4, include_corpus=inc_ds)
        X3 = build_gate_features(df_e3, include_corpus=inc_ds)

        # Logistic regression pair: P(B|x) > P(C|x)
        clf_b = LogisticRegression(max_iter=1000, C=1.0).fit(X4, is_b4)
        clf_c = LogisticRegression(max_iter=1000, C=1.0).fit(X4, is_c4)
        score3_lr = clf_b.predict_proba(X3)[:, 1] - clf_c.predict_proba(X3)[:, 1]
        esc3_lr = score3_lr > 0.0
        yp3_lr = s3.copy()
        yp3_lr[esc3_lr] = l3[esc3_lr]
        t3_lr = compute_2x2(y3[esc3_lr], s3[esc3_lr], l3[esc3_lr]) if esc3_lr.any() else {"b": 0, "c": 0, "net": 0}
        m3_lr = compute_metrics(y3, yp3_lr)

        # HistGradientBoosting pair: P(B|x) > P(C|x)
        gb_b = HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=42).fit(X4, is_b4)
        gb_c = HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=42).fit(X4, is_c4)
        score3_gb = gb_b.predict_proba(X3)[:, 1] - gb_c.predict_proba(X3)[:, 1]
        esc3_gb = score3_gb > 0.0
        yp3_gb = s3.copy()
        yp3_gb[esc3_gb] = l3[esc3_gb]
        t3_gb = compute_2x2(y3[esc3_gb], s3[esc3_gb], l3[esc3_gb]) if esc3_gb.any() else {"b": 0, "c": 0, "net": 0}
        m3_gb = compute_metrics(y3, yp3_gb)

        policy_space[f"{tag}_logreg"] = {
            "e3_esc_count": int(esc3_lr.sum()),
            "e3_esc_pct": float(esc3_lr.mean()),
            "e3_rescues_b": t3_lr["b"],
            "e3_regressions_c": t3_lr["c"],
            "e3_net": t3_lr["net"],
            "e3_f1": m3_lr["f1"],
            "e3_delta_f1_vs_spec": float(m3_lr["f1"] - per_cohort_stats["Pooled_E3"]["specialist_f1"]),
        }
        policy_space[f"{tag}_hgb"] = {
            "e3_esc_count": int(esc3_gb.sum()),
            "e3_esc_pct": float(esc3_gb.mean()),
            "e3_rescues_b": t3_gb["b"],
            "e3_regressions_c": t3_gb["c"],
            "e3_net": t3_gb["net"],
            "e3_f1": m3_gb["f1"],
            "e3_delta_f1_vs_spec": float(m3_gb["f1"] - per_cohort_stats["Pooled_E3"]["specialist_f1"]),
        }

    out["policy_space_comparison"] = policy_space

    # =========================================================================
    # 4. Calibration Comparison: Uncalibrated vs. Temperature Scaling vs. Isotonic
    # =========================================================================
    T_pooled = fit_temperature(df_e4["y"].values, df_e4["specialist_prob_uncal"].values)
    p3_uncal = df_e3["specialist_prob_uncal"].values
    p3_ts_pooled = expit(logit(np.clip(p3_uncal, 1e-6, 1.0 - 1e-6)) / T_pooled)

    # Per-corpus Temperature Scaling (matching per-corpus isotonic)
    T_per_corpus = {}
    p3_ts_per_corpus = np.zeros_like(p3_uncal)
    for ds in ["PWD2021", "EBBU2017", "URLPhish2025"]:
        m4 = df_e4["dataset"].values == ds
        m3 = df_e3["dataset"].values == ds
        T_ds = fit_temperature(df_e4["y"].values[m4], df_e4["specialist_prob_uncal"].values[m4])
        T_per_corpus[ds] = T_ds
        p3_ts_per_corpus[m3] = expit(logit(np.clip(p3_uncal[m3], 1e-6, 1.0 - 1e-6)) / T_ds)

    def eval_matched_rate(scores: np.ndarray, target_count: int) -> dict:
        # Escalate top `target_count` samples by score (exact matched budget)
        order = np.argsort(-scores, kind="mergesort")
        mask = np.zeros(len(scores), dtype=bool)
        mask[order[:target_count]] = True
        yp = s3.copy()
        yp[mask] = l3[mask]
        t = compute_2x2(y3[mask], s3[mask], l3[mask])
        m = compute_metrics(y3, yp)
        return {"esc_count": int(mask.sum()), "esc_pct": float(mask.mean()), "b": t["b"], "c": t["c"], "net": t["net"], "f1": m["f1"]}

    def bin_entropy(p: np.ndarray) -> np.ndarray:
        pc = np.clip(p, 1e-7, 1.0 - 1e-7)
        return -(pc * np.log2(pc) + (1.0 - pc) * np.log2(1.0 - pc))

    s_uncal_ent = bin_entropy(p3_uncal)
    s_ts_pool_ent = bin_entropy(p3_ts_pooled)
    s_ts_corp_ent = bin_entropy(p3_ts_per_corpus)
    s_iso_ent = df_e3["R3_entropy"].values
    s_ts_corp_conf = 1.0 - np.maximum(p3_ts_per_corpus, 1.0 - p3_ts_per_corpus)  # conformal non-conformity

    n_tau = int(m3_tau.sum())  # 3,346 (12.45%)
    n_1757 = 4724  # 17.57% peak
    out["calibration_comparison"] = {
        "T_pooled": T_pooled,
        "T_per_corpus": T_per_corpus,
        "test_ece": {
            "uncalibrated": compute_ece(y3, p3_uncal),
            "temp_scaling_pooled": compute_ece(y3, p3_ts_pooled),
            "temp_scaling_per_corpus": compute_ece(y3, p3_ts_per_corpus),
            "isotonic_per_corpus": compute_ece(y3, p3_s),
        },
        "test_brier": {
            "uncalibrated": float(np.mean((p3_uncal - y3) ** 2)),
            "temp_scaling_pooled": float(np.mean((p3_ts_pooled - y3) ** 2)),
            "temp_scaling_per_corpus": float(np.mean((p3_ts_per_corpus - y3) ** 2)),
            "isotonic_per_corpus": float(np.mean((p3_s - y3) ** 2)),
        },
        "spearman_ts_ent_vs_conformal": float(spearmanr(s_ts_corp_ent, s_ts_corp_conf).statistic),
        "spearman_ts_pool_vs_uncal": float(spearmanr(s_ts_pool_ent, s_uncal_ent).statistic),
        "matched_12_45_pct": {
            "uncalibrated_entropy": eval_matched_rate(s_uncal_ent, n_tau),
            "temp_scaling_pooled_entropy": eval_matched_rate(s_ts_pool_ent, n_tau),
            "temp_scaling_per_corpus_entropy": eval_matched_rate(s_ts_corp_ent, n_tau),
            "isotonic_per_corpus_entropy_exact_tau": {
                "esc_count": int(m3_tau.sum()),
                "esc_pct": float(m3_tau.mean()),
                "b": e3_top["b"],
                "c": e3_top["c"],
                "net": e3_top["net"],
                "f1": per_cohort_stats["Pooled_E3"]["selective_tau_f1"],
            },
        },
        "matched_17_57_pct": {
            "uncalibrated_entropy": eval_matched_rate(s_uncal_ent, n_1757),
            "temp_scaling_per_corpus_entropy": eval_matched_rate(s_ts_corp_ent, n_1757),
            "isotonic_per_corpus_entropy": eval_matched_rate(s_iso_ent, n_1757),
        },
    }

    # =========================================================================
    # 5. Budget-Mapping Diagnostics (DEV_POLICY vs. TEST)
    # =========================================================================
    budget_diag = {}
    for b_str in ["0.05", "0.1", "0.15", "0.2", "0.25", "0.35", "0.5", "0.7", "0.95"]:
        t = float(pf["thresholds"]["R3_entropy"][b_str])
        dev_real = float((df_dev["R3_entropy"].values >= t).mean())
        test_real = float((df_e3["R3_entropy"].values >= t).mean())
        # Also compute DEV_POLICY calibrated base-2 quantile for comparison
        b_val = float(b_str)
        t_cal_dev = float(np.percentile(df_dev["R3_entropy"].values, (1.0 - b_val) * 100.0))
        test_real_cal_aligned = float((df_e3["R3_entropy"].values >= t_cal_dev).mean())
        budget_diag[b_str] = {
            "frozen_tau_uncal_nats": t,
            "dev_realized_pct_on_cal_bits": dev_real,
            "test_realized_pct_on_cal_bits": test_real,
            "cal_bits_dev_quantile_tau": t_cal_dev,
            "test_realized_pct_if_cal_aligned": test_real_cal_aligned,
        }
    out["budget_mapping_diagnostics"] = budget_diag

    # =========================================================================
    # 6. Deduplicated Globally Disjoint Sensitivity Check (R1 #4)
    # =========================================================================
    # (a) Cross-split clean (already in clean_disjoint_test_metrics.json):
    #     remove TEST URLs appearing in TRAIN/DEV/CAL of ANY corpus + deduplicate TEST
    non_test_hashes = set()
    for f in sorted(glob.glob("data/manifests/*_manifest_seed42.csv")):
        mf = pd.read_csv(f)
        non_test_hashes.update(mf[mf["split"] != "TEST"]["canonical_url_sha256"])

    # Also build corpus-unique URLs (present in ONLY 1 source corpus across ALL splits)
    url_to_corpora: dict[str, set[str]] = {}
    for ds in ["PWD2021", "EBBU2017", "URLPhish2025"]:
        mf = pd.read_csv(f"data/manifests/{ds}_manifest_seed42.csv")
        for h in mf["canonical_url_sha256"].unique():
            url_to_corpora.setdefault(h, set()).add(ds)
    unique_to_one_corpus = {h for h, cs in url_to_corpora.items() if len(cs) == 1}

    df_dedup_strict = df_e3[df_e3["canonical_url_sha256"].isin(unique_to_one_corpus)].drop_duplicates("canonical_url_sha256")
    m_dedup = df_dedup_strict["R3_entropy"].values >= tau
    yc_dedup = df_dedup_strict["specialist_pred"].values.copy()
    yc_dedup[m_dedup] = df_dedup_strict["llm_pred"].values[m_dedup]

    out["strict_zero_overlap_sensitivity"] = {
        "n_strict_single_corpus_urls": int(len(df_dedup_strict)),
        "specialist_f1": compute_metrics(df_dedup_strict["y"].values, df_dedup_strict["specialist_pred"].values)["f1"],
        "selective_tau_f1": compute_metrics(df_dedup_strict["y"].values, yc_dedup)["f1"],
        "full_llm_f1": compute_metrics(df_dedup_strict["y"].values, df_dedup_strict["llm_pred"].values)["f1"],
        "full_2x2": compute_2x2(df_dedup_strict["y"].values, df_dedup_strict["specialist_pred"].values, df_dedup_strict["llm_pred"].values),
        "top_zone_2x2": compute_2x2(
            df_dedup_strict["y"].values[m_dedup],
            df_dedup_strict["specialist_pred"].values[m_dedup],
            df_dedup_strict["llm_pred"].values[m_dedup],
        ),
        "esc_pct": float(m_dedup.mean()),
    }

    # =========================================================================
    # 7. URL-Phish 2025 Cell C Shortcut Attribution (R2 #6)
    # =========================================================================
    df_up = df_e3[df_e3["dataset"] == "URLPhish2025"].copy()
    schemes, has_path, tlds = [], [], []
    for u in df_up["canonical_url"].astype(str).values:
        parsed = urlparse(u if "://" in u else "http://" + u)
        schemes.append(parsed.scheme.lower())
        has_path.append(len(parsed.path.strip("/")) > 0)
        host = parsed.hostname or ""
        parts = host.split(".")
        tlds.append(parts[-1] if len(parts) >= 2 else "")
    df_up["scheme"] = schemes
    df_up["has_path"] = has_path
    df_up["tld"] = tlds
    df_up["is_cell_c"] = (df_up["specialist_pred"] == df_up["y"]) & (df_up["llm_pred"] != df_up["y"])
    df_up["is_cell_c_fn"] = df_up["is_cell_c"] & (df_up["y"] == 1)
    df_up["is_cell_c_fp"] = df_up["is_cell_c"] & (df_up["y"] == 0)

    c_fn = df_up[df_up["is_cell_c_fn"]]
    phish_all = df_up[df_up["y"] == 1]
    legit_all = df_up[df_up["y"] == 0]
    out["urlphish25_cell_c_attribution"] = {
        "n_total": int(len(df_up)),
        "n_cell_c": int(df_up["is_cell_c"].sum()),
        "n_cell_c_fn": int(len(c_fn)),
        "n_cell_c_fp": int(df_up["is_cell_c_fp"].sum()),
        "pct_https_in_phish_all": float((phish_all["scheme"] == "https").mean()),
        "pct_https_in_legit_all": float((legit_all["scheme"] == "https").mean()),
        "pct_https_in_cell_c_fn": float((c_fn["scheme"] == "https").mean()),
        "pct_http_in_phish_all": float((phish_all["scheme"] == "http").mean()),
        "pct_http_in_cell_c_fn": float((c_fn["scheme"] == "http").mean()),
        "pct_no_path_in_phish_all": float((~phish_all["has_path"]).mean()),
        "pct_no_path_in_legit_all": float((~legit_all["has_path"]).mean()),
        "pct_no_path_in_cell_c_fn": float((~c_fn["has_path"]).mean()),
        "top_tlds_in_cell_c_fn": {str(k): int(v) for k, v in c_fn["tld"].value_counts().head(8).items()},
        "top_tlds_in_phish_all": {str(k): int(v) for k, v in phish_all["tld"].value_counts().head(8).items()},
        "top_tlds_in_legit_all": {str(k): int(v) for k, v in legit_all["tld"].value_counts().head(8).items()},
        "pct_cell_c_fn_in_bottom_zone": float((c_fn["R3_entropy"] < tau).mean()),
    }

    out_path = Path("results/artifacts/camera_ready_offline_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Print concise readable summary
    print("=== RECONCILIATION ===")
    print(json.dumps(out["reconciliation"], indent=2))
    print("\n=== 1. OOS BREAKEVEN PREDICTION (E4 -> E3) ===")
    for k, v in pred_results.items():
        e4e = v["e4_full_estimate"]
        e3t = v["e3_test_truth"]
        c200 = v["sample_size_curve"]["200"]
        c500 = v["sample_size_curve"]["500"]
        c1000 = v["sample_size_curve"]["1000"]
        print(
            f"{k:12s} | E4 pred={e4e['predicted_sign']} (eps={e4e['eps']*100:.2f}%, eps*={e4e['eps_star']*100:.2f}%) "
            f"-> E3 truth={e3t['actual_sign']} ({e3t['actual_net']:+d}) | "
            f"SignAcc m=200:{c200['sign_accuracy']*100:.1f}%, m=500:{c500['sign_accuracy']*100:.1f}%, m=1000:{c1000['sign_accuracy']*100:.1f}% | "
            f"MAE(margin) m=500:{c500['mae_margin']*100:.2f}pp"
        )
    print("\n=== 2. PER-COHORT F1 & BOOTSTRAP 95% CI (tau vs Specialist) ===")
    for k, v in per_cohort_stats.items():
        ci = v["tau_vs_specialist"]
        print(
            f"{k:16s} | Spec={v['specialist_f1']:.4f} | tau={v['selective_tau_f1']:.4f} "
            f"(dF1={ci['delta_f1_point']:+.4f}, 95% CI [{ci['delta_f1_ci95_low']:+.4f}, {ci['delta_f1_ci95_high']:+.4f}]) "
            f"| Full={v['full_llm_f1']:.4f} | Esc={v['esc_pct']*100:.2f}%"
        )
    print("\n=== 3. POLICY SPACE COMPARISON (Tuned on E4 -> Tested on E3) ===")
    for k, v in policy_space.items():
        print(f"{k:36s} | {v}")
    print("\n=== 4. CALIBRATION COMPARISON ===")
    print(json.dumps(out["calibration_comparison"], indent=2))
    print("\n=== 6. STRICT ZERO-OVERLAP SENSITIVITY ===")
    print(json.dumps(out["strict_zero_overlap_sensitivity"], indent=2))
    print("\n=== 7. URL-PHISH 2025 CELL C SHORTCUT ATTRIBUTION ===")
    print(json.dumps(out["urlphish25_cell_c_attribution"], indent=2))


if __name__ == "__main__":
    main()
