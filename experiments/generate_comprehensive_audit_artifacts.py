#!/usr/bin/env python3
"""
[ARCHIVED v1 EVALUATOR — SUPERSEDED BY experiments/reproduce_v2_all.py]
Retained for provenance of initial submission artifacts (results/E5/, results/E3/e3_dataset_breakdown.json).
For camera-ready v2 reproduction, run `experiments/reproduce_v2_all.py --verify`.
"""

import glob
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binom

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.llm.cache import LLMCache
from src.routing.policies import compute_routing_scores


def compute_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    acc = (tp + tn) / max(len(y_true), 1)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_2x2_and_mcnemar(y_true, y_spec, y_llm):
    spec_correct = (y_spec == y_true)
    llm_correct = (y_llm == y_true)
    
    cell_a = int(np.sum(spec_correct & llm_correct))
    cell_b = int(np.sum((~spec_correct) & llm_correct))
    cell_c = int(np.sum(spec_correct & (~llm_correct)))
    cell_d = int(np.sum((~spec_correct) & (~llm_correct)))
    
    # Error direction decomposition in Cell C
    cell_c_mask = spec_correct & (~llm_correct)
    c_benign_fp = int(np.sum(cell_c_mask & (y_true == 0)))
    c_phish_fn = int(np.sum(cell_c_mask & (y_true == 1)))
    
    n_disc = cell_b + cell_c
    if n_disc > 0:
        k = min(cell_b, cell_c)
        p_val = float(2.0 * binom.cdf(k, n_disc, 0.5))
        p_val = min(p_val, 1.0)
    else:
        p_val = 1.0
        
    net = cell_b - cell_c
    pop_nrr = net / max(len(y_true), 1)
    disc_adv = net / max(n_disc, 1)
    
    # Breakeven inequality parameters
    e_errors = int(np.sum(~spec_correct))
    n_non_errors = len(y_true) - e_errors
    rho_rescue = cell_b / max(e_errors, 1)
    eps_error_on_correct = cell_c / max(n_non_errors, 1)
    eps_breakeven = rho_rescue * (e_errors / max(n_non_errors, 1))
    
    return {
        "cell_a_waste": cell_a,
        "cell_b_rescue": cell_b,
        "cell_c_regression": cell_c,
        "cell_c_benign_fp": c_benign_fp,
        "cell_c_phish_fn": c_phish_fn,
        "cell_d_hard": cell_d,
        "net_rescues": net,
        "population_nrr": pop_nrr,
        "discordant_advantage": disc_adv,
        "mcnemar_p_value": p_val,
        "rho_rescue": rho_rescue,
        "eps_error_on_correct": eps_error_on_correct,
        "eps_breakeven": eps_breakeven,
        "breakeven_holds": eps_error_on_correct < eps_breakeven,
    }


def load_e3_dataset():
    manifest_path = Path("data/manifests/e3_test_all_26886.csv")
    cache_path = Path("results/E3/e3_gemini_test_cache.jsonl")
    
    cache_dict = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                cache_dict[rec["canonical_url_sha256"]] = rec
                
    df_res = pd.read_csv(manifest_path)
    shas = df_res["canonical_url_sha256"].values
    spec_preds = df_res["specialist_pred"].values
    
    llm_preds = []
    for sha, sp in zip(shas, spec_preds):
        rec = cache_dict.get(sha)
        if not rec:
            raise RuntimeError(f"Missing cache for {sha}")
        v = rec.get("parsed_verdict")
        llm_preds.append(1 if v == "phishing" else (0 if v == "legitimate" else int(sp)))
    df_res["llm_pred"] = llm_preds
    
    p_uncal = df_res["specialist_prob_uncal"].values
    p_cal = df_res["specialist_prob_cal"].values
    
    df_res["R3_entropy"] = compute_routing_scores("R3_entropy", uncal_probs=p_uncal, cal_probs=p_cal, seed=42)
    df_res["R2_margin"] = compute_routing_scores("R2_margin", uncal_probs=p_uncal, cal_probs=p_cal, seed=42)
    df_res["R1_max_prob"] = compute_routing_scores("R1_max_prob", uncal_probs=p_uncal, cal_probs=p_cal, seed=42)
    df_res["R0_random"] = compute_routing_scores("R0_random", uncal_probs=p_uncal, cal_probs=p_cal, seed=42)
    
    return df_res


def load_e9_dataset():
    manifest_path = Path("data/manifests/e9_shared_test_cohort_3000.csv")
    cache_dict = {}
    for cpath in [Path("results/E9/e9_openai_luna_cache.jsonl"), Path("results/E3/e3_gemini_test_cache.jsonl")]:
        if cpath.exists():
            with open(cpath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        cache_dict[rec["canonical_url_sha256"]] = rec
                        
    df_res = pd.read_csv(manifest_path)
    shas = df_res["canonical_url_sha256"].values
    spec_preds = df_res["specialist_pred"].values
    
    llm_preds = []
    for sha, sp in zip(shas, spec_preds):
        rec = cache_dict.get(sha)
        if not rec:
            raise RuntimeError(f"Missing cache for {sha}")
        v = rec.get("parsed_verdict")
        llm_preds.append(1 if v == "phishing" else (0 if v == "legitimate" else int(sp)))
    df_res["llm_pred"] = llm_preds
    
    p_uncal = df_res["specialist_prob_uncal"].values
    p_cal = df_res["specialist_prob_cal"].values
    df_res["R3_entropy"] = compute_routing_scores("R3_entropy", uncal_probs=p_uncal, cal_probs=p_cal, seed=42)
    return df_res


def main():
    print("Loading E3 dataset (N=26,886)...")
    df_e3 = load_e3_dataset()
    
    print("Loading Policy Freeze...")
    policy_freeze_path = Path("results/policies/policy_freeze.json")
    with open(policy_freeze_path, "r", encoding="utf-8") as f:
        policy_freeze = json.load(f)
    
    thresholds = policy_freeze["thresholds"]
    budgets = policy_freeze["metadata"]["budget_grid"]
    
    # -------------------------------------------------------------
    # 1. Generate results/E5/e5_frontier.json (All routers & budgets)
    # -------------------------------------------------------------
    print("\nEvaluating Pre-Registered Frontier for R0, R1, R2, R3 on E3 (N=26,886)...")
    spec_metrics = compute_metrics(df_e3["y"].values, df_e3["specialist_pred"].values)
    base_f1 = spec_metrics["f1"]
    
    frontier_results = {}
    
    router_map = {
        "R3_entropy": ("R3_entropy", thresholds["R3_entropy"]),
        "R2_margin": ("R2_margin", thresholds["R2_margin"]),
        "R1_max_prob": ("R1_max_prob", thresholds["R1_max_prob"]),
        "R0_random": ("R0_random", thresholds["R0_random"]),
    }
    
    for router_name, (signal_col, thresh_dict) in router_map.items():
        router_entries = []
        for b in budgets:
            b_str = str(b)
            th = thresh_dict.get(b_str)
            if th is None:
                continue
            
            if math.isinf(th) and th > 0:
                escalate_mask = np.zeros(len(df_e3), dtype=bool)
            elif math.isinf(th) and th < 0:
                escalate_mask = np.ones(len(df_e3), dtype=bool)
            else:
                escalate_mask = (df_e3[signal_col].values >= th)
            
            n_escalated = int(np.sum(escalate_mask))
            frac_escalated = n_escalated / len(df_e3)
            
            # Cascade prediction
            y_cascade = df_e3["specialist_pred"].values.copy()
            y_cascade[escalate_mask] = df_e3["llm_pred"].values[escalate_mask]
            
            casc_metrics = compute_metrics(df_e3["y"].values, y_cascade)
            delta_f1 = casc_metrics["f1"] - base_f1
            delta_acc = casc_metrics["accuracy"] - spec_metrics["accuracy"]
            
            # On escalated subset
            if n_escalated > 0:
                tax = compute_2x2_and_mcnemar(
                    df_e3["y"].values[escalate_mask],
                    df_e3["specialist_pred"].values[escalate_mask],
                    df_e3["llm_pred"].values[escalate_mask],
                )
                b_rescues = tax["cell_b_rescue"]
                c_regressions = tax["cell_c_regression"]
                net_rescues = b_rescues - c_regressions
                escalated_nrr = net_rescues / n_escalated
                pop_nrr = net_rescues / len(df_e3)
            else:
                b_rescues = 0
                c_regressions = 0
                net_rescues = 0
                escalated_nrr = 0.0
                pop_nrr = 0.0
                tax = {"cell_a_waste": 0, "cell_b_rescue": 0, "cell_c_regression": 0, "cell_d_hard": 0}
            
            router_entries.append({
                "nominal_budget": b,
                "frozen_threshold": th if not math.isinf(th) else ("+inf" if th > 0 else "-inf"),
                "escalated_count": n_escalated,
                "realized_escalation_rate": frac_escalated,
                "cascade_accuracy": casc_metrics["accuracy"],
                "delta_accuracy": delta_acc,
                "cascade_f1": casc_metrics["f1"],
                "delta_f1": delta_f1,
                "rescues_b": b_rescues,
                "regressions_c": c_regressions,
                "net_rescues": net_rescues,
                "escalated_nrr": escalated_nrr,
                "population_nrr": pop_nrr,
                "cell_a_waste": tax["cell_a_waste"],
                "cell_d_hard": tax["cell_d_hard"],
                "confusion_matrix": {
                    "tp": casc_metrics["tp"],
                    "fp": casc_metrics["fp"],
                    "fn": casc_metrics["fn"],
                    "tn": casc_metrics["tn"],
                }
            })
            
        frontier_results[router_name] = router_entries
    
    Path("results/E5").mkdir(parents=True, exist_ok=True)
    with open("results/E5/e5_frontier.json", "w", encoding="utf-8") as f:
        json.dump(frontier_results, f, indent=2)
    print("✅ Exported results/E5/e5_frontier.json")
    
    # -------------------------------------------------------------
    # 2. Per-Dataset Breakdown on E3 (EBBU2017, PWD2021, URLPhish2025)
    # -------------------------------------------------------------
    print("\nComputing per-dataset breakdown on E3 (N=26,886)...")
    dataset_breakdown = {}
    cutoff_25 = policy_freeze["metadata"]["primary_cutoff_25"]
    
    for dname, group in df_e3.groupby("dataset"):
        tax_full = compute_2x2_and_mcnemar(group["y"].values, group["specialist_pred"].values, group["llm_pred"].values)
        
        # Partition into top 25% uncertainty vs bottom 75%
        routed_mask = (group["R3_entropy"].values >= cutoff_25)
        tax_routed = compute_2x2_and_mcnemar(
            group["y"].values[routed_mask],
            group["specialist_pred"].values[routed_mask],
            group["llm_pred"].values[routed_mask],
        )
        tax_intercepted = compute_2x2_and_mcnemar(
            group["y"].values[~routed_mask],
            group["specialist_pred"].values[~routed_mask],
            group["llm_pred"].values[~routed_mask],
        )
        
        dataset_breakdown[dname] = {
            "n_samples": len(group),
            "specialist_accuracy": float(np.mean(group["y"].values == group["specialist_pred"].values)),
            "full_escalation": tax_full,
            "top_25_uncertainty_zone": {
                "n_routed": int(np.sum(routed_mask)),
                "routed_pct": float(np.mean(routed_mask)),
                "rescues_b": tax_routed["cell_b_rescue"],
                "regressions_c": tax_routed["cell_c_regression"],
                "net_advantage": tax_routed["net_rescues"],
            },
            "bot_75_certainty_zone": {
                "n_intercepted": int(np.sum(~routed_mask)),
                "intercepted_pct": float(np.mean(~routed_mask)),
                "rescues_b": tax_intercepted["cell_b_rescue"],
                "regressions_c": tax_intercepted["cell_c_regression"],
                "net_advantage": tax_intercepted["net_rescues"],
            }
        }
    
    with open("results/E3/e3_dataset_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(dataset_breakdown, f, indent=2)
    print("✅ Exported results/E3/e3_dataset_breakdown.json")
    
    # -------------------------------------------------------------
    # 3. E9 (OpenAI GPT-5.6 Luna) 2-Zone Partition & Frontier
    # -------------------------------------------------------------
    print("\nComputing E9 (GPT-5.6 Luna, N=3,000) 2-Zone Partition & Frontier...")
    df_e9 = load_e9_dataset()
    
    e9_tax_full = compute_2x2_and_mcnemar(df_e9["y"].values, df_e9["specialist_pred"].values, df_e9["llm_pred"].values)
    e9_routed_mask = (df_e9["R3_entropy"].values >= cutoff_25)
    e9_tax_routed = compute_2x2_and_mcnemar(
        df_e9["y"].values[e9_routed_mask],
        df_e9["specialist_pred"].values[e9_routed_mask],
        df_e9["llm_pred"].values[e9_routed_mask],
    )
    e9_tax_intercepted = compute_2x2_and_mcnemar(
        df_e9["y"].values[~e9_routed_mask],
        df_e9["specialist_pred"].values[~e9_routed_mask],
        df_e9["llm_pred"].values[~e9_routed_mask],
    )
    
    e9_analysis = {
        "metadata": {
            "n_samples": len(df_e9),
            "model": "gpt-5.6-luna",
            "provider": "openai",
            "frozen_cutoff_25": cutoff_25,
        },
        "full_escalation_2x2": e9_tax_full,
        "two_zone_advantage_partition": {
            "top_25_uncertainty_zone": {
                "n_routed": int(np.sum(e9_routed_mask)),
                "routed_pct": float(np.mean(e9_routed_mask)),
                "rescues_b": e9_tax_routed["cell_b_rescue"],
                "regressions_c": e9_tax_routed["cell_c_regression"],
                "net_advantage": e9_tax_routed["net_rescues"],
                "mcnemar_p": e9_tax_routed["mcnemar_p_value"],
            },
            "bot_75_certainty_zone": {
                "n_intercepted": int(np.sum(~e9_routed_mask)),
                "intercepted_pct": float(np.mean(~e9_routed_mask)),
                "rescues_b": e9_tax_intercepted["cell_b_rescue"],
                "regressions_c": e9_tax_intercepted["cell_c_regression"],
                "net_advantage": e9_tax_intercepted["net_rescues"],
                "mcnemar_p": e9_tax_intercepted["mcnemar_p_value"],
            }
        }
    }
    
    with open("results/E9/e9_openai_luna_partition_and_frontier.json", "w", encoding="utf-8") as f:
        json.dump(e9_analysis, f, indent=2)
    print("✅ Exported results/E9/e9_openai_luna_partition_and_frontier.json")
    
    # -------------------------------------------------------------
    # 4. Clean Globally Disjoint Test Cohort (N=25,323)
    # -------------------------------------------------------------
    print("\nComputing Clean Globally Disjoint Test Cohort...")
    train_hashes = set()
    dev_hashes = set()
    for f in sorted(glob.glob("data/manifests/*_manifest_seed42.csv")):
        mf_df = pd.read_csv(f)
        train_hashes.update(mf_df[mf_df["split"] == "TRAIN"]["canonical_url_sha256"])
        dev_hashes.update(mf_df[mf_df["split"] == "DEV_POLICY"]["canonical_url_sha256"])
        
    clean_mask = (~df_e3["canonical_url_sha256"].isin(train_hashes)) & (~df_e3["canonical_url_sha256"].isin(dev_hashes))
    df_clean = df_e3[clean_mask].drop_duplicates(subset=["canonical_url_sha256"]).copy()
    
    clean_spec_m = compute_metrics(df_clean["y"].values, df_clean["specialist_pred"].values)
    clean_full_tax = compute_2x2_and_mcnemar(df_clean["y"].values, df_clean["specialist_pred"].values, df_clean["llm_pred"].values)
    clean_full_m = compute_metrics(df_clean["y"].values, df_clean["llm_pred"].values)
    
    clean_routed_mask = (df_clean["R3_entropy"].values >= cutoff_25)
    y_clean_casc = df_clean["specialist_pred"].values.copy()
    y_clean_casc[clean_routed_mask] = df_clean["llm_pred"].values[clean_routed_mask]
    clean_casc_m = compute_metrics(df_clean["y"].values, y_clean_casc)
    clean_routed_tax = compute_2x2_and_mcnemar(
        df_clean["y"].values[clean_routed_mask],
        df_clean["specialist_pred"].values[clean_routed_mask],
        df_clean["llm_pred"].values[clean_routed_mask],
    )
    
    clean_metrics_payload = {
        "metadata": {
            "total_clean_samples": len(df_clean),
            "description": "Globally deduplicated test cohort strictly disjoint from TRAIN and DEV of all datasets",
            "frozen_cutoff_25": cutoff_25,
        },
        "specialist": clean_spec_m,
        "full_escalation": {
            "metrics": clean_full_m,
            "taxonomy": clean_full_tax,
        },
        "selective_r3": {
            "metrics": clean_casc_m,
            "routed_count": int(np.sum(clean_routed_mask)),
            "routed_pct": float(np.mean(clean_routed_mask)),
            "taxonomy": clean_routed_tax,
        }
    }
    
    Path("results/artifacts").mkdir(parents=True, exist_ok=True)
    with open("results/artifacts/clean_disjoint_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(clean_metrics_payload, f, indent=2)
    print("✅ Exported results/artifacts/clean_disjoint_test_metrics.json")
    
    # -------------------------------------------------------------
    # 5. Cell C Regression Decomposition (FP vs FN)
    # -------------------------------------------------------------
    e3_full_tax = compute_2x2_and_mcnemar(df_e3["y"].values, df_e3["specialist_pred"].values, df_e3["llm_pred"].values)
    cell_c_breakdown = {
        "total_regressions_c": e3_full_tax["cell_c_regression"],
        "benign_false_positives": e3_full_tax["cell_c_benign_fp"],
        "benign_fp_pct": e3_full_tax["cell_c_benign_fp"] / e3_full_tax["cell_c_regression"] * 100,
        "phishing_false_negatives": e3_full_tax["cell_c_phish_fn"],
        "phishing_fn_pct": e3_full_tax["cell_c_phish_fn"] / e3_full_tax["cell_c_regression"] * 100,
    }
    with open("results/artifacts/cell_c_error_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(cell_c_breakdown, f, indent=2)
    print("✅ Exported results/artifacts/cell_c_error_breakdown.json")

    # -------------------------------------------------------------
    # 6. Print Comprehensive Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print("  SUMMARY: AUTHORITATIVE BENCHMARK OVERVIEW (Frozen cutoff tau = 0.4363)")
    print("=" * 85)
    print(f"Cohort A (Full E3, N=26,886):")
    print(f"  Specialist F1: {spec_metrics['f1']:.4f}")
    r3_25_entry = [e for e in frontier_results['R3_entropy'] if e['nominal_budget'] == 0.25][0]
    print(f"  Selective R3 F1: {r3_25_entry['cascade_f1']:.4f} (ΔF1 = {r3_25_entry['delta_f1']:+.4f}, Routed = {r3_25_entry['escalated_count']} [{r3_25_entry['realized_escalation_rate']*100:.2f}%], B={r3_25_entry['rescues_b']}, C={r3_25_entry['regressions_c']}, Net={r3_25_entry['net_rescues']:+d})")
    full_entry = frontier_results['R3_entropy'][-1]
    print(f"  Full LLM F1:     {full_entry['cascade_f1']:.4f} (ΔF1 = {full_entry['delta_f1']:+.4f}, B={full_entry['rescues_b']}, C={full_entry['regressions_c']}, Net={full_entry['net_rescues']:+d}, p = {e3_full_tax['mcnemar_p_value']:.2e})")
    
    print(f"\nCohort B (Clean Disjoint Holdout, N={len(df_clean)}):")
    print(f"  Specialist F1: {clean_spec_m['f1']:.4f}")
    print(f"  Selective R3 F1: {clean_casc_m['f1']:.4f} (ΔF1 = {clean_casc_m['f1'] - clean_spec_m['f1']:+.4f}, Routed = {clean_metrics_payload['selective_r3']['routed_count']} [{clean_metrics_payload['selective_r3']['routed_pct']*100:.2f}%], B={clean_routed_tax['cell_b_rescue']}, C={clean_routed_tax['cell_c_regression']}, Net={clean_routed_tax['net_rescues']:+d})")
    print(f"  Full LLM F1:     {clean_full_m['f1']:.4f} (ΔF1 = {clean_full_m['f1'] - clean_spec_m['f1']:+.4f}, B={clean_full_tax['cell_b_rescue']}, C={clean_full_tax['cell_c_regression']}, Net={clean_full_tax['net_rescues']:+d}, p = {clean_full_tax['mcnemar_p_value']:.2e})")
    
    print(f"\nCell C Error Composition (N=1,553 regressions):")
    print(f"  Phishing False Negatives (Missed Phish): {cell_c_breakdown['phishing_false_negatives']} ({cell_c_breakdown['phishing_fn_pct']:.1f}%)")
    print(f"  Benign False Positives (False Alarms):   {cell_c_breakdown['benign_false_positives']} ({cell_c_breakdown['benign_fp_pct']:.1f}%)")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
