#!/usr/bin/env python3
"""
v2 Table policies (R1-2 / R2-3).

Mirrors run_camera_ready_offline_audit.py policy rows, with two v2 updates:
  * E3 = v2 composite (amendment A-2); E4 = August Gemini cache with P-49 URLs removed;
  * all E4-tuned quantities (c_L, alpha, learned gates, temperature) use 5-fold OOF
    calibrated specialist probabilities on E4 (per corpus, StratifiedKFold seed 42);
  * operating points: primary b=0.25 (policy_freeze_v2) and tau*_BE (tau_BE_v2).
Default output: results/v2_runs/v2_policies.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.v2.io import (
    REPO_ROOT,
    RESULTS_V2_DIR,
    V2DataBundle,
    manifest_to_arrays,
    verify_json_file,
    write_json,
)
from src.v2.metrics import binary_entropy_bits, compute_f1, fit_oof_isotonic_by_corpus

DEFAULT_OUT = RESULTS_V2_DIR / "v2_policies.json"


def _summarize_policy(
    dataset: dict[str, Any],
    y_pred: np.ndarray,
    queried_mask: np.ndarray,
) -> dict[str, Any]:
    sc = dataset["s"] == dataset["y"]
    pc = y_pred == dataset["y"]
    b_cnt = int((~sc & pc).sum())
    c_cnt = int((sc & ~pc).sum())
    return {
        "queried_pct": float(queried_mask.mean() * 100.0),
        "B": b_cnt,
        "C": c_cnt,
        "net": b_cnt - c_cnt,
        "F1": float(compute_f1(dataset["y"], y_pred)),
    }


def _tune_conf_gate(
    e4: dict[str, Any],
    e3: dict[str, Any],
    m4: np.ndarray,
    m3: np.ndarray,
) -> tuple[float, np.ndarray]:
    c4 = np.maximum(e4["pl"], 1.0 - e4["pl"])
    c3 = np.maximum(e3["pl"], 1.0 - e3["pl"])
    grid = np.round(np.linspace(0.50, 0.99, 50), 2)
    best = max(
        grid,
        key=lambda c: (
            compute_f1(e4["y"], np.where(m4 & (c4 >= c), e4["l"], e4["s"])),
            -c,
        ),
    )
    return float(best), np.where(m3 & (c3 >= best), e3["l"], e3["s"])


def _tune_soft_fusion(
    e4: dict[str, Any],
    e3: dict[str, Any],
    m4: np.ndarray,
    m3: np.ndarray,
) -> tuple[float, np.ndarray]:
    def fused(dataset: dict[str, Any], mask: np.ndarray, alpha: float) -> np.ndarray:
        prob = dataset["pc"].copy()
        prob[mask] = (1.0 - alpha) * dataset["pc"][mask] + alpha * dataset["pl"][mask]
        return np.where(mask, (prob >= 0.5).astype(int), dataset["s"])

    grid = np.round(np.linspace(0.05, 1.00, 20), 2)
    best = max(grid, key=lambda a: (compute_f1(e4["y"], fused(e4, m4, a)), -a))
    return float(best), fused(e3, m3, best)


def _matched_topk_mask(score: np.ndarray, k: int) -> np.ndarray:
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(-score, kind="mergesort")[:k]] = True
    return mask


def _gate_features(dataset: dict[str, Any]) -> np.ndarray:
    pu = np.clip(dataset["pu"], 1e-6, 1.0 - 1e-6)
    pc = np.clip(dataset["pc"], 1e-6, 1.0 - 1e-6)
    return np.column_stack(
        [
            pu,
            pc,
            dataset["h"],
            np.abs(2.0 * pu - 1.0),
            np.abs(2.0 * pc - 1.0),
            np.abs(logit(pu)),
        ]
    )


def run(bundle: V2DataBundle | None = None) -> dict[str, Any]:
    """Computes full v2 Table policies results dictionary."""
    data = bundle if bundle is not None else V2DataBundle.load()
    taus = {
        "b0.25": data.policy_freeze_v2["metadata"]["primary_tau"],
        "tauBE": data.tau_be_v2["tau_star_BE"],
    }

    e4 = manifest_to_arrays(data.e4_rows_p49_excluded, data.gem_e4_verdict_probs, with_prob=True)
    e3 = manifest_to_arrays(data.e3_rows, data.gem_e3_verdict_probs, with_prob=True)

    # 5-fold OOF calibrated specialist probability & decision on E4 (PR-062)
    oof_probs = fit_oof_isotonic_by_corpus(e4["y"], e4["pu"], e4["ds"], n_splits=5, seed=42)
    e4["pc"] = oof_probs
    e4["s"] = (oof_probs >= 0.5).astype(int)

    for dataset in (e4, e3):
        dataset["h"] = binary_entropy_bits(dataset["pc"])

    f1_spec = float(compute_f1(e3["y"], e3["s"]))
    rows_out: dict[str, Any] = {}
    all_e3 = np.ones(len(e3["y"]), dtype=bool)
    all_e4 = np.ones(len(e4["y"]), dtype=bool)

    c_full, yp_conf = _tune_conf_gate(e4, e3, all_e4, all_e3)
    rows_out["full_conf_gate"] = {"c_L": c_full} | _summarize_policy(e3, yp_conf, all_e3)
    rows_out["full_replace"] = _summarize_policy(e3, e3["l"], all_e3)
    a_full, yp_fus = _tune_soft_fusion(e4, e3, all_e4, all_e3)
    rows_out["full_soft_fusion"] = {"alpha": a_full} | _summarize_policy(e3, yp_fus, all_e3)

    # Per-corpus temperature scaling fit on E4 uncalibrated logits
    p_temp_e3 = np.zeros(len(e3["y"]), dtype=float)
    temps: dict[str, float] = {}
    for corpus in sorted(set(e4["ds"])):
        m4_c = e4["ds"] == corpus
        m3_c = e3["ds"] == corpus
        y_c = e4["y"][m4_c]
        z_c = logit(np.clip(e4["pu"][m4_c], 1e-6, 1.0 - 1e-6))

        def _nll(temp: float, yy: np.ndarray = y_c, zz: np.ndarray = z_c) -> float:
            pr = expit(zz / temp)
            return float(
                -np.mean(
                    yy * np.log(np.clip(pr, 1e-12, 1.0))
                    + (1 - yy) * np.log(np.clip(1.0 - pr, 1e-12, 1.0))
                )
            )

        temp_opt = float(minimize_scalar(_nll, bounds=(0.05, 20.0), method="bounded").x)
        temps[corpus] = temp_opt
        p_temp_e3[m3_c] = expit(logit(np.clip(e3["pu"][m3_c], 1e-6, 1.0 - 1e-6)) / temp_opt)

    # Learned deferral gates P(B|x) > P(C|x) trained on E4 OOF features
    is_b = ((e4["s"] != e4["y"]) & (e4["l"] == e4["y"])).astype(int)
    is_c = ((e4["s"] == e4["y"]) & (e4["l"] != e4["y"])).astype(int)
    x4, x3 = _gate_features(e4), _gate_features(e3)

    lr_diff = (
        LogisticRegression(max_iter=1000).fit(x4, is_b).predict_proba(x3)[:, 1]
        - LogisticRegression(max_iter=1000).fit(x4, is_c).predict_proba(x3)[:, 1]
    )
    hgb_diff = (
        HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=42)
        .fit(x4, is_b)
        .predict_proba(x3)[:, 1]
        - HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=42)
        .fit(x4, is_c)
        .predict_proba(x3)[:, 1]
    )
    rows_out["learned_logreg_gate"] = _summarize_policy(
        e3, np.where(lr_diff > 0, e3["l"], e3["s"]), lr_diff > 0
    )
    rows_out["learned_hgb_gate"] = _summarize_policy(
        e3, np.where(hgb_diff > 0, e3["l"], e3["s"]), hgb_diff > 0
    )

    h_temp_e3 = binary_entropy_bits(p_temp_e3)
    h_uncal_e3 = binary_entropy_bits(e3["pu"])
    for key, tau in taus.items():
        m3 = e3["h"] > tau
        m4 = e4["h"] > tau
        n_esc = int(m3.sum())
        rows_out[f"entropy_tau_{key}"] = _summarize_policy(
            e3, np.where(m3, e3["l"], e3["s"]), m3
        )
        m_temp = _matched_topk_mask(h_temp_e3, n_esc)
        rows_out[f"temp_scaling_entropy_matched_{key}"] = _summarize_policy(
            e3, np.where(m_temp, e3["l"], e3["s"]), m_temp
        )
        m_uncal = _matched_topk_mask(h_uncal_e3, n_esc)
        rows_out[f"uncalibrated_entropy_matched_{key}"] = _summarize_policy(
            e3, np.where(m_uncal, e3["l"], e3["s"]), m_uncal
        )
        a_k, yp_fus_k = _tune_soft_fusion(e4, e3, m4, m3)
        rows_out[f"tau_soft_fusion_{key}"] = {"alpha": a_k} | _summarize_policy(e3, yp_fus_k, m3)
        c_k, yp_conf_k = _tune_conf_gate(e4, e3, m4, m3)
        rows_out[f"tau_conf_gate_{key}"] = {"c_L": c_k} | _summarize_policy(e3, yp_conf_k, m3)

    return {
        "F1_specialist_E3": f1_spec,
        "taus": taus,
        "E4_rows_used": int(len(e4["y"])),
        "temperature_per_corpus": temps,
        "rows": rows_out,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify against committed frozen JSON without writing")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output JSON path")
    args = parser.parse_args(argv)

    result = run()
    if args.verify:
        diffs = verify_json_file(result, DEFAULT_OUT)
        if diffs:
            print("FAIL eval_v2_policies verification:", diffs[:10], file=sys.stderr)
            return 1
        print(f"PASS eval_v2_policies matches {DEFAULT_OUT.relative_to(REPO_ROOT)}")
        return 0

    write_json(args.output, result, indent=2)
    temps = result["temperature_per_corpus"]
    print(
        "F1 specialist",
        round(result["F1_specialist_E3"], 4),
        "| T per corpus",
        {k: round(v, 3) for k, v in temps.items()},
    )
    for key, val in result["rows"].items():
        extra = " ".join(f"{a}={val[a]}" for a in ("c_L", "alpha") if a in val)
        print(
            f"{key:42s} q={val['queried_pct']:6.2f}% B={val['B']:5d} C={val['C']:5d} "
            f"net={val['net']:+5d} F1={val['F1']:.4f} {extra}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
