#!/usr/bin/env python3
"""
Interim E9 cross-vendor evaluation (v2). Reads frozen caches only; zero API calls.

LLM failures fall back to specialist predictions. Thresholds come from policy_freeze_v2.json
(primary b=0.25) and legacy August tau 0.4363 (continuity row).
Default output: results/v2_runs/E9/e9_v2_interim_metrics.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.v2.io import (
    LEGACY_TAU_04363,
    REPO_ROOT,
    RESULTS_V2_DIR,
    V2DataBundle,
    load_llm_verdict_map,
    manifest_to_arrays,
    sha256_file,
    verify_json_file,
    write_json,
)
from src.v2.metrics import binary_entropy_bits, compute_f1, mcnemar_exact_p

DEFAULT_OUT = RESULTS_V2_DIR / "E9" / "e9_v2_interim_metrics.json"


def _interim_sources(repo_root: Path) -> dict[str, Path]:
    v2_e9 = repo_root / "results" / "v2_runs" / "E9"
    return {
        "deepseek-v4.1-flash (v2)": v2_e9 / "deepseek-flash" / "cache.jsonl",
        "claude-haiku-5-5 (v2)": v2_e9 / "claude-haiku-5-5" / "cache.jsonl",
        "gpt-5.6-luna (v2 re-run)": v2_e9 / "gpt-5.6-luna" / "cache.jsonl",
        "gpt-6-luna (v2, amendment A-1)": v2_e9 / "gpt-6-luna" / "cache.jsonl",
        "gpt-5.6-luna (Aug 2026, as in paper; reference only)": (
            repo_root / "results" / "E9" / "e9_openai_luna_cache.jsonl"
        ),
    }


def _evaluate_vendor(
    rows: list[dict[str, str]],
    verdict_map: dict[str, int],
    label: str,
    taus: dict[str, float],
) -> dict[str, Any]:
    arr = manifest_to_arrays(rows, verdict_map, with_prob=False)
    y, s, l, ds = arr["y"], arr["s"], arr["l"], arr["ds"]
    entropy_bits = binary_entropy_bits(arr["pc"])

    out: dict[str, Any] = {
        "model": label,
        "rows": len(rows),
        "rows_fallback_to_specialist": arr["n_fallback"],
    }
    groups: dict[str, np.ndarray] = {"pooled": np.ones(len(rows), dtype=bool)}
    for corpus in sorted(set(ds)):
        groups[corpus] = ds == corpus

    for grp, gmask in groups.items():
        yg, sg, lg = y[gmask], s[gmask], l[gmask]
        hg = entropy_bits[gmask]
        sc, lc = sg == yg, lg == yg
        a_cnt = int((sc & lc).sum())
        b_cnt = int((~sc & lc).sum())
        c_cnt = int((sc & ~lc).sum())
        d_cnt = int((~sc & ~lc).sum())
        n_grp = a_cnt + b_cnt + c_cnt + d_cnt
        e_cnt = b_cnt + d_cnt
        rho = b_cnt / e_cnt if e_cnt else float("nan")
        eps = c_cnt / (a_cnt + c_cnt)
        eps_star = rho * e_cnt / (n_grp - e_cnt)

        grp_res: dict[str, Any] = {
            "N": n_grp,
            "spec_acc": (a_cnt + c_cnt) / n_grp,
            "A": a_cnt,
            "B": b_cnt,
            "C": c_cnt,
            "D": d_cnt,
            "rho": rho,
            "eps": eps,
            "eps_star": eps_star,
            "full_net": b_cnt - c_cnt,
            "full_mcnemar_p": mcnemar_exact_p(b_cnt, c_cnt),
            "sign_consistent_with_breakeven": bool((b_cnt - c_cnt > 0) == (eps < eps_star)),
            "F1_specialist": float(compute_f1(yg, sg)),
            "F1_full": float(compute_f1(yg, lg)),
        }
        for tname, tau in taus.items():
            esc = hg > tau
            casc = np.where(esc, lg, sg)
            b1 = int((esc & ~sc & lc).sum())
            c1 = int((esc & sc & ~lc).sum())
            b0 = int((~esc & ~sc & lc).sum())
            c0 = int((~esc & sc & ~lc).sum())
            n_esc = int(esc.sum())
            grp_res[tname] = {
                "tau": tau,
                "escalated": n_esc,
                "esc_rate": n_esc / n_grp,
                "top_B": b1,
                "top_C": c1,
                "top_net": b1 - c1,
                "bottom_net": b0 - c0,
                "top_mcnemar_p": mcnemar_exact_p(b1, c1),
                "F1_selective": float(compute_f1(yg, casc)),
            }
        out[grp] = grp_res
    return out


def run(bundle: V2DataBundle | None = None) -> dict[str, Any]:
    """Computes E9 v2 interim metrics dictionary."""
    data = bundle if bundle is not None else V2DataBundle.load()
    man_path = data.repo_root / "data" / "manifests" / "e9_shared_test_cohort_3000.csv"
    taus = {
        "v2_primary_b0.25": data.policy_freeze_v2["metadata"]["primary_tau"],
        "legacy_aug_tau0.4363": LEGACY_TAU_04363,
    }
    report: dict[str, Any] = {
        "manifest": str(man_path.relative_to(data.repo_root)),
        "manifest_sha256": sha256_file(man_path),
        "taus": taus,
        "note": "Interim, engineering-grade. Failures fall back to specialist. No bootstrap CIs yet.",
        "results": [],
    }
    for label, cache_path in _interim_sources(data.repo_root).items():
        vmap = load_llm_verdict_map(cache_path, with_prob=False)
        report["results"].append(_evaluate_vendor(data.e9_rows, vmap, label, taus))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify against committed frozen JSON without writing")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output JSON path")
    args = parser.parse_args(argv)

    result = run()
    if args.verify:
        diffs = verify_json_file(result, DEFAULT_OUT)
        if diffs:
            print("FAIL eval_e9_v2_interim verification:", diffs[:10], file=sys.stderr)
            return 1
        print(f"PASS eval_e9_v2_interim matches {DEFAULT_OUT.relative_to(REPO_ROOT)}")
        return 0

    write_json(args.output, result, indent=2)
    print("wrote", args.output)
    for model_res in result["results"]:
        print("\n==", model_res["model"], "| fallback rows:", model_res["rows_fallback_to_specialist"])
        for grp in ("pooled", "PWD2021", "EBBU2017", "URLPhish2025"):
            x = model_res[grp]
            v = x["v2_primary_b0.25"]
            lg = x["legacy_aug_tau0.4363"]
            print(
                f"{grp:14s} {x['N']:5d} {x['spec_acc']:6.2%} {x['rho']:6.2%} "
                f"{x['eps_star']:6.2%} {x['eps']:6.2%} {x['full_net']:+8d} "
                f"{x['full_mcnemar_p']:9.1e} {'Y' if x['sign_consistent_with_breakeven'] else 'N'} | "
                f"{v['esc_rate']:9.2%} {v['top_net']:+5d} {v['bottom_net']:+6d} "
                f"{x['F1_specialist']:.4f}->{v['F1_selective']:.4f} | "
                f"{lg['esc_rate']:8.2%} {lg['top_net']:+5d} {lg['F1_selective']:.4f} | {x['F1_full']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
