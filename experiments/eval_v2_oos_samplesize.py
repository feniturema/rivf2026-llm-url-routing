#!/usr/bin/env python3
"""
Reviewer 2 (major): sample-size curve for out-of-sample breakeven sign prediction.

Estimates (rho, eps) on CAL samples of size n (1,000 draws without replacement, seed 42)
from E4 (August Gemini, P-49 URLs excluded) and predicts the E3 composite TEST sign.
Default output: results/v2_runs/oos_sample_size.json
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
    REPO_ROOT,
    RESULTS_V2_DIR,
    V2DataBundle,
    manifest_to_arrays,
    read_json,
    verify_json_file,
    write_json,
)

DEFAULT_OUT = RESULTS_V2_DIR / "oos_sample_size.json"
SAMPLE_SIZES = (100, 200, 500, 1000, 2000, 5000)
CORPORA = ("pooled", "PWD2021", "EBBU2017", "URLPhish2025")


def run(
    bundle: V2DataBundle | None = None,
    *,
    main_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Computes OOS breakeven sign prediction accuracy and MAE vs sample size n."""
    data = bundle if bundle is not None else V2DataBundle.load()
    m_data = (
        main_metrics
        if main_metrics is not None
        else read_json(data.repo_root / "results" / "v2_runs" / "v2_main_metrics.json")
    )

    out: dict[str, Any] = {}
    for group in CORPORA:
        sub_rows = (
            data.e4_rows_p49_excluded
            if group == "pooled"
            else [r for r in data.e4_rows_p49_excluded if r["dataset"] == group]
        )
        arr = manifest_to_arrays(sub_rows, data.gem_e4_verdicts, with_prob=False)
        sc = arr["s"] == arr["y"]
        lc = arr["l"] == arr["y"]

        test_stats = m_data["E3"][group]
        test_pos = test_stats["full_net"] > 0

        rng = np.random.default_rng(42)
        group_res: dict[str, Any] = {
            "TEST": {
                "rho": test_stats["rho"],
                "eps": test_stats["eps"],
                "eps_star": test_stats["eps_star"],
                "full_net_positive": test_pos,
            }
        }
        n_total = len(sub_rows)
        for n_draw in (*SAMPLE_SIZES, n_total):
            if n_draw > n_total:
                continue
            acc: list[bool] = []
            err_rho: list[float] = []
            err_eps: list[float] = []
            undefined = 0
            n_repeats = 1 if n_draw == n_total else 1000
            for _ in range(n_repeats):
                idx = rng.choice(n_total, n_draw, replace=False)
                sc_i, lc_i = sc[idx], lc[idx]
                a_cnt = (sc_i & lc_i).sum()
                b_cnt = (~sc_i & lc_i).sum()
                c_cnt = (sc_i & ~lc_i).sum()
                d_cnt = (~sc_i & ~lc_i).sum()
                e_cnt = b_cnt + d_cnt
                if e_cnt == 0 or (a_cnt + c_cnt) == 0:
                    undefined += 1
                    continue
                rho = b_cnt / e_cnt
                eps = c_cnt / (a_cnt + c_cnt)
                eps_star = rho * e_cnt / (n_draw - e_cnt)
                acc.append(bool((eps < eps_star) == test_pos))
                err_rho.append(abs(rho - test_stats["rho"]))
                err_eps.append(abs(eps - test_stats["eps"]))
            key = "full" if n_draw == n_total else str(n_draw)
            group_res[key] = {
                "n": n_draw,
                "draws_used": len(acc),
                "draws_undefined": undefined,
                "sign_accuracy": float(np.mean(acc)),
                "MAE_rho": float(np.mean(err_rho)),
                "MAE_eps": float(np.mean(err_eps)),
            }
        out[group] = group_res
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify against committed frozen JSON without writing")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output JSON path")
    args = parser.parse_args(argv)

    result = run()
    if args.verify:
        diffs = verify_json_file(result, DEFAULT_OUT)
        if diffs:
            print("FAIL eval_v2_oos_samplesize verification:", diffs[:10], file=sys.stderr)
            return 1
        print(f"PASS eval_v2_oos_samplesize matches {DEFAULT_OUT.relative_to(REPO_ROOT)}")
        return 0

    write_json(args.output, result, indent=2)
    for grp, res in result.items():
        t_info = res["TEST"]
        print(
            f"{grp} TEST eps={t_info['eps']:.4f} eps*={t_info['eps_star']:.4f} "
            f"pos={t_info['full_net_positive']}"
        )
        for key, val in res.items():
            if key != "TEST":
                print(
                    f"   n={val['n']:5d}  sign_acc={val['sign_accuracy']:.3f}  "
                    f"MAE_rho={val['MAE_rho']:.4f}  MAE_eps={val['MAE_eps']:.4f}  "
                    f"undefined={val['draws_undefined']}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
