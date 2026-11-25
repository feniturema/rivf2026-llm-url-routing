#!/usr/bin/env python3
"""
Small extras for the camera-ready manuscript, all from frozen artifacts:
  * E4 internal-check row (August E4, P-49 excluded, in-sample calibrated scores)
  * E3 confusion matrices (specialist, full escalation, b=0.25, tau*_BE)
  * Freeze and first-request UTC timestamps
  * List-price cost per URL from measured v2 tokens and spend summary

Default output: results/v2_runs/v2_paper_extras.json
"""

from __future__ import annotations

import argparse
import json
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
    read_jsonl,
    verify_json_file,
    write_json,
)
from src.v2.metrics import (
    binary_entropy_bits,
    compute_mcc,
    confusion_counts,
    mcnemar_exact_p,
)

DEFAULT_OUT = RESULTS_V2_DIR / "v2_paper_extras.json"


def run(bundle: V2DataBundle | None = None) -> dict[str, Any]:
    """Computes manuscript auxiliary metrics dictionary."""
    data = bundle if bundle is not None else V2DataBundle.load()
    v2_dir = data.repo_root / "results" / "v2_runs"
    t25 = data.policy_freeze_v2["metadata"]["primary_tau"]
    tbe = data.tau_be_v2["tau_star_BE"]

    # 1. E4 internal consistency check row (in-sample calibrated probabilities)
    arr_e4 = manifest_to_arrays(data.e4_rows_p49_excluded, data.gem_e4_verdicts, with_prob=False)
    y4, s4, l4 = arr_e4["y"], arr_e4["s"], arr_e4["l"]
    h4 = binary_entropy_bits(arr_e4["pc"])
    sc4, lc4 = s4 == y4, l4 == y4
    a_cnt = int((sc4 & lc4).sum())
    b_cnt = int((~sc4 & lc4).sum())
    c_cnt = int((sc4 & ~lc4).sum())
    d_cnt = int((~sc4 & ~lc4).sum())
    e_cnt = b_cnt + d_cnt
    rho = b_cnt / e_cnt
    eps = c_cnt / (a_cnt + c_cnt)
    e4_check: dict[str, Any] = {
        "N": len(y4),
        "spec_acc": (a_cnt + c_cnt) / len(y4),
        "rho": rho,
        "eps": eps,
        "eps_star": rho * e_cnt / (len(y4) - e_cnt),
        "full_net": b_cnt - c_cnt,
        "p": mcnemar_exact_p(b_cnt, c_cnt),
    }
    for key, tau in (("b0.25", t25), ("tauBE", tbe)):
        esc = h4 > tau
        e4_check[key] = {
            "esc_rate": float(esc.mean()),
            "top_net": int((esc & ~sc4 & lc4).sum() - (esc & sc4 & ~lc4).sum()),
            "bottom_net": int((~esc & ~sc4 & lc4).sum() - (~esc & sc4 & ~lc4).sum()),
        }

    # 2. E3 composite confusion matrices
    arr_e3 = manifest_to_arrays(data.e3_rows, data.gem_e3_verdicts, with_prob=False)
    y3, s3, l3 = arr_e3["y"], arr_e3["s"], arr_e3["l"]
    h3 = binary_entropy_bits(arr_e3["pc"])
    cs = confusion_counts(y3, s3)
    cf = confusion_counts(y3, l3)
    e3_conf = {
        "specialist": cs,
        "full_escalation": cf,
        "specialist_acc": (cs["TP"] + cs["TN"]) / len(y3),
        "specialist_f1": 2 * cs["TP"] / (2 * cs["TP"] + cs["FP"] + cs["FN"]),
        "specialist_mcc": compute_mcc(cs),
        "b0.25": confusion_counts(y3, np.where(h3 > t25, l3, s3)),
        "tauBE": confusion_counts(y3, np.where(h3 > tbe, l3, s3)),
    }

    # 3. Pre-registration / execution timestamps
    bridge_ledger = read_jsonl(v2_dir / "BRIDGE" / "gemini-3.5-flash-lite" / "ledger.jsonl")
    timestamps = {
        "policy_freeze_v1_utc": "2026-08-27T16:20:00Z",
        "first_aug_E3_test_inference_utc": "2026-08-27T19:24:03Z",
        "policy_freeze_v2_utc": data.policy_freeze_v2["metadata"]["timestamp_utc"],
        "tau_BE_v2_utc": data.tau_be_v2["timestamp_utc"],
        "first_v2_gemini_test_request_utc": min(r["timestamp"] for r in bridge_ledger),
    }

    # 4. List-price cost per URL & total v2 spend
    backfill_summary = read_json(v2_dir / "E3_BACKFILL" / "gemini-3.5-flash-lite" / "summary.json")
    n_req = backfill_summary["unique_requests"]
    tok_in = backfill_summary["tokens"]["in"] / n_req
    tok_out = backfill_summary["tokens"]["out"] / n_req
    per_call_usd = (tok_in * 0.30 + tok_out * 2.50) / 1e6
    cost_list = {
        "gemini_in_tokens_per_call": tok_in,
        "gemini_out_tokens_per_call": tok_out,
        "usd_per_call": per_call_usd,
        "usd_per_million_urls_full": per_call_usd * 1e6,
        "usd_per_million_urls_b0.25": per_call_usd * 1e6 * float((h3 > t25).mean()),
        "usd_per_million_urls_tauBE": per_call_usd * 1e6 * float((h3 > tbe).mean()),
    }

    spend: dict[str, Any] = {}
    for subdir in sorted(v2_dir.glob("*/*")):
        summary_file = subdir / "summary.json"
        if summary_file.exists():
            spend[f"{subdir.parent.name}/{subdir.name}"] = read_json(summary_file)[
                "estimated_cost_usd_list_price"
            ]
    spend["total"] = round(sum(spend.values()), 2)

    return {
        "E4_check": e4_check,
        "E3_confusion": e3_conf,
        "timestamps": timestamps,
        "cost_list_price": cost_list,
        "v2_list_price_spend_usd": spend,
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
            print("FAIL eval_v2_paper_extras verification:", diffs[:10], file=sys.stderr)
            return 1
        print(f"PASS eval_v2_paper_extras matches {DEFAULT_OUT.relative_to(REPO_ROOT)}")
        return 0

    write_json(args.output, result, indent=2)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
