#!/usr/bin/env python3
"""
v2 supplements for paper tables: routing frontier (Table IV / Fig. 4), error-type composition
(Table VII), URL-Phish 2025 Cell C attribution, and strict single-corpus sensitivity.

E3 = v2 composite (amendment A-2). Escalates iff binary entropy > tau (strict inequality).
Default output: results/v2_runs/v2_frontier_and_error.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.v2.io import (
    REPO_ROOT,
    RESULTS_V2_DIR,
    V2DataBundle,
    manifest_to_arrays,
    read_manifest_csv,
    verify_json_file,
    write_json,
)
from src.v2.metrics import binary_entropy_bits, compute_f1

DEFAULT_OUT = RESULTS_V2_DIR / "v2_frontier_and_error.json"
CORPORA = ("PWD2021", "EBBU2017", "URLPhish2025")


def run(bundle: V2DataBundle | None = None) -> dict[str, Any]:
    """Computes frontier, error composition, URL-Phish attribution, and strict single-corpus metrics."""
    data = bundle if bundle is not None else V2DataBundle.load()
    fz = data.policy_freeze_v2
    t25 = fz["metadata"]["primary_tau"]
    tbe = data.tau_be_v2["tau_star_BE"]

    rows = data.e3_rows
    arr = manifest_to_arrays(rows, data.gem_e3_verdicts, with_prob=False)
    y, s, l, ds = arr["y"], arr["s"], arr["l"], arr["ds"]
    h = binary_entropy_bits(arr["pc"])
    f1_spec = float(compute_f1(y, s))
    sc, lc = s == y, l == y

    def _point(mask: np.ndarray, label: str, nominal: float | None = None) -> dict[str, Any]:
        b_cnt = int((mask & ~sc & lc).sum())
        c_cnt = int((mask & sc & ~lc).sum())
        n_esc = int(mask.sum())
        f1_val = float(compute_f1(y, np.where(mask, l, s)))
        return {
            "label": label,
            "nominal_b": nominal,
            "esc_pct": 100.0 * n_esc / len(y),
            "escalated": n_esc,
            "B": b_cnt,
            "C": c_cnt,
            "net": b_cnt - c_cnt,
            "NRR_esc": (b_cnt - c_cnt) / n_esc if n_esc else None,
            "F1": f1_val,
            "dF1_vs_spec": float(f1_val - f1_spec),
        }

    frontier: list[dict[str, Any]] = []
    for b_str, tau_val in fz["R3_entropy_thresholds"].items():
        frontier.append(_point(h > float(tau_val), f"entropy b={b_str}", float(b_str)))
    frontier.append(_point(h > tbe, "entropy tau*_BE", None))
    u = np.random.default_rng(42).random(len(y))
    frontier.append(_point(u > 0.75, "random 25% (seed 42)", 0.25))

    # Error-type composition under full escalation
    comp: dict[str, Any] = {}
    for grp in (*CORPORA, "pooled"):
        mask = np.ones(len(y), dtype=bool) if grp == "pooled" else (ds == grp)
        fn_s = int((mask & (y == 1) & (s == 0)).sum())
        fp_s = int((mask & (y == 0) & (s == 1)).sum())
        fn_l = int((mask & (y == 1) & (l == 0)).sum())
        fp_l = int((mask & (y == 0) & (l == 1)).sum())
        c_mask = mask & sc & ~lc
        c_fn = int((c_mask & (y == 1)).sum())
        c_total = int(c_mask.sum())
        comp[grp] = {
            "delta_missed_phishing": fn_l - fn_s,
            "delta_false_alarms": fp_l - fp_s,
            "net_check": -((fn_l - fn_s) + (fp_l - fp_s)),
            "C": c_total,
            "C_missed_phishing": c_fn,
            "missed_phish_share_of_C": float(c_fn / max(c_total, 1)),
        }

    # URL-Phish 2025 Cell C attribution
    up_mask = ds == "URLPhish2025"
    parsed_urls = [
        urlparse(r["canonical_url"] if "://" in r["canonical_url"] else "http://" + r["canonical_url"])
        for r in rows
    ]
    schemes = np.array([p.scheme.lower() for p in parsed_urls])
    has_nonempty_path = np.array([len(p.path.strip("/")) > 0 for p in parsed_urls])

    cfn = up_mask & sc & ~lc & (y == 1)
    leg = up_mask & (y == 0)
    ph = up_mask & (y == 1)
    urlphish_attr = {
        "n_cell_c": int((up_mask & sc & ~lc).sum()),
        "n_cell_c_fn": int(cfn.sum()),
        "pct_https_cell_c_fn": float((schemes[cfn] == "https").mean()),
        "pct_https_legit": float((schemes[leg] == "https").mean()),
        "pct_https_phish_all": float((schemes[ph] == "https").mean()),
        "pct_nonempty_path_cell_c_fn": float(has_nonempty_path[cfn].mean()),
        "pct_nonempty_path_legit": float(has_nonempty_path[leg].mean()),
        "pct_cell_c_fn_in_bottom_zone_b0.25": float((h[cfn] <= t25).mean()),
        "pct_cell_c_fn_in_bottom_zone_tauBE": float((h[cfn] <= tbe).mean()),
    }

    # Strict single-corpus sensitivity (URLs occurring in exactly 1 corpus across all splits)
    url_to_corpora: dict[str, set[str]] = {}
    man_dir = data.repo_root / "data" / "manifests"
    for corpus in CORPORA:
        for row in read_manifest_csv(man_dir / f"{corpus}_manifest_seed42.csv"):
            url_to_corpora.setdefault(row["canonical_url_sha256"], set()).add(corpus)

    seen: set[str] = set()
    keep_idx: list[int] = []
    for idx, row in enumerate(rows):
        sha = row["canonical_url_sha256"]
        if len(url_to_corpora.get(sha, ())) == 1 and sha not in seen:
            seen.add(sha)
            keep_idx.append(idx)

    k_arr = np.array(keep_idx, dtype=int)
    yk, sk, lk, hk = y[k_arr], s[k_arr], l[k_arr], h[k_arr]
    sck, lck = sk == yk, lk == yk

    def _sub_metrics(tau: float) -> dict[str, Any]:
        esc = hk > tau
        return {
            "esc_pct": float(100.0 * esc.mean()),
            "top_net": int((esc & ~sck & lck).sum() - (esc & sck & ~lck).sum()),
            "F1": float(compute_f1(yk, np.where(esc, lk, sk))),
        }

    strict_single = {
        "N": int(len(k_arr)),
        "F1_specialist": float(compute_f1(yk, sk)),
        "F1_full": float(compute_f1(yk, lk)),
        "full_net": int((~sck & lck).sum() - (sck & ~lck).sum()),
        "b0.25": _sub_metrics(t25),
        "tauBE": _sub_metrics(tbe),
    }

    return {
        "F1_specialist": f1_spec,
        "frontier": frontier,
        "error_composition_full_escalation": comp,
        "urlphish25_cell_c_attribution": urlphish_attr,
        "strict_single_corpus": strict_single,
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
            print("FAIL eval_v2_supplement verification:", diffs[:10], file=sys.stderr)
            return 1
        print(f"PASS eval_v2_supplement matches {DEFAULT_OUT.relative_to(REPO_ROOT)}")
        return 0

    write_json(args.output, result, indent=2)
    print("F1 spec", round(result["F1_specialist"], 4))
    for q in result["frontier"]:
        nrr = f"{q['NRR_esc']:+.3f}" if q["NRR_esc"] is not None else "  ---"
        print(
            f"{q['label']:22s} esc={q['esc_pct']:6.2f}% B={q['B']:4d} C={q['C']:5d} "
            f"net={q['net']:+5d} NRR={nrr} F1={q['F1']:.4f} dF1={q['dF1_vs_spec']:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
