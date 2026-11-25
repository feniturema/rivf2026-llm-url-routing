#!/usr/bin/env python3
"""
tau*_BE (secondary operating point) per policy_freeze_v2.json + amendment A-2. Uses E4 only.

E4 LLM source: August Gemini cache (A-2); LLM failure -> specialist decision.
Excludes P-49 CAL URLs that also occur in TEST.
Scores use 5-fold OOF isotonic probabilities per corpus; net gain uses frozen
specialist_pred (PR-062/PR-063 documented asymmetry).
Refuses to overwrite existing results/policies/tau_BE_v2.json unless --output or --verify.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.v2.io import (
    POLICIES_DIR,
    REPO_ROOT,
    V2DataBundle,
    manifest_to_arrays,
    sha256_file,
    verify_json_file,
    write_json,
)
from src.v2.metrics import binary_entropy_bits, fit_oof_isotonic_by_corpus

DEFAULT_OUT = POLICIES_DIR / "tau_BE_v2.json"


def _tau_star(entropy: np.ndarray, gain: np.ndarray) -> tuple[float, int, int]:
    order = np.argsort(-entropy, kind="stable")
    h_sorted = entropy[order]
    g_sorted = gain[order]
    cum = np.cumsum(g_sorted)
    last_of_block = np.r_[h_sorted[1:] != h_sorted[:-1], True]
    cand_idx = np.where(last_of_block)[0]
    nets = np.r_[0, cum[cand_idx]]
    taus = np.r_[
        np.inf,
        [h_sorted[i + 1] if i + 1 < len(h_sorted) else -np.inf for i in cand_idx],
    ]
    best = nets.max()
    k = int(np.where(nets == best)[0].min())  # tie-break: fewest escalations = largest tau
    return float(taus[k]), int(best), int(np.r_[0, cand_idx + 1][k])


def run(bundle: V2DataBundle | None = None) -> dict[str, Any]:
    """Computes frozen tau_BE_v2 payload from E4 CAL_ROUTER (P-49 excluded)."""
    data = bundle if bundle is not None else V2DataBundle.load()
    fz = data.policy_freeze_v2
    n_all = len(data.e4_rows_all)
    rows = data.e4_rows_p49_excluded

    arr = manifest_to_arrays(rows, data.gem_e4_verdicts, with_prob=False)
    y, s, l, pu, ds = arr["y"], arr["s"], arr["l"], arr["pu"], arr["ds"]
    n_fallback = arr["n_fallback"]

    poof = fit_oof_isotonic_by_corpus(y, pu, ds, n_splits=5, seed=42)
    h_oof = binary_entropy_bits(poof)
    # Note: intentionally uses manifest specialist_pred per frozen tau_BE_v2 rule (PR-062/PR-063)
    gain = np.where((s != y) & (l == y), 1, np.where((s == y) & (l != y), -1, 0))

    tau_be, net_best, n_esc = _tau_star(h_oof, gain)

    rng = np.random.default_rng(42)
    boots = np.empty(2000, dtype=float)
    n_rows = len(rows)
    for b_idx in range(2000):
        sample_idx = rng.integers(0, n_rows, n_rows)
        boots[b_idx] = _tau_star(h_oof[sample_idx], gain[sample_idx])[0]
    fin = boots[np.isfinite(boots)]

    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rule": fz["secondary_tau_BE_rule"],
        "amendment": "A-2 (E4 source = August Gemini cache)",
        "e4_rows_total": n_all,
        "e4_rows_used": n_rows,
        "p49_excluded_rows": n_all - n_rows,
        "llm_fallback_rows": n_fallback,
        "tau_star_BE": tau_be,
        "net_B_minus_C_at_tau_star": net_best,
        "escalated_rows_at_tau_star": n_esc,
        "escalated_rate_E4": n_esc / n_rows,
        "bootstrap_2000_seed42": {
            "p2.5": float(np.percentile(fin, 2.5)),
            "p50": float(np.percentile(fin, 50)),
            "p97.5": float(np.percentile(fin, 97.5)),
            "n_infinite": int((~np.isfinite(boots)).sum()),
        },
        "primary_tau_b0.25_for_reference": fz["metadata"]["primary_tau"],
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
            print("FAIL compute_tau_be_v2 verification:", diffs[:10], file=sys.stderr)
            return 1
        print(f"PASS compute_tau_be_v2 matches {DEFAULT_OUT.relative_to(REPO_ROOT)}")
        return 0

    if args.output.exists():
        print(f"refusing to overwrite {args.output}", file=sys.stderr)
        return 1
    write_json(args.output, result, indent=2)
    print("wrote", args.output, "sha256", sha256_file(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
