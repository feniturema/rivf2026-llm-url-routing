#!/usr/bin/env python3
"""
policy_freeze_v2 (camera-ready correction of the P-46 scale mismatch).

Pre-registration rule (matching August policy_freeze.json metadata):
  primary_policy = R3_entropy, primary_budget b = 0.25,
  tau(b) = (1-b) quantile of binary entropy (bits) of specialist_prob_cal
  on pooled DEV_POLICY, escalated strictly via `score > tau`.

Refuses to overwrite existing results/policies/policy_freeze_v2.json unless
writing to a different --output path or running in --verify mode.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.v2.io import (
    DATA_MANIFESTS_DIR,
    POLICIES_DIR,
    REPO_ROOT,
    read_manifest_csv,
    sha256_file,
    verify_json_file,
    write_json,
)
from src.v2.metrics import binary_entropy_bits

DEFAULT_OUT = POLICIES_DIR / "policy_freeze_v2.json"
GRID = [
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00,
]
CORPORA = (
    "EBBU2017_manifest_seed42.csv",
    "PWD2021_manifest_seed42.csv",
    "URLPhish2025_manifest_seed42.csv",
)


def run(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Computes frozen policy_freeze_v2 payload from DEV_POLICY manifest splits only."""
    man_dir = repo_root / "data" / "manifests"
    rows: list[dict[str, str]] = []
    hashes: dict[str, str] = {}
    for fname in CORPORA:
        path = man_dir / fname
        hashes[fname] = sha256_file(path)
        rows.extend(read_manifest_csv(path))

    dev = [r for r in rows if r["split"] == "DEV_POLICY"]
    cal = [r for r in rows if r["split"] == "CAL_ROUTER"]
    test_sha = {r["canonical_url_sha256"] for r in rows if r["split"] == "TEST"}
    excl = sorted({r["canonical_url_sha256"] for r in cal} & test_sha)

    scores = binary_entropy_bits([float(r["specialist_prob_cal"]) for r in dev])
    thresholds: dict[str, float] = {}
    realized: dict[str, float] = {}
    for b in GRID:
        if b == 0.0:
            tau = math.inf
        elif b == 1.0:
            tau = -math.inf
        else:
            tau = float(np.percentile(scores, (1.0 - b) * 100.0))
        thresholds[str(b)] = tau
        realized[str(b)] = (
            float((scores > tau).mean())
            if math.isfinite(tau)
            else (0.0 if tau > 0 else 1.0)
        )

    return {
        "metadata": {
            "version": "v2.0_frozen",
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "supersedes": "results/policies/policy_freeze.json (scale mismatch, P-46/P-47)",
            "decision_record": "docs/superpowers/plans/2026-11-22-rivf2026-upgrade-plan-defense-review.md s.3 Q2",
            "seed": 42,
            "dev_policy_rows": len(dev),
            "manifest_sha256": hashes,
            "routing_score": "binary entropy (bits) of specialist_prob_cal, clip 1e-12; escalate iff score > tau",
            "threshold_rule": "tau(b) = numpy.percentile(DEV_POLICY scores, 100*(1-b)), linear interpolation",
            "primary_policy": "R3_entropy",
            "primary_budget": 0.25,
            "primary_tau": thresholds["0.25"],
            "legacy_point_for_continuity": {
                "tau": 0.4362536023939326,
                "note": "August operating point; reported as one frontier row only",
            },
        },
        "R3_entropy_thresholds": thresholds,
        "R3_entropy_dev_realized_rate": realized,
        "secondary_tau_BE_rule": {
            "data": "E4 CAL_ROUTER LLM responses (v2 Gemini run), EXCLUDING the P-49 list below",
            "specialist_probability": (
                "5-fold out-of-fold isotonic calibration within CAL_ROUTER, per corpus, "
                "StratifiedKFold(n_splits=5, shuffle=True, random_state=42), fit on specialist_prob_uncal"
            ),
            "score": "binary entropy (bits) of the OOF calibrated probability",
            "objective": (
                "maximise B(tau) - C(tau) over candidate taus = distinct OOF scores; "
                "escalate iff score > tau; LLM failure => keep specialist decision"
            ),
            "tie_break": "largest tau (fewest escalations)",
            "uncertainty": "2,000 bootstrap resamples of E4 URLs, seed 42 -> 95% percentile CI of tau*_BE",
            "transfer": "apply tau*_BE unchanged to TEST entropy of specialist_prob_cal; never re-tuned on TEST",
            "must_be_computed_and_hashed_before": "any metric is computed on the v2 E3 TEST responses",
        },
        "p49_cal_test_shared_urls": {
            "count": len(excl),
            "sha256_of_sorted_list": hashlib.sha256("\n".join(excl).encode("utf-8")).hexdigest(),
            "list": excl,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify against committed frozen JSON without writing")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output JSON path")
    args = parser.parse_args(argv)

    payload = run()
    if args.verify:
        diffs = verify_json_file(payload, DEFAULT_OUT)
        if diffs:
            print("FAIL freeze_policy_thresholds_v2 verification:", diffs[:10], file=sys.stderr)
            return 1
        print(f"PASS freeze_policy_thresholds_v2 matches {DEFAULT_OUT.relative_to(REPO_ROOT)}")
        return 0

    if args.output.exists():
        print(f"refusing to overwrite {args.output}", file=sys.stderr)
        return 1
    write_json(args.output, payload, indent=2)
    print("wrote", args.output, "sha256", sha256_file(args.output))
    print(
        "primary tau(b=0.25) =",
        payload["R3_entropy_thresholds"]["0.25"],
        "DEV realized",
        payload["R3_entropy_dev_realized_rate"]["0.25"],
    )
    print("P-49 excluded CAL URLs:", payload["p49_cal_test_shared_urls"]["count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
