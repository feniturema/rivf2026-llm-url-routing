#!/usr/bin/env python3
"""
[ARCHIVED v1 SCRIPT — SUPERSEDED BY scripts/freeze_policy_thresholds_v2.py]
Retained for provenance of results/policies/policy_freeze.json. For camera-ready
reproduction, use `experiments/reproduce_v2_all.py` and `scripts/freeze_policy_thresholds_v2.py`.
"""

import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.routing.policies import compute_routing_scores, PolicyType

MANIFEST_DIR = Path("data/manifests")
OUT_DIR = Path("results/policies")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BUDGET_GRID = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
POLICIES: list[PolicyType] = [
    "R0_random",
    "R1_max_prob",
    "R2_margin",
    "R3_entropy",
    "R4_calibrated_conf",
    "R5_conformal",
]


def freeze_policies():
    print("================================================================================")
    print("  Pre-Registering and Freezing Routing Policy Thresholds on DEV_POLICY")
    print("================================================================================")

    # Aggregate DEV_POLICY from primary workload (PWD2021) and champion portfolio
    manifest_files = list(MANIFEST_DIR.glob("*_manifest_seed42.csv"))
    if not manifest_files:
        raise FileNotFoundError(f"No manifest files found in {MANIFEST_DIR}")

    dev_frames = []
    manifest_hashes = {}

    for mf in sorted(manifest_files):
        df = pd.read_csv(mf)
        h = hashlib.sha256(mf.read_bytes()).hexdigest()
        manifest_hashes[mf.name] = h
        dev_split = df[df["split"] == "DEV_POLICY"].copy()
        dev_frames.append(dev_split)
        print(f"Loaded {len(dev_split)} DEV_POLICY rows from {mf.name} (hash: {h[:12]}...)")

    df_dev = pd.concat(dev_frames, ignore_index=True)
    n_dev = len(df_dev)
    print(f"\nTotal Pooled DEV_POLICY Samples: {n_dev}")

    p_uncal = df_dev["specialist_prob_uncal"].values
    p_cal = df_dev["specialist_prob_cal"].values

    frozen_thresholds: dict[str, dict[str, float]] = {}

    for pol in POLICIES:
        scores = compute_routing_scores(pol, uncal_probs=p_uncal, cal_probs=p_cal, seed=42)
        frozen_thresholds[pol] = {}
        for b in BUDGET_GRID:
            if b == 0.0:
                thresh = float("inf")
            elif b == 1.0:
                thresh = float("-inf")
            else:
                thresh = float(np.percentile(scores, (1.0 - b) * 100.0))
            frozen_thresholds[pol][str(b)] = thresh

    # 25% primary policy threshold for R3_entropy
    r3_scores = compute_routing_scores("R3_entropy", uncal_probs=p_uncal, cal_probs=p_cal, seed=42)
    cutoff_25 = float(np.percentile(r3_scores, 75.0))

    freeze_payload = {
        "metadata": {
            "version": "v1.0_frozen",
            "timestamp_utc": "2026-08-28T00:20:00Z",
            "seed": 42,
            "dev_policy_total_samples": n_dev,
            "manifest_hashes": manifest_hashes,
            "primary_policy": "R3_entropy",
            "primary_budget": 0.25,
            "primary_cutoff_25": cutoff_25,
            "budget_grid": BUDGET_GRID,
        },
        "thresholds": frozen_thresholds,
    }

    out_file = OUT_DIR / "policy_freeze.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(freeze_payload, f, indent=2)

    # Compute freeze payload hash
    payload_hash = hashlib.sha256(out_file.read_bytes()).hexdigest()
    print(f"\nSuccessfully froze policy thresholds to: {out_file}")
    print(f"Policy Freeze SHA-256: {payload_hash}")
    print(f"Primary R3_entropy 25% cutoff: {cutoff_25:.6f}")


if __name__ == "__main__":
    freeze_policies()
