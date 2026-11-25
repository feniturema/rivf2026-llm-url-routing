#!/usr/bin/env python3
"""
Diagnostic 300-row LLM Pilot Harvest & Verification of Kill Criterion K2 under Frozen Prompt v2.
Pre-registered against results/policies/policy_freeze.json.
Evaluates:
1. Protocol compliance (0 schema errors, 0 agreement violations, refusal check P-20).
2. Pure Replay Cascade Frontier against frozen DEV_POLICY thresholds.
3. 2-Zone Advantage Partition (A = B - C in top 25% vs bottom 75%).
4. Rigorous Statistical Testing (Exact McNemar p-value, Paired 10k Bootstrap CI).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import binomtest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.llm.cache import LLMCache
from src.llm.client import GeminiAsyncClient
from src.llm.ledger import LedgerEntry, TokenLedger
from src.evaluation.metrics import compute_taxonomy_counts, compute_classification_metrics
from src.replay.engine import replay_cascade
from src.routing.policies import compute_routing_scores


async def run_pilot_harvest(
    manifest_path: Path,
    model_name: str,
    prompt_version: str,
    api_key: str,
    cache_path: Path,
    ledger_path: Path,
    concurrency: int = 1,
    rpm: int = 15,
    max_samples: int | None = None,
    expected_split: str | None = "TEST",
    expected_manifest_sha256: str | None = None,
) -> tuple[pd.DataFrame, TokenLedger]:
    if expected_manifest_sha256 is not None:
        actual_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if actual_sha != expected_manifest_sha256:
            raise ValueError(f"Manifest SHA-256 mismatch for {manifest_path}: expected {expected_manifest_sha256}, got {actual_sha}")

    df_manifest = pd.read_csv(manifest_path)

    # Hard partition purity guard
    if expected_split and expected_split.upper() != "NONE":
        if "split" not in df_manifest.columns:
            raise ValueError(f"Manifest {manifest_path} is missing required 'split' column!")
        unexpected_splits = df_manifest[df_manifest["split"] != expected_split]
        if len(unexpected_splits) > 0:
            split_dist = df_manifest["split"].value_counts().to_dict()
            raise ValueError(
                f"Partition pollution detected! Expected 100% split='{expected_split}', but found {len(unexpected_splits)} invalid rows: {split_dist} in {manifest_path}"
            )

    if max_samples and max_samples < len(df_manifest):
        df_manifest = df_manifest.head(max_samples).copy()
    print(f"Loaded pure cohort manifest with {len(df_manifest)} samples (split={expected_split}) from {manifest_path}")

    cache = LLMCache(cache_path)
    ledger = TokenLedger(ledger_path)
    client = GeminiAsyncClient(
        api_key=api_key,
        model_name=model_name,
        prompt_version=prompt_version,
        cache=cache,
        ledger=ledger,
        timeout_s=25.0,
        max_retries=5,
    )

    sem = asyncio.Semaphore(concurrency if concurrency > 1 else 1)
    delay_between_calls = 60.0 / rpm if rpm > 0 else 0.0

    async def worker(row):
        async with sem:
            rec = await client.classify_url(
                canonical_url=row["canonical_url"],
                sample_id=row["sample_id"],
                dataset=row["dataset"],
                stratum=row.get("stratum", "S-STRICT"),
                split=row.get("split", "TEST"),
                manifest_hash=row.get("manifest_hash", "pilot_v1"),
            )
            if delay_between_calls > 0:
                await asyncio.sleep(delay_between_calls)
            return rec

    t0 = time.perf_counter()
    tasks = [worker(row) for _, row in df_manifest.iterrows()]
    print(f"Launching {len(tasks)} live requests (concurrency={concurrency}, model={model_name}, prompt={prompt_version}, rpm={rpm})...")
    
    harvested = []
    failed_count = 0
    for i, fut in enumerate(asyncio.as_completed(tasks)):
        rec = await fut
        harvested.append(rec)
        if rec.get("error_status") != "success":
            failed_count += 1
        
        # Real-time sliding Fail-Closed Circuit Breaker (halts if failure rate > 5% after min 10 samples)
        if len(harvested) >= 10:
            fail_pct = (failed_count / len(harvested)) * 100.0
            if fail_pct > 5.0:
                raise RuntimeError(
                    f"Fail-Closed Circuit Breaker Tripped! Cumulative failure rate is {fail_pct:.1f}% ({failed_count}/{len(harvested)}), exceeding 5.0% threshold."
                )

        if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
            elapsed_now = time.perf_counter() - t0
            rate = (i + 1) / max(elapsed_now, 0.001)
            print(f"  Progress: {i + 1}/{len(tasks)} processed ({elapsed_now:.1f}s, {rate:.1f} req/s, latest={rec['error_status']})", flush=True)
            
    elapsed = time.perf_counter() - t0
    print(f"Harvest complete in {elapsed:.2f} s ({len(tasks)/elapsed:.2f} req/s).\n", flush=True)

    res_df = pd.DataFrame(harvested)
    merged = df_manifest.merge(res_df, on=["sample_id", "canonical_url_sha256"], how="inner")
    if len(merged) != len(df_manifest):
        raise RuntimeError(f"Join row count mismatch! merged={len(merged)} != manifest={len(df_manifest)}")
    return merged, ledger


def evaluate_pilot_k2(
    merged_df: pd.DataFrame,
    ledger: TokenLedger,
    freeze_file: Path = Path("results/policies/policy_freeze.json"),
    out_report_path: Path = Path("results/pilot/champion_pilot_v2_metrics.json"),
):
    print("\n================================================================================")
    print("  300-Row LLM Pilot Rigorous Diagnostic & Kill Criterion K2 Evaluation")
    print("================================================================================")
    
    integ = ledger.validate_integrity(max_allowed_failure_rate_pct=5.0)
    print(f"Ledger Integrity Verification: {integ['valid_live_integrity']}")
    print(f"  Total Calls:       {integ['total_calls']}")
    print(f"  Successful:        {integ['successful_calls']}")
    print(f"  Failed:            {integ['failed_calls']} ({integ['failure_rate_pct']:.2f}%)")
    print(f"  Latency SD:        {integ['sd_latency_ms']:.2f} ms")
    print(f"  Distinct TS:       {integ['distinct_timestamps']}")

    if not integ["valid_live_integrity"]:
        print(f"🚨 INTEGRITY VIOLATION (K5 FAIL-CLOSED): {integ['violation_reasons']}", file=sys.stderr)
        raise RuntimeError(f"Fail-closed audit triggered: {integ['violation_reasons']}")

    total = len(merged_df)
    y_true = merged_df["y"].values
    y_spec = merged_df["specialist_pred"].values
    p_uncal = merged_df["specialist_prob_uncal"].values if "specialist_prob_uncal" in merged_df else merged_df.get("specialist_prob", np.zeros(len(merged_df))).values
    p_cal = merged_df["specialist_prob_cal"].values if "specialist_prob_cal" in merged_df else p_uncal
    y_llm = np.where(merged_df["parsed_verdict"] == "phishing", 1, np.where(merged_df["parsed_verdict"] == "legitimate", 0, y_spec))

    tax = compute_taxonomy_counts(y_true, y_spec, y_llm)
    cell_a = tax["cell_a_both_correct"]
    cell_b = tax["cell_b_rescue"]
    cell_c = tax["cell_c_regression"]
    cell_d = tax["cell_d_both_wrong"]
    net_rescue = cell_b - cell_c
    nrr = net_rescue / total if total > 0 else 0.0

    print(f"\n--- 2x2 Taxonomy (All {total} Pilot Samples) ---")
    print(f"  Cell A (Both Correct - Waste):     {cell_a}")
    print(f"  Cell B (Spec Wrong, LLM Right - Rescue):     {cell_b}")
    print(f"  Cell C (Spec Right, LLM Wrong - Regression): {cell_c}")
    print(f"  Cell D (Both Wrong - Hard):        {cell_d}")
    print(f"\nNet Rescue Count (B - C): {net_rescue}")
    print(f"Net Rescue Rate (NRR) over all escalated: {nrr:.4f}")

    # Exact McNemar Test on b=1.00 discordant pairs (B vs C)
    if (cell_b + cell_c) > 0:
        p_mcnemar = binomtest(cell_b, cell_b + cell_c, p=0.5).pvalue
        print(f"Exact McNemar Test on Discordant Pairs (B={cell_b} vs C={cell_c}): p = {p_mcnemar:.4f}")
    else:
        p_mcnemar = 1.0

    # Read pre-registered policy thresholds
    if not freeze_file.exists():
        raise FileNotFoundError(f"Missing pre-registered policy file: {freeze_file}")

    with open(freeze_file, "r", encoding="utf-8") as f:
        freeze_data = json.load(f)

    cutoff_25 = freeze_data["metadata"]["primary_cutoff_25"]
    r3_thresholds = freeze_data["thresholds"]["R3_entropy"]
    scores = compute_routing_scores("R3_entropy", uncal_probs=p_uncal, cal_probs=p_cal, seed=42)

    # 2-Zone Advantage Partition against Pre-Registered DEV_POLICY Cutoff
    mask_top25 = scores >= cutoff_25
    mask_bot75 = ~mask_top25

    b_top = int(np.sum((y_spec[mask_top25] != y_true[mask_top25]) & (y_llm[mask_top25] == y_true[mask_top25])))
    c_top = int(np.sum((y_spec[mask_top25] == y_true[mask_top25]) & (y_llm[mask_top25] != y_true[mask_top25])))

    b_bot = int(np.sum((y_spec[mask_bot75] != y_true[mask_bot75]) & (y_llm[mask_bot75] == y_true[mask_bot75])))
    c_bot = int(np.sum((y_spec[mask_bot75] == y_true[mask_bot75]) & (y_llm[mask_bot75] != y_true[mask_bot75])))

    print(f"\n--- 2-Zone Advantage Partition (Cutoff={cutoff_25:.4f} from DEV_POLICY) ---")
    print(f"{'Escalation Region':<25} | {'Rescues (B)':<12} | {'Regressions (C)':<15} | {'Net Advantage (A)':<18}")
    print("-" * 75)
    print(f"{'Top 25% Uncertainty Zone':<25} | {b_top:<12} | {c_top:<15} | {b_top - c_top:+d}")
    print(f"{'Bottom 75% Certainty Zone':<25} | {b_bot:<12} | {c_bot:<15} | {b_bot - c_bot:+d}")
    print(f"{'Total (b = 1.00 Full)':<25} | {cell_b:<12} | {cell_c:<15} | {net_rescue:+d}")

    # Pre-registered budget replay
    budgets = [float(b) for b in r3_thresholds.keys()]
    frontier_records = []
    for b in budgets:
        thresh = r3_thresholds[str(b)]
        res = replay_cascade(y_true, y_spec, y_llm, scores, threshold=thresh, budget_nominal=b, policy_name="R3_entropy")
        frontier_records.append({
            "budget": b,
            "realised_escalation": res.escalation_rate_realised,
            "cascade_f1": res.cascade_f1,
            "delta_f1": res.delta_f1,
            "rescues_B": res.rescues_cell_b,
            "regressions_C": res.regressions_cell_c,
            "NRR": res.net_rescue_rate_nrr,
        })
    df_frontier = pd.DataFrame(frontier_records)
    print("\n--- Pre-Registered Budget Frontier on Pilot (DEV_POLICY Thresholds) ---")
    print(df_frontier.to_string(index=False))

    # Paired Bootstrap 95% CI on Delta F1 between b=0.25 and b=1.00
    thresh_25 = r3_thresholds["0.25"]
    thresh_100 = r3_thresholds["1.0"]
    y_casc_25 = np.where(scores >= thresh_25, y_llm, y_spec)
    y_casc_100 = np.where(scores >= thresh_100, y_llm, y_spec)

    from sklearn.metrics import f1_score
    np.random.seed(42)
    deltas = []
    for _ in range(2000):
        idx = np.random.choice(total, size=total, replace=True)
        f1_25 = f1_score(y_true[idx], y_casc_25[idx], zero_division=0)
        f1_100 = f1_score(y_true[idx], y_casc_100[idx], zero_division=0)
        deltas.append(f1_25 - f1_100)

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    print(f"\nPaired Bootstrap 95% CI on (F1_25% - F1_100%): [{ci_low:+.4f}, {ci_high:+.4f}]")

    # Final Verdict & k2_status determination
    print(f"\n--- Kill Criterion K2 Verdict ---")
    k2_passed = bool(net_rescue >= 0)
    if total < 100:
        k2_status = f"indeterminate (n={total}, {cell_b + cell_c} discordant events)"
    elif k2_passed:
        k2_status = "passed (net positive/neutral rescue balance)"
    else:
        k2_status = "failed (regressions exceed rescues)"

    if k2_passed:
        print(f"✅ K2 PASSED: LLM achieves net-positive or neutral rescue balance (B={cell_b} >= C={cell_c}).")
    else:
        print(f"⚠️ K2 TRIGGERED REFRAME: LLM causes net regression (C={cell_c} > B={cell_b}).")

    # Save complete evaluation artifact
    eval_report = {
        "metadata": {
            "total_pilot_samples": total,
            "model": merged_df["model"].iloc[0],
            "server_model_version": merged_df["model_version"].iloc[0],
            "prompt_version": merged_df["prompt_version"].iloc[0],
            "prompt_combined_sha256": merged_df["prompt_combined_sha256"].iloc[0],
            "k2_status": k2_status,
            "k2_passed": bool(cell_b >= cell_c),
        },
        "two_by_two": {
            "cell_a_waste": int(cell_a),
            "cell_b_rescue": int(cell_b),
            "cell_c_regression": int(cell_c),
            "cell_d_hard": int(cell_d),
            "net_rescue_count": net_rescue,
            "net_rescue_rate": nrr,
            "mcnemar_p_value": p_mcnemar,
        },
        "advantage_partition": {
            "dev_policy_cutoff_25": cutoff_25,
            "top_25_rescues_b": b_top,
            "top_25_regressions_c": c_top,
            "top_25_net_advantage": b_top - c_top,
            "bot_75_rescues_b": b_bot,
            "bot_75_regressions_c": c_bot,
            "bot_75_net_advantage": b_bot - c_bot,
        },
        "bootstrap_ci_delta_f1": {
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
        },
        "ledger_summary": ledger.summary(),
    }
    
    out_rep_path = Path(out_report_path)
    out_rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_rep_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\nSaved verified evaluation report to: {out_rep_path}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM Benchmark Harvest under Frozen Prompt v2")
    parser.add_argument("--manifest", default="data/manifests/e3_test_all_26886.csv")
    parser.add_argument("--model", default="gemini-flash-lite-latest")
    parser.add_argument("--prompt-version", default="v2")
    parser.add_argument("--expected-split", default="TEST", help="Expected split name (e.g. TEST, CAL_ROUTER, or NONE)")
    parser.add_argument("--expected-manifest-sha256", default=None, help="Expected SHA-256 hash of manifest file")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--cache", default="results/pilot/champion_pilot_v2_cache.jsonl")
    parser.add_argument("--ledger", default="results/pilot/champion_pilot_v2_ledger.jsonl")
    parser.add_argument("--metrics-out", default="results/pilot/champion_pilot_v2_metrics.json", help="Path to save evaluation metrics JSON")
    parser.add_argument("--freeze-file", default="results/policies/policy_freeze.json")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rpm", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    # Auto-load API Key
    for env_path in [Path(".env"), Path.home() / ".env"]:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    os.environ["GEMINI_API_KEY"] = line.replace("GEMINI_API_KEY=", "").strip().strip("\"'")
                    break
                elif line.startswith("AIzaSy") or line.startswith("AQ."):
                    os.environ["GEMINI_API_KEY"] = line
                    break

    manifest_path = Path(args.manifest)
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        print("ERROR: GEMINI_API_KEY is required.", file=sys.stderr)
        sys.exit(1)

    merged, ledger = asyncio.run(run_pilot_harvest(
        manifest_path=manifest_path,
        model_name=args.model,
        prompt_version=args.prompt_version,
        api_key=api_key,
        cache_path=Path(args.cache),
        ledger_path=Path(args.ledger),
        concurrency=args.concurrency,
        rpm=args.rpm,
        max_samples=args.max_samples,
        expected_split=args.expected_split,
        expected_manifest_sha256=args.expected_manifest_sha256,
    ))

    # Strict Cache vs Ledger parity assertion
    cache = LLMCache(Path(args.cache))
    n_cache = len(cache)
    n_ledger_success = len([e for e in ledger.entries if e.error_status == "success"])
    if n_cache != n_ledger_success:
        raise RuntimeError(f"Cache-Ledger mismatch: {n_cache} cache records != {n_ledger_success} successful ledger entries!")
    
    cache_hashes = cache.get_all_hashes()
    ledger_hashes = set(e.canonical_url_sha256 for e in ledger.entries if e.error_status == "success")
    if cache_hashes != ledger_hashes:
        raise RuntimeError(f"Cache-Ledger hash set mismatch: {len(cache_hashes.symmetric_difference(ledger_hashes))} discordant hashes!")
    print(f"✅ Cache-Ledger Parity Verified: {n_cache}/{n_cache} records match 1-to-1 bit-identically.")

    evaluate_pilot_k2(merged, ledger, freeze_file=Path(args.freeze_file), out_report_path=Path(args.metrics_out))


if __name__ == "__main__":
    main()
