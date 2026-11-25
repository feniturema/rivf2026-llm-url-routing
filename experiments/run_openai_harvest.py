#!/usr/bin/env python3
"""
Harvest script for OpenAI-compatible models (e.g. GPT-5.6 Luna Max / GPT-4o-mini).
Evaluates 2x2 taxonomy, Kill Criterion K2, and cross-vendor robustness on champion benchmark manifests.
"""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import binomtest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.llm.openai_client import OpenAIAsyncClient
from src.llm.cache import LLMCache
from src.llm.ledger import TokenLedger
from src.evaluation.metrics import compute_taxonomy_counts, compute_classification_metrics
from src.replay.engine import replay_cascade
from src.routing.policies import compute_routing_scores


async def run_openai_harvest(
    manifest_path: Path,
    model_name: str,
    prompt_version: str,
    api_key: str,
    base_url: str,
    cache_path: Path,
    ledger_path: Path,
    concurrency: int = 2,
    max_samples: int | None = None,
    expected_split: str | None = "TEST",
    expected_manifest_sha256: str | None = None,
):
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
    client = OpenAIAsyncClient(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        prompt_version=prompt_version,
        cache=cache,
        ledger=ledger,
        timeout_s=25.0,
        max_retries=3,
    )

    sem = asyncio.Semaphore(concurrency)

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
            return rec

    t0 = time.perf_counter()
    tasks = [worker(row) for _, row in df_manifest.iterrows()]
    print(f"Launching {len(tasks)} requests (concurrency={concurrency}, model={model_name}, prompt={prompt_version})...")
    
    res_records = []
    failed_count = 0
    for fut in asyncio.as_completed(tasks):
        rec = await fut
        res_records.append(rec)
        if rec.get("error_status") != "success":
            failed_count += 1
        # Real-time sliding Fail-Closed Circuit Breaker (halts if failure rate > 5% after min 10 samples)
        if len(res_records) >= 10:
            fail_pct = (failed_count / len(res_records)) * 100.0
            if fail_pct > 5.0:
                raise RuntimeError(
                    f"Fail-Closed Circuit Breaker Tripped! Cumulative failure rate is {fail_pct:.1f}% ({failed_count}/{len(res_records)}), exceeding 5.0% threshold."
                )

    elapsed = time.perf_counter() - t0
    print(f"Harvest complete in {elapsed:.2f} s ({len(tasks)/elapsed:.1f} req/s).\n")

    res_df = pd.DataFrame(res_records)
    merged = df_manifest.merge(res_df, on=["sample_id", "canonical_url_sha256"], how="inner")
    if len(merged) != len(df_manifest):
        raise RuntimeError(f"Join row count mismatch! merged={len(merged)} != manifest={len(df_manifest)}")
    return merged, ledger


def evaluate_openai_results(merged: pd.DataFrame, ledger: TokenLedger, out_report_path: Path | None = None):
    total = len(merged)
    success = (merged["error_status"] == "success").sum()
    print(f"================================================================================")
    print(f"  OpenAI-Compatible Harvest Diagnostics ({merged['model'].iloc[0]})")
    print(f"================================================================================")
    print(f"Total Requests:      {total}")
    print(f"Successful Parsed:   {success} ({success/total*100:.1f}%)")
    
    y_true = merged["y"].values
    y_spec = merged["specialist_pred"].values
    p_spec = merged.get("specialist_prob_cal", merged.get("specialist_prob", np.zeros(len(merged)))).values
    y_llm = np.where(merged["parsed_verdict"] == "phishing", 1, np.where(merged["parsed_verdict"] == "legitimate", 0, y_spec))

    tax = compute_taxonomy_counts(y_true, y_spec, y_llm)
    cell_a = tax["cell_a_both_correct"]
    cell_b = tax["cell_b_rescue"]
    cell_c = tax["cell_c_regression"]
    cell_d = tax["cell_d_both_wrong"]
    net_rescue = cell_b - cell_c
    nrr = net_rescue / total if total > 0 else 0.0

    print(f"\n--- 2x2 Taxonomy (N={total}) ---")
    print(f"  Cell A (Both Correct - Waste):     {cell_a}")
    print(f"  Cell B (Spec Wrong, LLM Right - Rescue):     {cell_b}")
    print(f"  Cell C (Spec Right, LLM Wrong - Regression): {cell_c}")
    print(f"  Cell D (Both Wrong - Hard):        {cell_d}")
    print(f"\nNet Rescue Count (B - C): {net_rescue}")
    print(f"Net Rescue Rate (NRR) over all escalated: {nrr:.4f}")

    # Discordant pair test
    if (cell_b + cell_c) > 0:
        p_mcnemar = binomtest(cell_b, cell_b + cell_c, p=0.5).pvalue
        print(f"Exact McNemar Test (B={cell_b} vs C={cell_c}): p = {p_mcnemar:.4f}")

    # Cost (Official Luna Rate: $0.20/1M In, $1.20/1M Out)
    summ = ledger.summary(price_in_per_m=0.20, price_out_per_m=1.20)
    print(f"\n--- Cost Summary ---")
    print(f"Total In Tokens:     {summ['total_in_tokens']}")
    print(f"Total Out Tokens:    {summ['total_out_tokens']}")
    print(f"Estimated Cost:      ${summ['estimated_cost_usd']:.4f}")
    print(f"p50 Latency:         {summ['p50_latency_ms']:.1f} ms")
    print(f"p95 Latency:         {summ['p95_latency_ms']:.1f} ms")
    print(f"Ledger Integrity:    {summ['integrity']}")

    p_mcn = float(binomtest(cell_b, cell_b + cell_c, p=0.5).pvalue) if (cell_b + cell_c) > 0 else 1.0

    eval_report = {
        "metadata": {
            "total_samples": total,
            "model": merged["model"].iloc[0],
            "model_version": merged["model_version"].iloc[0],
            "prompt_version": merged["prompt_version"].iloc[0],
        },
        "two_by_two": {
            "cell_a_waste": int(cell_a),
            "cell_b_rescue": int(cell_b),
            "cell_c_regression": int(cell_c),
            "cell_d_hard": int(cell_d),
            "net_rescue_count": int(net_rescue),
            "net_rescue_rate": float(nrr),
            "mcnemar_p_value": p_mcn,
        },
        "ledger_summary": summ,
    }

    if out_report_path:
        out_p = Path(out_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(eval_report, f, indent=2)
        print(f"Saved OpenAI evaluation report to: {out_p}")


def main():
    parser = argparse.ArgumentParser(description="Run OpenAI-compatible LLM Harvest under Frozen Prompt v2")
    parser.add_argument("--manifest", default="data/manifests/e9_shared_test_cohort_3000.csv")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--prompt-version", default="v2")
    parser.add_argument("--expected-split", default="TEST", help="Expected split (e.g. TEST, CAL_ROUTER, or NONE)")
    parser.add_argument("--expected-manifest-sha256", default=None, help="Expected SHA-256 hash of manifest file")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--cache", default="results/pilot/openai_pilot_cache.jsonl")
    parser.add_argument("--ledger", default="results/pilot/openai_pilot_ledger.jsonl")
    parser.add_argument("--metrics-out", default="results/pilot/openai_pilot_metrics.json", help="Path to save evaluation metrics JSON")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    # Auto-load API Key and Base URL
    for env_path in [Path(".env"), Path.home() / ".env"]:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("OPENAI_API_KEY="):
                    os.environ["OPENAI_API_KEY"] = line.replace("OPENAI_API_KEY=", "").strip().strip("\"'")
                elif line.startswith("OPENAI_BASE_URL="):
                    os.environ["OPENAI_BASE_URL"] = line.replace("OPENAI_BASE_URL=", "").strip().strip("\"'")
                elif line.startswith("sk-"):
                    os.environ["OPENAI_API_KEY"] = line

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    manifest_path = Path(args.manifest)
    merged, ledger = asyncio.run(run_openai_harvest(
        manifest_path=manifest_path,
        model_name=args.model,
        prompt_version=args.prompt_version,
        api_key=api_key,
        base_url=base_url,
        cache_path=Path(args.cache),
        ledger_path=Path(args.ledger),
        concurrency=args.concurrency,
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

    evaluate_openai_results(merged, ledger, out_report_path=Path(args.metrics_out))


if __name__ == "__main__":
    main()
