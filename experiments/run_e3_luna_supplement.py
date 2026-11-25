#!/usr/bin/env python3
"""
[ARCHIVED v1 HARVESTER — SUPERSEDED BY experiments/run_harvest_v2.py]
August 2026 supplemental harvester retained for provenance. Camera-ready v2 uses
pure single-model Gemini backfill (`experiments/run_harvest_v2.py --vendor gemini --phase e3_backfill`).
"""

import asyncio
import datetime
import json
import os
import sys
import time
from pathlib import Path
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.llm.openai_client import OpenAIAsyncClient
from src.llm.cache import LLMCache
from src.llm.ledger import TokenLedger, LedgerEntry
from experiments.run_pilot import evaluate_pilot_k2


def load_env_keys():
    for p in [Path(".env"), Path.home() / ".env"]:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("\"'")
                elif line.startswith("sk-"):
                    os.environ["OPENAI_API_KEY"] = line


async def main():
    load_env_keys()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in environment or .env!")

    manifest_path = Path("data/manifests/e3_test_all_26886.csv")
    cache_path = Path("results/E3/e3_gemini_test_cache.jsonl")
    ledger_path = Path("results/E3/e3_gemini_test_ledger.jsonl")
    metrics_path = Path("results/E3/e3_gemini_test_metrics.json")

    df_manifest = pd.read_csv(manifest_path)
    cache = LLMCache(cache_path)
    ledger = TokenLedger(ledger_path)

    # Find missing rows from Gemini cache
    missing_rows = []
    for _, row in df_manifest.iterrows():
        url_sha = row["canonical_url_sha256"]
        c_gemini = cache.get(url_sha, "gemini-flash-lite-latest", "v2")
        c_luna = cache.get(url_sha, "gpt-5.6-luna", "v2")
        if not c_gemini and not c_luna:
            missing_rows.append(row)

    df_missing = pd.DataFrame(missing_rows)
    print(f"Total E3 manifest rows: {len(df_manifest)}")
    print(f"Already cached (Gemini or Luna): {len(df_manifest) - len(df_missing)}")
    print(f"Remaining to harvest with GPT-5.6 Luna: {len(df_missing)} ({df_missing['canonical_url_sha256'].nunique() if len(df_missing)>0 else 0} unique URLs)", flush=True)

    if len(df_missing) > 0:
        client = OpenAIAsyncClient(
            api_key=api_key,
            model_name="gpt-5.6-luna",
            prompt_version="v2",
            base_url=base_url,
        )

        sem = asyncio.Semaphore(4)

        async def worker(row):
            url_sha = row["canonical_url_sha256"]
            cached_res = cache.get(url_sha, "gpt-5.6-luna", "v2") or cache.get(url_sha, "gemini-flash-lite-latest", "v2")
            if cached_res:
                return {
                    "sample_id": row["sample_id"],
                    "canonical_url_sha256": url_sha,
                    "parsed_verdict": cached_res.get("parsed_verdict"),
                    "parsed_probability": cached_res.get("parsed_probability"),
                    "error_status": cached_res.get("error_status", "success"),
                }

            async with sem:
                res = await client.classify_url(
                    canonical_url=row["canonical_url"],
                    sample_id=row["sample_id"],
                    dataset=row.get("dataset", "UNKNOWN"),
                    stratum=row.get("stratum", "UNKNOWN"),
                    split=row.get("split", "TEST"),
                )
                cache.put(res)
                entry = LedgerEntry(
                    sample_id=row["sample_id"],
                    canonical_url_sha256=url_sha,
                    model="gpt-5.6-luna",
                    prompt_version="v2",
                    input_tokens=res.get("input_tokens", 0),
                    output_tokens=res.get("output_tokens", 0),
                    latency_ms=res.get("latency_ms", 0.0),
                    retry_count=res.get("retry_count", 0),
                    error_status=res.get("error_status", "unknown"),
                    timestamp=res.get("timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
                    api_account="openai_supplementary",
                )
                ledger.record(entry)
                return {
                    "sample_id": row["sample_id"],
                    "canonical_url_sha256": url_sha,
                    "parsed_verdict": res.get("parsed_verdict"),
                    "parsed_probability": res.get("parsed_probability"),
                    "error_status": res.get("error_status", "unknown"),
                }

        tasks = [worker(row) for _, row in df_missing.iterrows()]
        print(f"Launching {len(tasks)} OpenAI requests (concurrency=4, model=gpt-5.6-luna)...", flush=True)

        t0 = time.perf_counter()
        harvested = []
        for i, fut in enumerate(asyncio.as_completed(tasks)):
            rec = await fut
            harvested.append(rec)
            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                elapsed_now = time.perf_counter() - t0
                rate = (i + 1) / max(elapsed_now, 0.001)
                print(f"  Progress: {i + 1}/{len(tasks)} harvested ({elapsed_now:.1f}s, {rate:.1f} req/s, latest={rec['error_status']})", flush=True)

        print(f"Supplemental harvest finished in {time.perf_counter() - t0:.2f} s.\n", flush=True)

    # Assemble full 26,886 dataset from unified cache
    print("Building full 26,886 test cohort from unified cache...", flush=True)
    all_results = []
    for _, row in df_manifest.iterrows():
        url_sha = row["canonical_url_sha256"]
        cached_res = cache.get(url_sha, "gemini-flash-lite-latest", "v2") or cache.get(url_sha, "gpt-5.6-luna", "v2")
        if not cached_res:
            raise RuntimeError(f"Missing cache entry for url_sha {url_sha}!")
        all_results.append({
            "sample_id": row["sample_id"],
            "canonical_url_sha256": url_sha,
            "model": cached_res.get("model", "gemini-flash-lite-latest"),
            "model_version": cached_res.get("model_version", "gemini-3.5-flash-lite"),
            "prompt_version": cached_res.get("prompt_version", "v2"),
            "prompt_combined_sha256": cached_res.get("prompt_combined_sha256", "e67d6cb3b1b546ce3cd2441f829f611be878553d632d9dd27bf372ba8797d8d3"),
            "parsed_verdict": cached_res.get("parsed_verdict"),
            "parsed_probability": cached_res.get("parsed_probability"),
            "error_status": cached_res.get("error_status", "success"),
        })

    res_df = pd.DataFrame(all_results)
    merged = df_manifest.merge(res_df, on=["sample_id", "canonical_url_sha256"], how="inner")
    if len(merged) != len(df_manifest):
        raise RuntimeError(f"Row count mismatch: merged={len(merged)} != manifest={len(df_manifest)}")

    print(f"✅ Full 100% E3 Cohort Assembly Complete: {len(merged)} rows (100.0% coverage)!", flush=True)
    evaluate_pilot_k2(merged, ledger, out_report_path=metrics_path)
    print(f"✅ Full E3 Metrics exported to {metrics_path}!\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
