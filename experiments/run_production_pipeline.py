#!/usr/bin/env python3
"""
[ARCHIVED v1 ORCHESTRATOR — SUPERSEDED BY experiments/run_*harvest_v2.py]
August 2026 initial harvest orchestrator retained for provenance.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.run_pilot import run_pilot_harvest, evaluate_pilot_k2
from experiments.run_openai_harvest import run_openai_harvest, evaluate_openai_results
from src.llm.cache import LLMCache
from src.llm.ledger import TokenLedger


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
                elif line.startswith("AIzaSy") or line.startswith("AQ."):
                    os.environ["GEMINI_API_KEY"] = line


async def run_e9_production(cfg: dict, max_samples: int | None = None):
    exp = cfg["experiments"]["E9_OPENAI_LUNA_3000"]
    print("\n" + "=" * 80)
    print("  PHASE 1 / 3: E9 CROSS-VENDOR BENCHMARK (GPT-5.6 Luna Max, N=3,000)")
    print("=" * 80)
    
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment or .env!")

    manifest_p = Path(exp["manifest"])
    cache_p = Path(exp["cache"])
    ledger_p = Path(exp["ledger"])
    metrics_p = Path(exp["metrics_out"])

    merged, ledger = await run_openai_harvest(
        manifest_path=manifest_p,
        model_name=exp["model"],
        prompt_version=exp["prompt_version"],
        api_key=api_key,
        base_url=base_url,
        cache_path=cache_p,
        ledger_path=ledger_p,
        concurrency=exp.get("concurrency", 4),
        max_samples=max_samples,
        expected_split=exp["expected_split"],
        expected_manifest_sha256=exp["manifest_sha256"],
    )

    evaluate_openai_results(merged, ledger, out_report_path=metrics_p)
    print(f"✅ Phase 1 (E9) Complete! Output saved to {metrics_p}")
    return merged, ledger


async def run_e4_production(cfg: dict, max_samples: int | None = None):
    exp = cfg["experiments"]["E4_CAL_ROUTER"]
    print("\n" + "=" * 80)
    print("  PHASE 2 / 3: E4 GATE CALIBRATION HARVEST (Gemini Flash-Lite, N=13,446)")
    print("=" * 80)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment or .env!")

    manifest_p = Path(exp["manifest"])
    cache_p = Path(exp["cache"])
    ledger_p = Path(exp["ledger"])
    metrics_p = Path(exp["metrics_out"])

    merged, ledger = await run_pilot_harvest(
        manifest_path=manifest_p,
        model_name=exp["model"],
        prompt_version=exp["prompt_version"],
        api_key=api_key,
        cache_path=cache_p,
        ledger_path=ledger_p,
        concurrency=exp.get("concurrency", 1),
        rpm=exp.get("rpm", 0),
        max_samples=max_samples,
        expected_split=exp["expected_split"],
        expected_manifest_sha256=exp["manifest_sha256"],
    )

    evaluate_pilot_k2(merged, ledger, out_report_path=metrics_p)
    print(f"✅ Phase 2 (E4) Complete! Output saved to {metrics_p}")
    return merged, ledger


async def run_e3_production(cfg: dict, max_samples: int | None = None):
    exp = cfg["experiments"]["E3_FULL_TEST"]
    print("\n" + "=" * 80)
    print("  PHASE 3 / 3: E3 FULL TEST EVALUATION (Gemini Flash-Lite, N=26,886)")
    print("=" * 80)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment or .env!")

    manifest_p = Path(exp["manifest"])
    cache_p = Path(exp["cache"])
    ledger_p = Path(exp["ledger"])
    metrics_p = Path(exp["metrics_out"])

    merged, ledger = await run_pilot_harvest(
        manifest_path=manifest_p,
        model_name=exp["model"],
        prompt_version=exp["prompt_version"],
        api_key=api_key,
        cache_path=cache_p,
        ledger_path=ledger_p,
        concurrency=exp.get("concurrency", 1),
        rpm=exp.get("rpm", 0),
        max_samples=max_samples,
        expected_split=exp["expected_split"],
        expected_manifest_sha256=exp["manifest_sha256"],
    )

    evaluate_pilot_k2(merged, ledger, out_report_path=metrics_p)
    print(f"✅ Phase 3 (E3) Complete! Output saved to {metrics_p}")
    return merged, ledger


async def main():
    parser = argparse.ArgumentParser(description="Run Production Harvest Pipeline")
    parser.add_argument("--config", default="configs/production_experiments_freeze.json")
    parser.add_argument("--target", choices=["all", "e9", "e4", "e3"], default="all", help="Target experiment to run")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max samples for pilot or dry runs")
    args = parser.parse_args()

    load_env_keys()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config freeze missing: {cfg_path}")
    
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print("=" * 80)
    print("  OFFICIAL PRODUCTION HARVEST PIPELINE INITIATED")
    print(f"  Protocol: {cfg.get('protocol', 'RIVF-2026')}")
    print(f"  Target: {args.target.upper()}")
    print("=" * 80)

    t_start = time.perf_counter()

    if args.target in ["all", "e9"]:
        await run_e9_production(cfg, max_samples=args.max_samples)

    if args.target in ["all", "e4"]:
        await run_e4_production(cfg, max_samples=args.max_samples)

    if args.target in ["all", "e3"]:
        await run_e3_production(cfg, max_samples=args.max_samples)

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"  ALL PRODUCTION EXPERIMENTS COMPLETED IN {total_time:.2f} s ({total_time/60:.2f} min)")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
