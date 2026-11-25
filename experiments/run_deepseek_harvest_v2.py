#!/usr/bin/env python3
"""
DeepSeek-V4.1-Flash harvester for the v2 camera-ready re-collection (E9 cross-vendor).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import ssl
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.llm.harvest_common import (  # noqa: E402
    CA_SOURCE,
    OUT_ROOT,
    SSL_CTX,
    Abort,
    BaseHarvester,
    canonical_config_sha256,
    ensure_frozen_config,
    load_api_key,
    load_verified_manifest_rows,
    percentile,
    read_jsonl,
    verify_smoke_gate_passed,
    write_summary_and_manifest,
)
from src.llm.prompt import FROZEN_V2_COMBINED_SHA256, FrozenPromptLoader  # noqa: E402

BASE_URL = "https://api.deepseek.com"
MODEL_SLUG = "deepseek-flash"

FROZEN_CALL_CONFIG: dict[str, Any] = {
    "provider": "deepseek_first_party",
    "endpoint": BASE_URL + "/chat/completions",
    "model": "deepseek-flash",
    "documented_model_version": "DeepSeek-V4.1-Flash-1017",
    "thinking": {"type": "disabled"},
    "temperature": 0.0,
    "top_p_note": "fixed at 1.0 by provider in non-thinking mode",
    "max_tokens": 256,
    "response_format": {"type": "json_object"},
    "messages_layout": "system=prompt_v2.system.txt, user=prompt_v2.user.txt rendered with canonical_url",
    "prompt_version": "v2",
    "prompt_combined_sha256": FROZEN_V2_COMBINED_SHA256,
    "stream": False,
    "repeat_index": 0,
}

PHASES: dict[str, dict[str, Any]] = {
    "smoke": {
        "manifest": REPO / "data/manifests/e4_cal_all_13446.csv",
        "manifest_sha256": "7ecb9929e2939acaea687f16f7e1251d7587125673343e152972b5d35c373195",
        "split": "CAL_ROUTER",
        "rows": 13446,
        "out": OUT_ROOT / "SMOKE" / MODEL_SLUG,
        "sample_per_cell": 5,
    },
    "e9": {
        "manifest": REPO / "data/manifests/e9_shared_test_cohort_3000.csv",
        "manifest_sha256": "f33b8a178e95d98f4b9071f531b327f51d106918d18e903ab8f07871b31810c4",
        "split": "TEST",
        "rows": 3000,
        "out": OUT_ROOT / "E9" / MODEL_SLUG,
        "sample_per_cell": None,
    },
}

PRICES = {
    "offpeak": {"hit": 0.003, "miss": 0.15, "out": 0.60},
    "peak": {"hit": 0.006, "miss": 0.30, "out": 1.20},
}
SMOKE_PASS_MIN_SUCCESS = 0.98
FAIL_RATE_ABORT = 0.05
MAX_RETRIES = 4
TIMEOUT_S = 60


def config_sha256() -> str:
    return canonical_config_sha256(FROZEN_CALL_CONFIG)


def billing_window(t: dt.datetime) -> str:
    t = t.astimezone(dt.timezone.utc)
    if t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10):
        return "peak"
    return "offpeak"


def load_rows(phase: str) -> list[dict[str, Any]]:
    spec = PHASES[phase]
    return load_verified_manifest_rows(
        spec["manifest"],
        spec["manifest_sha256"],
        spec["split"],
        spec["rows"],
        sample_per_cell=spec["sample_per_cell"],
    )


class Harvester(BaseHarvester):
    """DeepSeek-V4.1-Flash first-party v2 harvester."""

    def __init__(
        self,
        phase: str,
        api_key: str,
        concurrency: int,
        *,
        out_dir_override: Path | None = None,
        endpoint_override: str | None = None,
        ssl_ctx: ssl.SSLContext = SSL_CTX,
    ) -> None:
        self.phase = phase
        super().__init__(
            out_dir=out_dir_override or PHASES[phase]["out"],
            cfg_sha=config_sha256(),
            api_key=api_key,
            concurrency=concurrency,
            endpoint_override=endpoint_override,
            ssl_ctx=ssl_ctx,
            timeout_s=TIMEOUT_S,
        )
        if self.loader.combined_hash != FROZEN_CALL_CONFIG["prompt_combined_sha256"]:
            raise Abort("prompt hash mismatch")

    def build_payload(self, canonical_url: str) -> dict[str, Any]:
        c = FROZEN_CALL_CONFIG
        return {
            "model": c["model"],
            "messages": [
                {"role": "system", "content": self.loader.system_text},
                {"role": "user", "content": self.loader.render_user_prompt(canonical_url)},
            ],
            "thinking": c["thinking"],
            "temperature": c["temperature"],
            "max_tokens": c["max_tokens"],
            "response_format": c["response_format"],
            "stream": c["stream"],
        }

    def classify(self, row: dict[str, Any]) -> dict[str, Any]:
        sha = row["canonical_url_sha256"]
        payload = self.build_payload(row["canonical_url"])
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        def _build_ledger(
            attempt: int,
            status: int,
            data: dict[str, Any] | None,
            detail: str,
            latency_ms: float,
            t_req: dt.datetime,
        ) -> dict[str, Any]:
            return {
                "canonical_url_sha256": sha,
                "sample_id": row["sample_id"],
                "attempt": attempt,
                "http_status": status,
                "detail": detail if status != 200 else "",
                "model": FROZEN_CALL_CONFIG["model"],
                "server_model": (data or {}).get("model"),
                "response_id": (data or {}).get("id"),
                "usage": (data or {}).get("usage"),
                "latency_ms": latency_ms,
                "timestamp": t_req.isoformat(),
                "billing_window": billing_window(t_req),
                "config_sha256": self.cfg_sha,
                "phase": self.phase,
            }

        status, data, _detail, attempt, latency_ms = self.post_with_retries(
            url=BASE_URL + "/chat/completions",
            headers=headers,
            body=payload,
            max_retries=MAX_RETRIES,
            retry_statuses=(-1, 429, 500, 502, 503, 504),
            abort_statuses=(401, 402, 403),
            honor_retry_after=False,
            ledger_builder=_build_ledger,
        )

        rec: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "canonical_url_sha256": sha,
            "dataset": row["dataset"],
            "split": row["split"],
            "manifest_hash": row.get("manifest_hash", ""),
            "provider": FROZEN_CALL_CONFIG["provider"],
            "model": FROZEN_CALL_CONFIG["model"],
            "server_model": None,
            "model_version": None,
            "system_fingerprint": None,
            "response_id": None,
            "finish_reason": None,
            "prompt_version": "v2",
            "prompt_combined_sha256": self.loader.combined_hash,
            "config_sha256": self.cfg_sha,
            "repeat_index": 0,
            "raw_response": "",
            "reasoning_content_present": False,
            "parsed_verdict": None,
            "parsed_probability": None,
            "parsed_signals": [],
            "agreement_violation": False,
            "input_tokens": None,
            "output_tokens": None,
            "prompt_cache_hit_tokens": None,
            "prompt_cache_miss_tokens": None,
            "usage_missing": True,
            "latency_ms": latency_ms,
            "retry_count": attempt,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "error_status": None,
        }
        if status != 200 or data is None:
            rec["error_status"] = f"http_{status}" if status > 0 else "network_error"
            return rec

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage")
        rec.update(
            {
                "server_model": data.get("model"),
                "model_version": data.get("model"),
                "system_fingerprint": data.get("system_fingerprint"),
                "response_id": data.get("id"),
                "finish_reason": choice.get("finish_reason"),
                "raw_response": msg.get("content") or "",
                "reasoning_content_present": bool(msg.get("reasoning_content")),
            }
        )
        if isinstance(usage, dict):
            rec.update(
                {
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
                    "prompt_cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
                    "usage_missing": usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None,
                }
            )

        self.check_server_model_drift(rec["server_model"])
        if rec["finish_reason"] == "length":
            rec["error_status"] = "truncated"
        elif not rec["raw_response"].strip():
            rec["error_status"] = "empty_response"
        else:
            rec["error_status"] = self.validate_and_populate_verdict(rec, rec["raw_response"])
        if not rec["server_model"]:
            rec["error_status"] = "missing_server_model"
        return rec

    def run(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        done_ok = {
            r["canonical_url_sha256"]
            for r in read_jsonl(self.cache_path)
            if r.get("error_status") == "success"
        }
        unique: dict[str, dict[str, Any]] = {}
        for row in rows:
            unique.setdefault(row["canonical_url_sha256"], row)
        todo = [row for sha, row in unique.items() if sha not in done_ok]
        print(
            f"[{self.phase}] manifest rows={len(rows)} unique URLs={len(unique)} "
            f"already done={len(unique) - len(todo)} to request={len(todo)}",
            flush=True,
        )
        self.run_concurrent(
            todo,
            self.classify,
            fail_rate_abort=FAIL_RATE_ABORT,
            log_every=250,
            unit_label="URL/s",
        )
        return self.summarize(rows, unique)

    def summarize(self, rows: list[dict[str, Any]], unique: dict[str, dict[str, Any]]) -> dict[str, Any]:
        cache = {
            r["canonical_url_sha256"]: r
            for r in read_jsonl(self.cache_path)
            if r.get("error_status") == "success"
        }
        ledger = read_jsonl(self.ledger_path)
        last_fail = {r["canonical_url_sha256"]: r["error_status"] for r in read_jsonl(self.failures_path)}
        unresolved = {sha: last_fail.get(sha, "not_attempted") for sha in unique if sha not in cache}

        cost = 0.0
        tok: Counter[str] = Counter()
        for entry in ledger:
            usage = entry.get("usage") or {}
            prices = PRICES[entry.get("billing_window", "offpeak")]
            hit = usage.get("prompt_cache_hit_tokens") or 0
            miss = usage.get("prompt_cache_miss_tokens")
            if miss is None:
                miss = (usage.get("prompt_tokens") or 0) - hit
            out = usage.get("completion_tokens") or 0
            cost += (hit * prices["hit"] + miss * prices["miss"] + out * prices["out"]) / 1e6
            tok["hit"] += hit
            tok["miss"] += miss
            tok["out"] += out

        latencies = sorted(r["latency_ms"] for r in cache.values())
        n_ok_rows = sum(1 for r in rows if r["canonical_url_sha256"] in cache)
        summary: dict[str, Any] = {
            "phase": self.phase,
            "config_sha256": self.cfg_sha,
            "manifest": str(PHASES[self.phase]["manifest"].relative_to(REPO)),
            "manifest_sha256": PHASES[self.phase]["manifest_sha256"],
            "manifest_rows": len(rows),
            "unique_urls": len(unique),
            "duplicate_row_reuses": len(rows) - len(unique),
            "unique_success": len(cache),
            "rows_with_valid_response": n_ok_rows,
            "true_failed_fallback_rows": len(rows) - n_ok_rows,
            "unresolved_by_status": dict(Counter(unresolved.values())),
            "http_attempts": len(ledger),
            "http_status_counts": dict(Counter(str(e["http_status"]) for e in ledger)),
            "server_models": dict(Counter(r["server_model"] for r in cache.values())),
            "system_fingerprints": dict(Counter(r["system_fingerprint"] for r in cache.values())),
            "finish_reasons": dict(Counter(r["finish_reason"] for r in cache.values())),
            "reasoning_content_present": sum(r["reasoning_content_present"] for r in cache.values()),
            "usage_missing": sum(r["usage_missing"] for r in cache.values()),
            "agreement_violations": sum(r["agreement_violation"] for r in cache.values()),
            "tokens": dict(tok),
            "estimated_cost_usd_list_price": round(cost, 4),
            "billing_windows": dict(Counter(e.get("billing_window") for e in ledger)),
            "latency_ms_p50": percentile(latencies, 0.5),
            "latency_ms_p95": percentile(latencies, 0.95),
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "note": "Engineering summary only. No 2x2 / rho / epsilon computed here.",
        }
        if self.phase == "smoke":
            ok = (
                summary["unique_success"] / max(1, summary["unique_urls"]) >= SMOKE_PASS_MIN_SUCCESS
                and len(summary["server_models"]) == 1
                and summary["usage_missing"] == 0
                and summary["reasoning_content_present"] == 0
            )
            summary["smoke_gate"] = "PASS" if ok else "FAIL"
        write_summary_and_manifest(self.out, summary)
        return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--dry-run", action="store_true", help="validate manifest/prompt/config, print one payload, no network")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--allow-peak", action="store_true", help="permit running inside DeepSeek peak windows (2x price)")
    args = parser.parse_args(argv)

    try:
        rows = load_rows(args.phase)
        out = PHASES[args.phase]["out"]
        ensure_frozen_config(
            out,
            {"config": FROZEN_CALL_CONFIG, "config_sha256": config_sha256()},
            dry_run=args.dry_run,
        )
        now = dt.datetime.now(dt.timezone.utc)
        print(f"CA={CA_SOURCE}  config_sha256={config_sha256()}  window={billing_window(now)}  rows={len(rows)}")
        if args.dry_run:
            h = Harvester.__new__(Harvester)
            h.loader = FrozenPromptLoader(version="v2")
            payload = Harvester.build_payload(h, rows[0]["canonical_url"])
            payload["messages"][0]["content"] = payload["messages"][0]["content"][:80] + "..."
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            print(
                f"DRY RUN OK: unique URLs={len({r['canonical_url_sha256'] for r in rows})}; "
                f"API key present={'yes' if load_api_key('DEEPSEEK_API_KEY') else 'NO'}"
            )
            return 0
        if billing_window(now) == "peak" and not args.allow_peak:
            raise Abort("inside DeepSeek peak window (Mon-Fri 01-04/06-10 UTC); rerun off-peak or pass --allow-peak")
        if args.phase == "e9" and (PHASES["e9"]["out"] / "summary.json").exists():
            raise Abort(
                "E9 harvest is complete and final (PR-049); re-running would re-sample failed items. "
                "Delete nothing; record a new decision first if a re-run is really intended."
            )
        if args.phase == "e9":
            verify_smoke_gate_passed(PHASES["smoke"]["out"] / "summary.json", config_sha256())
        key = load_api_key("DEEPSEEK_API_KEY")
        if not key:
            raise Abort("DEEPSEEK_API_KEY not found in environment or .env")
        summary = Harvester(args.phase, key, args.concurrency).run(rows)
        print(json.dumps(summary, indent=2))
        if args.phase == "smoke":
            print(f"SMOKE GATE: {summary['smoke_gate']}")
        return 0 if summary.get("smoke_gate", "PASS") == "PASS" else 2
    except Abort as err:
        print(f"ABORTED (fail-closed): {err}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
