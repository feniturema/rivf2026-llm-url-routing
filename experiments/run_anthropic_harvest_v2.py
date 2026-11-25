#!/usr/bin/env python3
"""
Claude Haiku 5.5 harvester for the v2 camera-ready re-collection (E9 cross-vendor).
"""

from __future__ import annotations

import argparse
import copy
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

BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2026-10-21"
MODEL_SLUG = "claude-haiku-5-5"

STRUCTURED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["phishing", "legitimate"],
            "description": "Classification decision: phishing or legitimate",
        },
        "probability": {
            "type": "number",
            "description": "Estimated probability that the URL is phishing, P(phishing|URL) in [0.00, 1.00]",
        },
        "signals": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "1 to 3 concise phrases identifying the dominant lexical, brand, or structural signals",
        },
    },
    "required": ["verdict", "probability", "signals"],
    "additionalProperties": False,
}

FROZEN_CALL_CONFIG: dict[str, Any] = {
    "provider": "anthropic_first_party",
    "endpoint": BASE_URL + "/v1/messages",
    "anthropic_version": ANTHROPIC_VERSION,
    "model": "claude-haiku-5-5",
    "thinking": {"type": "disabled"},
    "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": STRUCTURED_SCHEMA}},
    "sampling": "provider default (temperature/top_p/top_k not accepted on claude-haiku-5-5)",
    "max_tokens": 256,
    "messages_layout": "system=prompt_v2.system.txt, user=prompt_v2.user.txt rendered with canonical_url",
    "schema_derivation": "prompt_v2.schema.json minus signals.maxItems, plus additionalProperties=false",
    "prompt_version": "v2",
    "prompt_combined_sha256": FROZEN_V2_COMBINED_SHA256,
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

PRICES = {"in": 0.10, "out": 0.50, "cache_read": 0.01, "cache_write_5m": 0.125}
SMOKE_PASS_MIN_SUCCESS = 0.98
FAIL_RATE_ABORT = 0.05
MAX_RETRIES = 6
TIMEOUT_S = 60


def derive_schema(frozen: dict[str, Any]) -> dict[str, Any]:
    derived = copy.deepcopy(frozen)
    derived["properties"]["signals"].pop("maxItems", None)
    derived["additionalProperties"] = False
    return derived


def config_sha256() -> str:
    return canonical_config_sha256(FROZEN_CALL_CONFIG)


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
    """Claude Haiku 5.5 first-party v2 harvester."""

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
        if derive_schema(json.loads(self.loader.schema_json)) != STRUCTURED_SCHEMA:
            raise Abort("structured-output schema is not the documented derivation of prompt_v2.schema.json")

    def build_payload(self, canonical_url: str) -> dict[str, Any]:
        c = FROZEN_CALL_CONFIG
        return {
            "model": c["model"],
            "max_tokens": c["max_tokens"],
            "system": self.loader.system_text,
            "messages": [{"role": "user", "content": self.loader.render_user_prompt(canonical_url)}],
            "thinking": c["thinking"],
            "output_config": c["output_config"],
        }

    def classify(self, row: dict[str, Any]) -> dict[str, Any]:
        sha = row["canonical_url_sha256"]
        payload = self.build_payload(row["canonical_url"])
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

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
                "config_sha256": self.cfg_sha,
                "phase": self.phase,
            }

        status, data, _detail, attempt, latency_ms = self.post_with_retries(
            url=BASE_URL + "/v1/messages",
            headers=headers,
            body=payload,
            max_retries=MAX_RETRIES,
            retry_statuses=(-1, 429, 500, 502, 503, 504, 529),
            abort_statuses=(400, 401, 403, 404),
            honor_retry_after=True,
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
            "response_id": None,
            "stop_reason": None,
            "prompt_version": "v2",
            "prompt_combined_sha256": self.loader.combined_hash,
            "config_sha256": self.cfg_sha,
            "repeat_index": 0,
            "raw_response": "",
            "thinking_blocks": 0,
            "parsed_verdict": None,
            "parsed_probability": None,
            "parsed_signals": [],
            "agreement_violation": False,
            "input_tokens": None,
            "output_tokens": None,
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None,
            "usage_missing": True,
            "latency_ms": latency_ms,
            "retry_count": attempt,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "error_status": None,
        }
        if status != 200 or data is None:
            rec["error_status"] = f"http_{status}" if status > 0 else "network_error"
            return rec

        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = data.get("usage")
        rec.update(
            {
                "server_model": data.get("model"),
                "model_version": data.get("model"),
                "response_id": data.get("id"),
                "stop_reason": data.get("stop_reason"),
                "raw_response": text,
                "thinking_blocks": sum(1 for b in blocks if b.get("type") == "thinking"),
            }
        )
        if isinstance(usage, dict):
            rec.update(
                {
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
                    "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
                    "usage_missing": usage.get("input_tokens") is None or usage.get("output_tokens") is None,
                }
            )

        self.check_server_model_drift(rec["server_model"])
        if rec["stop_reason"] == "refusal":
            rec["error_status"] = "refusal"
        elif rec["stop_reason"] == "max_tokens":
            rec["error_status"] = "truncated"
        elif not text.strip():
            rec["error_status"] = "empty_response"
        else:
            rec["error_status"] = self.validate_and_populate_verdict(rec, text)
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
            in_t = usage.get("input_tokens") or 0
            out_t = usage.get("output_tokens") or 0
            cr_t = usage.get("cache_read_input_tokens") or 0
            cw_t = usage.get("cache_creation_input_tokens") or 0
            cost += (
                in_t * PRICES["in"]
                + out_t * PRICES["out"]
                + cr_t * PRICES["cache_read"]
                + cw_t * PRICES["cache_write_5m"]
            ) / 1e6
            tok["in"] += in_t
            tok["out"] += out_t
            tok["cache_read"] += cr_t
            tok["cache_write"] += cw_t

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
            "stop_reasons": dict(Counter(r["stop_reason"] for r in cache.values())),
            "thinking_blocks_total": sum(r["thinking_blocks"] for r in cache.values()),
            "usage_missing": sum(r["usage_missing"] for r in cache.values()),
            "agreement_violations": sum(r["agreement_violation"] for r in cache.values()),
            "tokens": dict(tok),
            "estimated_cost_usd_list_price": round(cost, 4),
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
                and summary["thinking_blocks_total"] == 0
            )
            summary["smoke_gate"] = "PASS" if ok else "FAIL"
        write_summary_and_manifest(self.out, summary)
        return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--dry-run", action="store_true", help="validate manifest/prompt/schema/config, print one payload, no network")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args(argv)

    try:
        rows = load_rows(args.phase)
        loader = FrozenPromptLoader(version="v2")
        if derive_schema(json.loads(loader.schema_json)) != STRUCTURED_SCHEMA:
            raise Abort("structured-output schema is not the documented derivation of prompt_v2.schema.json")
        ensure_frozen_config(
            PHASES[args.phase]["out"],
            {"config": FROZEN_CALL_CONFIG, "config_sha256": config_sha256()},
            dry_run=args.dry_run,
        )
        print(f"CA={CA_SOURCE}  config_sha256={config_sha256()}  rows={len(rows)}")
        if args.dry_run:
            h = Harvester.__new__(Harvester)
            h.loader = loader
            payload = Harvester.build_payload(h, rows[0]["canonical_url"])
            payload["system"] = payload["system"][:80] + "..."
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            print(
                f"DRY RUN OK: unique URLs={len({r['canonical_url_sha256'] for r in rows})}; "
                f"API key present={'yes' if load_api_key('ANTHROPIC_API_KEY') else 'NO'}"
            )
            return 0
        if args.phase == "e9" and (PHASES["e9"]["out"] / "summary.json").exists():
            raise Abort(
                "E9 harvest is complete and final (PR-049); re-running would re-sample failed items. "
                "Delete nothing; record a new decision first if a re-run is really intended."
            )
        if args.phase == "e9":
            verify_smoke_gate_passed(PHASES["smoke"]["out"] / "summary.json", config_sha256())
        key = load_api_key("ANTHROPIC_API_KEY")
        if not key:
            raise Abort("ANTHROPIC_API_KEY not found in environment or .env")
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
