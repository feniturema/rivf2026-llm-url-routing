#!/usr/bin/env python3
"""
v2 camera-ready harvester for the primary model (Gemini 3.5 Flash-Lite: E4, E3, E8) and the
OpenAI replication (GPT-5.6 Luna / GPT-6 Luna: E9). DeepSeek / Haiku have their own runners.

"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
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
    DATA_MANIFESTS_DIR as MAN,
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
    sha256_file,
    verify_smoke_gate_passed,
    write_summary_and_manifest,
)
from src.llm.prompt import FROZEN_V2_COMBINED_SHA256, FrozenPromptLoader  # noqa: E402

FREEZE_V2 = (
    REPO / "results/policies/policy_freeze_v2.json",
    "3f7d24fcccb99e008d4eeb421df431270f0a1978ccbb1e8b2c16a5dff9375ae5",
)
SEL = REPO / "results/v2_runs/selections"
BRIDGE_MIN_AGREEMENT = 0.98
SMOKE_PASS_MIN_SUCCESS = 0.98
FAIL_RATE_ABORT = 0.05
MAX_RETRIES = 6
TIMEOUT_S = 60


def _frozen_schema() -> dict[str, Any]:
    schema = json.loads(FrozenPromptLoader(version="v2").schema_json)
    schema.pop("additionalProperties", None)  # identical to the August Gemini client
    return schema


VENDORS: dict[str, dict[str, Any]] = {
    "gemini": {
        "slug": "gemini-3.5-flash-lite",
        "key_env": "GEMINI_API_KEY",
        "config": {
            "provider": "google_gemini_api_first_party",
            "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent",
            "auth": "x-goog-api-key header",
            "model": "gemini-3.5-flash-lite",
            "generationConfig": {
                "temperature": 0.0,
                "topP": 1.0,
                "maxOutputTokens": 256,
                "responseMimeType": "application/json",
                "responseSchema": "prompt_v2.schema.json minus additionalProperties (as in August)",
                "thinkingConfig": {"thinkingLevel": "minimal"},
            },
            "changes_vs_august": (
                "pinned model id (was alias gemini-flash-lite-latest); maxOutputTokens 120->256 "
                "(only affects truncated outputs); thinkingLevel minimal set explicitly (documented default); "
                "modelVersion/responseId/finishReason recorded"
            ),
            "prompt_version": "v2",
            "prompt_combined_sha256": FROZEN_V2_COMBINED_SHA256,
        },
        "prices": {"in": 0.30, "out": 2.50, "cached": 0.03},
        "phases": ["smoke", "bridge", "e3_backfill", "e8", "e4", "e3"],
    },
    "luna": {
        "slug": "gpt-5.6-luna",
        "key_env": "OPENAI_API_KEY",
        "config": {
            "provider": "openai_first_party",
            "endpoint": "https://api.openai.com/v1/chat/completions",
            "model": "gpt-5.6-luna",
            "reasoning_effort": "medium",
            "max_completion_tokens": 1024,
            "response_format": {"type": "json_object"},
            "sampling": "API default (reasoning model)",
            "changes_vs_august": (
                "max_completion_tokens 200->1024 to remove the reasoning-token truncation (P-47); "
                "server model / fingerprint / finish_reason / reasoning_tokens recorded"
            ),
            "prompt_version": "v2",
            "prompt_combined_sha256": FROZEN_V2_COMBINED_SHA256,
        },
        "prices": {"in": 0.20, "out": 1.20, "cached": 0.02},
        "phases": ["smoke", "e9"],
    },
    "luna6": {
        "slug": "gpt-6-luna",
        "key_env": "OPENAI_API_KEY",
        "config": {
            "provider": "openai_first_party",
            "endpoint": "https://api.openai.com/v1/chat/completions",
            "model": "gpt-6-luna",
            "reasoning_effort": "medium",
            "max_completion_tokens": 1024,
            "response_format": {"type": "json_object"},
            "sampling": "API default (reasoning model)",
            "changes_vs_august": (
                "added by amendment A-1 (2026-11-22 11:00 +08); parameters identical to v2 gpt-5.6-luna"
            ),
            "prompt_version": "v2",
            "prompt_combined_sha256": FROZEN_V2_COMBINED_SHA256,
        },
        "prices": {"in": 0.10, "out": 0.50, "cached": 0.01},
        "phases": ["smoke", "e9"],
    },
}

PHASE_SPECS: dict[str, dict[str, Any]] = {
    "smoke": {
        "manifest": "e4_cal_all_13446.csv",
        "sha": "7ecb9929e2939acaea687f16f7e1251d7587125673343e152972b5d35c373195",
        "split": "CAL_ROUTER",
        "rows": 13446,
        "test": False,
    },
    "e4": {
        "manifest": "e4_cal_all_13446.csv",
        "sha": "7ecb9929e2939acaea687f16f7e1251d7587125673343e152972b5d35c373195",
        "split": "CAL_ROUTER",
        "rows": 13446,
        "test": False,
    },
    "e3": {
        "manifest": "e3_test_all_26886.csv",
        "sha": "3a34662b9d86abb8846a0cdbf9079bee7759b142df7a9f5285011a90685b7d7e",
        "split": "TEST",
        "rows": 26886,
        "test": True,
    },
    "e9": {
        "manifest": "e9_shared_test_cohort_3000.csv",
        "sha": "f33b8a178e95d98f4b9071f531b327f51d106918d18e903ab8f07871b31810c4",
        "split": "TEST",
        "rows": 3000,
        "test": True,
    },
    "bridge": {
        "manifest": "e3_test_all_26886.csv",
        "sha": "3a34662b9d86abb8846a0cdbf9079bee7759b142df7a9f5285011a90685b7d7e",
        "split": "TEST",
        "rows": 26886,
        "test": True,
        "select": ("bridge.txt", "fe714f832a727855658b8a093327f2b4ecb4c0e9b0ea42cdccdeb64af778feae"),
    },
    "e3_backfill": {
        "manifest": "e3_test_all_26886.csv",
        "sha": "3a34662b9d86abb8846a0cdbf9079bee7759b142df7a9f5285011a90685b7d7e",
        "split": "TEST",
        "rows": 26886,
        "test": True,
        "select": ("e3_backfill.txt", "40bae39e121b83c43e3e4f015ca8a49bf49315c8eed3008da9b40ff2f2cda51b"),
    },
    "e8": {
        "manifest": "e9_shared_test_cohort_3000.csv",
        "sha": "f33b8a178e95d98f4b9071f531b327f51d106918d18e903ab8f07871b31810c4",
        "split": "TEST",
        "rows": 3000,
        "test": True,
    },
}

KNOWN_HASH_EXCEPTIONS = {
    "PWD2021_042169": "a934a7896af0a955a586c7fda555047103fde0ef59408faa0075be3f7cb03289",
}
E8_RULE = (
    "from the 2,993 unique E9 URLs: the 100 with the highest binary entropy (bits) of specialist_prob_cal "
    "(ties: sha ascending) + 100 drawn with random.Random(42).sample from the remaining URLs sorted by sha; "
    "each requested at repeat_index 1..5 (E3 response is repeat 0)"
)


def config_for(vendor: str, phase: str) -> dict[str, Any]:
    """Returns the frozen call configuration dict for `(vendor, phase)`."""
    cfg = dict(VENDORS[vendor]["config"])
    if phase == "e8":
        cfg["e8_selection_rule"] = E8_RULE
        cfg["repeat_indices"] = [1, 2, 3, 4, 5]
    else:
        cfg["repeat_indices"] = [0]
    if "select" in PHASE_SPECS[phase]:
        cfg["selection_file"], cfg["selection_sha256"] = PHASE_SPECS[phase]["select"]
    return cfg


def config_sha(vendor: str, phase: str) -> str:
    return canonical_config_sha256(config_for(vendor, phase))


def smoke_config_sha(vendor: str) -> str:
    return config_sha(vendor, "smoke")


def h_bits(p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def load_rows(phase: str) -> list[dict[str, Any]]:
    spec = PHASE_SPECS[phase]
    sel_file, sel_sha = spec["select"] if "select" in spec else (None, None)
    rows = load_verified_manifest_rows(
        MAN / spec["manifest"],
        spec["sha"],
        spec["split"],
        spec["rows"],
        known_hash_exceptions=KNOWN_HASH_EXCEPTIONS,
        selection_path=(SEL / sel_file) if sel_file else None,
        selection_sha256=sel_sha,
        sample_per_cell=5 if phase == "smoke" else None,
    )
    for row in rows:
        row["repeat_index"] = 0
    if phase == "e8":
        uniq: dict[str, dict[str, Any]] = {}
        for row in rows:
            uniq.setdefault(row["canonical_url_sha256"], row)
        ranked = sorted(
            uniq.values(),
            key=lambda r: (-h_bits(float(r["specialist_prob_cal"])), r["canonical_url_sha256"]),
        )
        top = ranked[:100]
        top_sha = {r["canonical_url_sha256"] for r in top}
        rest = sorted(
            (r for r in uniq.values() if r["canonical_url_sha256"] not in top_sha),
            key=lambda r: r["canonical_url_sha256"],
        )
        selected = top + random.Random(42).sample(rest, 100)
        expanded: list[dict[str, Any]] = []
        for row in selected:
            stratum = "top_entropy" if row["canonical_url_sha256"] in top_sha else "random_rest"
            for repeat_idx in (1, 2, 3, 4, 5):
                copy_row = dict(row)
                copy_row["repeat_index"] = repeat_idx
                copy_row["e8_stratum"] = stratum
                expanded.append(copy_row)
        rows = expanded
    return rows


def check_test_preconditions(vendor: str = "") -> dict[str, str]:
    got: dict[str, str] = {}
    req = (FREEZE_V2,)
    for path, want in req:
        if not path.exists():
            raise Abort(f"TEST phase blocked: {path.relative_to(REPO)} missing")
        digest = sha256_file(path)
        if digest != want:
            raise Abort(f"TEST phase blocked: {path.relative_to(REPO)} SHA-256 {digest} != frozen {want}")
        got[str(path.relative_to(REPO))] = digest
    return got


class Harvester(BaseHarvester):
    """Gemini 3.5 Flash-Lite and OpenAI Luna v2 harvester."""

    def __init__(
        self,
        vendor: str,
        phase: str,
        api_key: str,
        concurrency: int,
        *,
        out_root: Path = OUT_ROOT,
        endpoint_override: str | None = None,
        ssl_ctx: ssl.SSLContext = SSL_CTX,
    ) -> None:
        self.vendor = vendor
        self.phase = phase
        self.v = VENDORS[vendor]
        super().__init__(
            out_dir=out_root / phase.upper() / self.v["slug"],
            cfg_sha=config_sha(vendor, phase),
            api_key=api_key,
            concurrency=concurrency,
            endpoint_override=endpoint_override,
            ssl_ctx=ssl_ctx,
            timeout_s=TIMEOUT_S,
        )
        self.schema = _frozen_schema()

    def build_request(self, canonical_url: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        system = self.loader.system_text
        user = self.loader.render_user_prompt(canonical_url)
        cfg = self.v["config"]
        if self.vendor == "gemini":
            gc = dict(cfg["generationConfig"])
            gc["responseSchema"] = self.schema
            body = {
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "systemInstruction": {"parts": [{"text": system}]},
                "generationConfig": gc,
            }
            return cfg["endpoint"], {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}, body
        body = {
            "model": cfg["model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": cfg["response_format"],
            "reasoning_effort": cfg["reasoning_effort"],
            "max_completion_tokens": cfg["max_completion_tokens"],
        }
        return cfg["endpoint"], {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, body

    def parse(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.vendor == "gemini":
            cands = data.get("candidates") or []
            u = data.get("usageMetadata") or {}
            parsed = {
                "server_model": data.get("modelVersion"),
                "response_id": data.get("responseId"),
                "system_fingerprint": None,
                "finish_reason": None,
                "text": "",
                "input_tokens": u.get("promptTokenCount"),
                "output_tokens": u.get("candidatesTokenCount"),
                "thinking_tokens": u.get("thoughtsTokenCount"),
                "cached_tokens": u.get("cachedContentTokenCount"),
                "usage_missing": u.get("promptTokenCount") is None or u.get("candidatesTokenCount") is None,
                "block_reason": (data.get("promptFeedback") or {}).get("blockReason"),
            }
            if cands:
                c0 = cands[0]
                parsed["finish_reason"] = c0.get("finishReason")
                parts = (c0.get("content") or {}).get("parts") or []
                parsed["text"] = "".join(x.get("text", "") for x in parts if not x.get("thought"))
            return parsed

        ch = (data.get("choices") or [{}])[0]
        u = data.get("usage") or {}
        return {
            "server_model": data.get("model"),
            "response_id": data.get("id"),
            "system_fingerprint": data.get("system_fingerprint"),
            "finish_reason": ch.get("finish_reason"),
            "text": (ch.get("message") or {}).get("content") or "",
            "input_tokens": u.get("prompt_tokens"),
            "output_tokens": u.get("completion_tokens"),
            "thinking_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "usage_missing": u.get("prompt_tokens") is None or u.get("completion_tokens") is None,
            "block_reason": None,
        }

    def classify_status(self, parsed: dict[str, Any]) -> str:
        fr = parsed["finish_reason"] or ""
        if parsed["block_reason"]:
            return "blocked"
        if fr.upper() in ("MAX_TOKENS", "LENGTH"):
            return "truncated"
        if self.vendor == "gemini" and fr and fr != "STOP":
            return f"finish_{fr}"
        if self.vendor in ("luna", "luna6") and fr and fr != "stop":
            return f"finish_{fr}"
        if not parsed["text"].strip():
            return "empty_response"
        return "parse"

    def classify(self, row: dict[str, Any]) -> dict[str, Any]:
        sha = row["canonical_url_sha256"]
        rep = int(row["repeat_index"])
        url, headers, body = self.build_request(row["canonical_url"])

        def _build_ledger(
            attempt: int,
            status: int,
            data: dict[str, Any] | None,
            detail: str,
            latency_ms: float,
            t_req: dt.datetime,
        ) -> dict[str, Any]:
            pu = self.parse(data) if (status == 200 and data) else {}
            return {
                "canonical_url_sha256": sha,
                "repeat_index": rep,
                "sample_id": row["sample_id"],
                "attempt": attempt,
                "http_status": status,
                "detail": detail if status != 200 else "",
                "model": self.v["config"]["model"],
                "server_model": pu.get("server_model"),
                "response_id": pu.get("response_id"),
                "usage": {
                    k: pu.get(k)
                    for k in ("input_tokens", "output_tokens", "thinking_tokens", "cached_tokens")
                }
                if pu
                else None,
                "latency_ms": latency_ms,
                "timestamp": t_req.isoformat(),
                "config_sha256": self.cfg_sha,
                "phase": self.phase,
            }

        status, data, _detail, attempt, latency_ms = self.post_with_retries(
            url=url,
            headers=headers,
            body=body,
            max_retries=MAX_RETRIES,
            retry_statuses=(-1, 429, 500, 502, 503, 504, 529),
            abort_statuses=(400, 401, 403, 404),
            honor_retry_after=True,
            insufficient_quota_abort=True,
            ledger_builder=_build_ledger,
        )

        rec: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "canonical_url_sha256": sha,
            "dataset": row["dataset"],
            "split": row["split"],
            "manifest_hash": row.get("manifest_hash", ""),
            "provider": self.v["config"]["provider"],
            "model": self.v["config"]["model"],
            "repeat_index": rep,
            "e8_stratum": row.get("e8_stratum"),
            "prompt_version": "v2",
            "prompt_combined_sha256": self.loader.combined_hash,
            "config_sha256": self.cfg_sha,
            "server_model": None,
            "model_version": None,
            "response_id": None,
            "system_fingerprint": None,
            "finish_reason": None,
            "raw_response": "",
            "parsed_verdict": None,
            "parsed_probability": None,
            "parsed_signals": [],
            "agreement_violation": False,
            "input_tokens": None,
            "output_tokens": None,
            "thinking_tokens": None,
            "cached_tokens": None,
            "usage_missing": True,
            "latency_ms": latency_ms,
            "retry_count": attempt,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "error_status": None,
        }
        if status != 200 or data is None:
            rec["error_status"] = f"http_{status}" if status > 0 else "network_error"
            return rec

        parsed = self.parse(data)
        rec.update(
            {
                k: parsed[k]
                for k in (
                    "server_model",
                    "response_id",
                    "system_fingerprint",
                    "finish_reason",
                    "input_tokens",
                    "output_tokens",
                    "thinking_tokens",
                    "cached_tokens",
                    "usage_missing",
                )
            }
        )
        rec["model_version"] = parsed["server_model"]
        rec["raw_response"] = parsed["text"]
        self.check_server_model_drift(parsed["server_model"])

        st = self.classify_status(parsed)
        if st == "parse":
            st = self.validate_and_populate_verdict(rec, parsed["text"])
        if not parsed["server_model"]:
            st = "missing_server_model"
        rec["error_status"] = st
        return rec

    def run(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        done_ok = {
            (r["canonical_url_sha256"], r["repeat_index"])
            for r in read_jsonl(self.cache_path)
            if r.get("error_status") == "success"
        }
        unique: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            unique.setdefault((row["canonical_url_sha256"], int(row["repeat_index"])), row)
        todo = [row for key, row in unique.items() if key not in done_ok]
        print(
            f"[{self.vendor}/{self.phase}] manifest rows={len(rows)} unique requests={len(unique)} "
            f"already done={len(unique) - len(todo)} to request={len(todo)}",
            flush=True,
        )
        self.run_concurrent(
            todo,
            self.classify,
            fail_rate_abort=FAIL_RATE_ABORT,
            log_every=500,
            unit_label="req/s",
        )
        return self.summarize(rows, unique)

    def summarize(self, rows: list[dict[str, Any]], unique: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
        cache = {
            (r["canonical_url_sha256"], r["repeat_index"]): r
            for r in read_jsonl(self.cache_path)
            if r.get("error_status") == "success"
        }
        ledger = read_jsonl(self.ledger_path)
        last_fail = {
            (r["canonical_url_sha256"], r["repeat_index"]): r["error_status"]
            for r in read_jsonl(self.failures_path)
        }
        unresolved = Counter(last_fail.get(k, "not_attempted") for k in unique if k not in cache)

        prices = self.v["prices"]
        cost = 0.0
        tok: Counter[str] = Counter()
        for entry in ledger:
            usage = entry.get("usage") or {}
            in_t = usage.get("input_tokens") or 0
            out_t = usage.get("output_tokens") or 0
            th_t = usage.get("thinking_tokens") or 0
            ca_t = usage.get("cached_tokens") or 0
            if self.vendor == "gemini":
                cost += ((in_t - ca_t) * prices["in"] + ca_t * prices["cached"] + (out_t + th_t) * prices["out"]) / 1e6
            else:
                cost += ((in_t - ca_t) * prices["in"] + ca_t * prices["cached"] + out_t * prices["out"]) / 1e6
            tok["in"] += in_t
            tok["out"] += out_t
            tok["thinking"] += th_t
            tok["cached"] += ca_t

        latencies = sorted(r["latency_ms"] for r in cache.values())
        n_ok_rows = sum(1 for r in rows if (r["canonical_url_sha256"], int(r["repeat_index"])) in cache)
        summary: dict[str, Any] = {
            "vendor": self.vendor,
            "phase": self.phase,
            "config_sha256": self.cfg_sha,
            "ca_source": CA_SOURCE,
            "manifest": PHASE_SPECS[self.phase]["manifest"],
            "manifest_sha256": PHASE_SPECS[self.phase]["sha"],
            "manifest_rows": len(rows),
            "unique_requests": len(unique),
            "duplicate_row_reuses": len(rows) - len(unique),
            "unique_success": len(cache),
            "rows_with_valid_response": n_ok_rows,
            "true_failed_fallback_rows": len(rows) - n_ok_rows,
            "unresolved_by_status": dict(unresolved),
            "http_attempts": len(ledger),
            "http_status_counts": dict(Counter(str(e["http_status"]) for e in ledger)),
            "server_models": dict(Counter(r["server_model"] for r in cache.values())),
            "system_fingerprints": dict(Counter(r["system_fingerprint"] for r in cache.values())),
            "finish_reasons": dict(Counter(r["finish_reason"] for r in cache.values())),
            "usage_missing": sum(r["usage_missing"] for r in cache.values()),
            "agreement_violations": sum(r["agreement_violation"] for r in cache.values()),
            "tokens": dict(tok),
            "estimated_cost_usd_list_price": round(cost, 4),
            "latency_ms_p50": percentile(latencies, 0.5),
            "latency_ms_p95": percentile(latencies, 0.95),
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "note": "Engineering summary only. No 2x2 / rho / epsilon computed here.",
        }
        if self.phase == "bridge":
            aug = {
                r["canonical_url_sha256"]: r.get("parsed_verdict")
                for r in read_jsonl(REPO / "results/E3/e3_gemini_test_cache.jsonl")
                if r.get("model") == "gemini-flash-lite-latest" and r.get("error_status") == "success"
            }
            pairs = [(aug.get(k[0]), r["parsed_verdict"]) for k, r in cache.items()]
            agree = sum(1 for a, b in pairs if a is not None and a == b)
            n_cmp = sum(1 for a, _ in pairs if a is not None)
            rate = agree / n_cmp if n_cmp else 0.0
            cover = len(cache) / max(1, len(unique))
            summary["bridge"] = {
                "compared": n_cmp,
                "agree": agree,
                "agreement": rate,
                "v2_success_coverage": cover,
                "threshold": BRIDGE_MIN_AGREEMENT,
                "flips_aug_to_v2": dict(Counter(f"{a}->{b}" for a, b in pairs if a is not None and a != b)),
            }
            summary["bridge_decision"] = (
                "C_STANDS" if (rate >= BRIDGE_MIN_AGREEMENT and cover >= 0.98) else "FALLBACK_TO_A"
            )
        if self.phase == "smoke":
            ok = (
                summary["unique_success"] / max(1, summary["unique_requests"]) >= SMOKE_PASS_MIN_SUCCESS
                and len(summary["server_models"]) == 1
                and summary["usage_missing"] == 0
            )
            summary["smoke_gate"] = "PASS" if ok else "FAIL"
        write_summary_and_manifest(self.out, summary)
        return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vendor", choices=sorted(VENDORS), required=True)
    parser.add_argument("--phase", choices=sorted(PHASE_SPECS), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args(argv)

    v = VENDORS[args.vendor]
    try:
        if args.phase not in v["phases"]:
            raise Abort(f"phase {args.phase} is not part of the frozen design for {args.vendor}: {v['phases']}")
        rows = load_rows(args.phase)
        extra: dict[str, str] = {}
        if PHASE_SPECS[args.phase]["test"]:
            extra = check_test_preconditions(args.vendor)
        out = OUT_ROOT / args.phase.upper() / v["slug"]
        cfg_payload = {
            "vendor": args.vendor,
            "phase": args.phase,
            "config": config_for(args.vendor, args.phase),
            "config_sha256": config_sha(args.vendor, args.phase),
        }
        ensure_frozen_config(out, cfg_payload, dry_run=args.dry_run, extra_preconditions=extra)
        n_unique = len({(r["canonical_url_sha256"], r["repeat_index"]) for r in rows})
        print(
            f"CA={CA_SOURCE}  vendor={args.vendor} phase={args.phase} "
            f"config_sha256={config_sha(args.vendor, args.phase)} rows={len(rows)} unique_requests={n_unique}"
        )
        if args.dry_run:
            h = Harvester.__new__(Harvester)
            h.vendor, h.v, h.api_key = args.vendor, v, "<redacted>"
            h.loader, h.schema = FrozenPromptLoader(version="v2"), _frozen_schema()
            url, _headers, body = Harvester.build_request(h, rows[0]["canonical_url"])
            if "systemInstruction" in body:
                body["systemInstruction"]["parts"][0]["text"] = (
                    body["systemInstruction"]["parts"][0]["text"][:60] + "..."
                )
            else:
                body["messages"][0]["content"] = body["messages"][0]["content"][:60] + "..."
            print(url)
            print(json.dumps(body, indent=1, ensure_ascii=False)[:1800])
            print(
                f"DRY RUN OK; preconditions={'ok ' + str(list(extra)) if extra else 'n/a (non-TEST)'}; "
                f"API key present={'yes' if load_api_key(v['key_env']) else 'NO'}"
            )
            return 0
        if args.vendor == "gemini" and args.phase in ("e3_backfill", "e3", "e4"):
            bp = OUT_ROOT / "BRIDGE" / v["slug"] / "summary.json"
            dec = json.loads(bp.read_text(encoding="utf-8")).get("bridge_decision") if bp.exists() else None
            need = "C_STANDS" if args.phase == "e3_backfill" else "FALLBACK_TO_A"
            if dec != need:
                raise Abort(f"A-2 gate: phase {args.phase} needs bridge_decision={need}, found {dec}")
        if args.phase == "e9" and (OUT_ROOT / "E9" / "E9_FINAL_manifest.json").exists():
            raise Abort("E9 is locked (E9_FINAL_manifest.json, PR-055); record a new decision before any re-run")
        if args.phase != "smoke":
            verify_smoke_gate_passed(OUT_ROOT / "SMOKE" / v["slug"] / "summary.json", smoke_config_sha(args.vendor))
        key = load_api_key(v["key_env"])
        if not key:
            raise Abort(f"{v['key_env']} not found in environment or .env")
        conc = args.concurrency or (16 if args.vendor == "gemini" else 8)
        summary = Harvester(args.vendor, args.phase, key, conc).run(rows)
        print(json.dumps(summary, indent=2))
        if args.phase == "smoke":
            print(f"SMOKE GATE: {summary['smoke_gate']}")
        return 0 if summary.get("smoke_gate", "PASS") == "PASS" else 2
    except Abort as err:
        print(f"ABORTED (fail-closed): {err}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
