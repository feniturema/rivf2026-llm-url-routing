"""
Universal OpenAI-Compatible Async Client for RIVF 2026.
Supports GPT-5.6 Luna Max, GPT-4o-mini, and any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional
import httpx

from .cache import LLMCache
from .ledger import LedgerEntry, TokenLedger
from .prompt import FROZEN_COMBINED_SHA256, FrozenPromptLoader
from .schema import validate_llm_response


class OpenAIAsyncClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "gpt-5.6-luna",
        prompt_version: str = "v2",
        cache: Optional[LLMCache] = None,
        ledger: Optional[TokenLedger] = None,
        timeout_s: float = 20.0,
        max_retries: int = 3,
        api_account: str = "openai_primary",
    ):
        # Auto-load from .env or ~/.env
        if not api_key:
            for env_path in [Path(".env"), Path.home() / ".env"]:
                if env_path.exists():
                    for line in env_path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line.startswith("OPENAI_API_KEY="):
                            api_key = line.replace("OPENAI_API_KEY=", "").strip().strip("\"'")
                            break
                        elif line.startswith("sk-"):
                            api_key = line
                            break
                    if api_key:
                        break
            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY", "")

        if not base_url:
            for env_path in [Path(".env"), Path.home() / ".env"]:
                if env_path.exists():
                    for line in env_path.read_text(encoding="utf-8").splitlines():
                        if line.strip().startswith("OPENAI_BASE_URL="):
                            base_url = line.replace("OPENAI_BASE_URL=", "").strip().strip("\"'")
                            break
                    if base_url:
                        break
            if not base_url:
                base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.prompt_version = prompt_version
        self.cache = cache
        self.ledger = ledger
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.api_account = api_account
        self.prompt_loader = FrozenPromptLoader(version=prompt_version)

    async def classify_url(
        self,
        canonical_url: str,
        sample_id: str,
        dataset: str,
        stratum: str = "S-STRICT",
        split: str = "TEST",
        manifest_hash: str = "pilot_v1",
        repeat_index: int = 0,
    ) -> Dict[str, Any]:
        """Classify URL using OpenAI-compatible endpoint."""
        import hashlib
        canonical_url_sha256 = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()

        # Check Cache
        if self.cache is not None:
            hit = self.cache.get(
                canonical_url_sha256=canonical_url_sha256,
                model=self.model_name,
                prompt_version=self.prompt_version,
                repeat_index=repeat_index,
            )
            if hit:
                rec = dict(hit)
                rec["sample_id"] = sample_id
                rec["dataset"] = dataset
                rec["stratum"] = stratum
                rec["split"] = split
                rec["manifest_hash"] = manifest_hash
                return rec

        prompt_text = self.prompt_loader.render_user_prompt(canonical_url)
        system_text = self.prompt_loader.system_text

        raw_resp = ""
        in_tokens = 0
        out_tokens = 0
        latency_ms = 0.0
        retry_count = 0
        error_status = "success"

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt_text}
            ],
            "response_format": {"type": "json_object"}
        }

        # Reasoning models share max_completion_tokens across reasoning and final JSON output
        if any(tok in self.model_name for tok in ("gpt-5", "gpt-6", "o1", "o3")):
            payload["max_completion_tokens"] = 1024
        else:
            payload["max_tokens"] = 256
            payload["temperature"] = 0.0

        async with httpx.AsyncClient(timeout=self.timeout_s) as http_client:
            for attempt in range(self.max_retries + 1):
                t0 = time.perf_counter()
                try:
                    resp = await http_client.post(url, headers=headers, json=payload)
                    latency_ms = (time.perf_counter() - t0) * 1000.0

                    if resp.status_code == 200:
                        data = resp.json()
                        choices = data.get("choices", [])
                        if choices:
                            raw_resp = choices[0].get("message", {}).get("content", "")
                        usage = data.get("usage", {})
                        in_tokens = usage.get("prompt_tokens", 0)
                        out_tokens = usage.get("completion_tokens", 0)
                        server_model_version = data.get("model", f"{self.model_name}-live")
                        system_fingerprint = data.get("system_fingerprint", "")
                        response_id = data.get("id", "")
                        error_status = "success"
                        break
                    elif resp.status_code in (429, 500, 502, 503, 504):
                        retry_count += 1
                        if attempt == self.max_retries:
                            error_status = f"http_{resp.status_code}"
                        else:
                            backoff = 5.0 + (3.0 * attempt) + random.uniform(0.5, 1.5)
                            await asyncio.sleep(backoff)
                    else:
                        error_status = f"http_{resp.status_code}"
                        break
                except Exception as e:
                    retry_count += 1
                    if attempt == self.max_retries:
                        error_status = f"exception_{type(e).__name__}"
                    else:
                        await asyncio.sleep(3.0 + (2.0 * attempt))

        # Parse and Validate Response
        if error_status == "success":
            if not raw_resp.strip():
                error_status = "empty_response"
                parsed_obj, agreement_violation = None, False
            else:
                parsed_obj, agreement_violation, parse_status = validate_llm_response(raw_resp)
                if parse_status != "success":
                    error_status = parse_status
        else:
            parsed_obj, agreement_violation = None, False
            response_id = ""

        record = {
            "sample_id": sample_id,
            "canonical_url_sha256": canonical_url_sha256,
            "dataset": dataset,
            "stratum": stratum,
            "split": split,
            "manifest_hash": manifest_hash,
            "provider": "openai_compatible",
            "api_account": self.api_account,
            "model": self.model_name,
            "model_version": server_model_version if error_status == "success" else "unknown",
            "system_fingerprint": system_fingerprint if error_status == "success" else "",
            "response_id": response_id,
            "prompt_version": self.prompt_version,
            "prompt_combined_sha256": self.prompt_loader.combined_hash,
            "repeat_index": repeat_index,
            "raw_response": raw_resp,
            "parsed_verdict": parsed_obj.verdict if parsed_obj else None,
            "parsed_probability": parsed_obj.probability if parsed_obj else None,
            "parsed_signals": parsed_obj.signals if parsed_obj else [],
            "agreement_violation": agreement_violation,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "latency_ms": latency_ms,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "retry_count": retry_count,
            "error_status": error_status,
        }

        # Write to Cache & Ledger
        if self.cache is not None and error_status == "success":
            self.cache.put(record)
        if self.ledger is not None:
            self.ledger.record(LedgerEntry(
                sample_id=sample_id,
                canonical_url_sha256=canonical_url_sha256,
                model=self.model_name,
                prompt_version=self.prompt_version,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                latency_ms=latency_ms,
                retry_count=retry_count,
                error_status=error_status,
                timestamp=record["timestamp"],
                api_account=self.api_account,
            ))

        return record
