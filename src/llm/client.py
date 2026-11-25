"""
High-Performance Async Gemini API Client.
Supports JSON structured output mode, exponential backoff with jitter,
retry limits, timeout management, content-addressed caching, and ledger accounting.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import httpx

from .cache import LLMCache
from .ledger import LedgerEntry, TokenLedger
from .prompt import FROZEN_COMBINED_SHA256, FrozenPromptLoader
from .schema import validate_llm_response


class GeminiAsyncClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.5-flash-lite",
        prompt_version: str = "v2",
        cache: Optional[LLMCache] = None,
        ledger: Optional[TokenLedger] = None,
        timeout_s: float = 20.0,
        max_retries: int = 3,
        api_account: str = "primary",
    ):
        # Auto-load from .env or ~/.env
        if not api_key and not os.environ.get("GEMINI_API_KEY"):
            for env_path in [Path(".env"), Path.home() / ".env"]:
                if env_path.exists():
                    for line in env_path.read_text(encoding="utf-8").splitlines():
                        if line.strip() and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() == "GEMINI_API_KEY":
                                os.environ["GEMINI_API_KEY"] = v.strip().strip("\"'")
                                break
        
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.cache = cache
        self.ledger = ledger
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.api_account = api_account
        self.prompt_version = prompt_version
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
        """Classify a canonical URL with caching and ledger recording."""
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

        # Perform Live API Call
        prompt_text = self.prompt_loader.render_user_prompt(canonical_url)
        system_text = self.prompt_loader.system_text

        raw_resp = ""
        in_tokens = 0
        out_tokens = 0
        latency_ms = 0.0
        retry_count = 0
        error_status = "success"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
        schema = json.loads(self.prompt_loader.schema_json)
        schema.pop("additionalProperties", None)

        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt_text}]}
            ],
            "systemInstruction": {
                "parts": [{"text": system_text}]
            },
            "generationConfig": {
                "temperature": 0.0,
                "topP": 1.0,
                "maxOutputTokens": 120,
                "responseMimeType": "application/json",
                "responseSchema": schema
            }
        }

        async with httpx.AsyncClient(timeout=self.timeout_s) as http_client:
            for attempt in range(self.max_retries + 1):
                t0 = time.perf_counter()
                try:
                    resp = await http_client.post(url, json=payload)
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            part = candidates[0].get("content", {}).get("parts", [{}])[0]
                            raw_resp = part.get("text", "")
                        
                        usage = data.get("usageMetadata", {})
                        in_tokens = usage.get("promptTokenCount", 359)
                        out_tokens = usage.get("candidatesTokenCount", 60)
                        server_model_version = data.get("modelVersion", f"{self.model_name}-live")
                        response_id = data.get("responseId", "")
                        error_status = "success"
                        break
                    elif resp.status_code in (429, 500, 502, 503, 504):
                        retry_count += 1
                        if attempt == self.max_retries:
                            error_status = f"http_{resp.status_code}"
                        else:
                            # Exponential backoff with jitter for rate limit recovery
                            base_wait = min(60.0, (2.0 ** attempt) * 3.0)
                            backoff = base_wait + random.uniform(1.0, 4.0)
                            await asyncio.sleep(backoff)
                    else:
                        error_status = f"http_{resp.status_code}"
                        break
                except Exception as e:
                    retry_count += 1
                    if attempt == self.max_retries:
                        error_status = f"exception_{type(e).__name__}"
                    else:
                        await asyncio.sleep(5.0 + (3.0 * attempt))

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
            "provider": "google",
            "api_account": self.api_account,
            "model": self.model_name,
            "model_version": server_model_version if error_status == "success" else "unknown",
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
