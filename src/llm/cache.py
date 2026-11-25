"""
Content-Addressed Write-Once JSONL LLM Response Cache (D-12 / D-18 / P-38).
Key = sha256(canonical_url) ‖ model ‖ prompt_version ‖ repeat_index.
Guarantees byte-level persistence of raw LLM responses and server identifiers.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Set


class LLMCache:
    def __init__(self, cache_file: Path):
        self.cache_file = Path(cache_file)
        self.lock = threading.Lock()
        self._cache_index: Dict[str, Dict[str, Any]] = {}
        self._load_existing()

    def _make_key(self, canonical_url_sha256: str, model: str, prompt_version: str, repeat_index: int = 0) -> str:
        return f"{canonical_url_sha256}::{model}::{prompt_version}::{repeat_index}"

    def _load_existing(self) -> None:
        if not self.cache_file.exists():
            return
        with open(self.cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    sha = record.get("canonical_url_sha256", "")
                    if len(sha) == 64:
                        k = self._make_key(
                            sha,
                            record["model"],
                            record["prompt_version"],
                            record.get("repeat_index", 0),
                        )
                        self._cache_index[k] = record
                except Exception:
                    continue

    def get(self, canonical_url_sha256: str, model: str, prompt_version: str, repeat_index: int = 0) -> Optional[Dict[str, Any]]:
        k = self._make_key(canonical_url_sha256, model, prompt_version, repeat_index)
        return self._cache_index.get(k)

    def put(self, record: Dict[str, Any]) -> None:
        sha = record.get("canonical_url_sha256", "")
        if len(sha) != 64 or not all(c in "0123456789abcdefABCDEF" for c in sha):
            raise ValueError(f"Invalid canonical_url_sha256 in cache record: '{sha}' (must be 64-char hex)")

        k = self._make_key(
            sha,
            record["model"],
            record["prompt_version"],
            record.get("repeat_index", 0),
        )
        with self.lock:
            if k in self._cache_index:
                return  # Write once
            self._cache_index[k] = record
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    def get_all_hashes(self) -> Set[str]:
        return set(rec["canonical_url_sha256"] for rec in self._cache_index.values())

    def __len__(self) -> int:
        return len(self._cache_index)
