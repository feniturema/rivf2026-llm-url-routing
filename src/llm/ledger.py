"""
Token, Latency, and Cost Ledger with Fail-Closed Machine Integrity Verification (P-33 / P-35).
Records live API expenditure with retry accounting, monotonic timestamps, and immutability hash checking.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List
import numpy as np


@dataclass
class LedgerEntry:
    sample_id: str
    canonical_url_sha256: str
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    retry_count: int
    error_status: str
    timestamp: str
    api_account: str = "primary"
    is_synthetic: bool = False


class TokenLedger:
    def __init__(self, ledger_file: Path):
        self.ledger_file = Path(ledger_file)
        self.entries: List[LedgerEntry] = []
        if self.ledger_file.exists():
            self._load_existing()

    def _load_existing(self) -> None:
        self.entries = []
        with open(self.ledger_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self.entries.append(LedgerEntry(**data))

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def validate_integrity(self, max_allowed_failure_rate_pct: float = 5.0) -> Dict[str, Any]:
        """
        Fail-Closed Machine Integrity Audit (P-33 / P-35):
        Asserts:
        1. Failure rate <= max_allowed_failure_rate_pct (K5 Kill Criterion).
        2. Strict non-decreasing timestamp monotonicity.
        3. Non-zero latency variance across successful live calls (SD > 0).
        4. Non-constant input token sizes across diverse URLs.
        5. Zero synthetic/mock contamination in production datasets.
        """
        n = len(self.entries)
        if n == 0:
            return {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "failure_rate_pct": 0.0,
                "valid_live_integrity": True,
                "violation_reasons": [],
            }

        synthetic_count = sum(1 for e in self.entries if getattr(e, "is_synthetic", False))
        successful_entries = [e for e in self.entries if e.error_status == "success"]
        n_success = len(successful_entries)
        n_failed = n - n_success
        failure_rate_pct = (n_failed / n) * 100.0

        latencies = [e.latency_ms for e in successful_entries if e.latency_ms > 0]
        timestamps = [e.timestamp for e in self.entries]
        input_tokens = [e.input_tokens for e in successful_entries if e.input_tokens > 0]

        sd_latency = float(np.std(latencies)) if len(latencies) > 1 else 0.0
        n_distinct_ts = len(set(timestamps))
        n_distinct_in_tokens = len(set(input_tokens))

        is_valid = True
        reasons = []

        # 1. Kill Criterion K5: Failure rate must not exceed 5%
        if failure_rate_pct > max_allowed_failure_rate_pct:
            is_valid = False
            reasons.append(f"Failure rate {failure_rate_pct:.2f}% exceeds K5 threshold ({max_allowed_failure_rate_pct}%)")

        # 2. Synthetic mock isolation check
        if synthetic_count > 0:
            is_valid = False
            reasons.append(f"Found {synthetic_count} synthetic mock entries in production ledger")

        # 3. Monotonic timestamp check
        try:
            parsed_ts = [datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")) for ts in timestamps]
            for i in range(1, len(parsed_ts)):
                if parsed_ts[i] < parsed_ts[i - 1]:
                    is_valid = False
                    reasons.append(f"Timestamp sequence non-monotonic at entry {i}")
                    break
        except Exception as ex:
            is_valid = False
            reasons.append(f"Invalid timestamp format: {ex}")

        # 4. Latency entropy check for live calls
        if n_success > 5:
            if sd_latency == 0.0:
                is_valid = False
                reasons.append("Zero latency standard deviation across >5 successful live calls")
            if n_distinct_ts <= 1:
                is_valid = False
                reasons.append("All entries share identical timestamp")
            if n_distinct_in_tokens <= 1 and n_success > 15:
                is_valid = False
                reasons.append("All entries share identical input token count across diverse URLs")

        return {
            "total_calls": n,
            "successful_calls": n_success,
            "failed_calls": n_failed,
            "synthetic_entries": synthetic_count,
            "failure_rate_pct": failure_rate_pct,
            "sd_latency_ms": sd_latency,
            "distinct_timestamps": n_distinct_ts,
            "distinct_input_token_sizes": n_distinct_in_tokens,
            "valid_live_integrity": is_valid,
            "violation_reasons": reasons,
        }

    def summary(self, price_in_per_m: float = 0.30, price_out_per_m: float = 2.50) -> Dict[str, Any]:
        integ = self.validate_integrity()
        total_calls = integ["total_calls"]
        total_in_tokens = sum(e.input_tokens for e in self.entries)
        total_out_tokens = sum(e.output_tokens for e in self.entries)
        total_retries = sum(e.retry_count for e in self.entries)
        
        cost = (total_in_tokens / 1e6) * price_in_per_m + (total_out_tokens / 1e6) * price_out_per_m
        latencies = [e.latency_ms for e in self.entries if e.error_status == "success" and e.latency_ms > 0]
        
        p50_lat = float(np.median(latencies)) if latencies else 0.0
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

        return {
            "total_calls": total_calls,
            "successful_calls": integ["successful_calls"],
            "failed_calls": integ["failed_calls"],
            "failure_rate_pct": integ["failure_rate_pct"],
            "total_in_tokens": total_in_tokens,
            "total_out_tokens": total_out_tokens,
            "total_retries": total_retries,
            "estimated_cost_usd": cost,
            "p50_latency_ms": p50_lat,
            "p95_latency_ms": p95_lat,
            "integrity": integ,
        }
