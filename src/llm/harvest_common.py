"""
Shared fail-closed harvesting infrastructure for v2 first-party LLM collectors.

Provides TLS certificate setup, SHA-256 verification, frozen config locking,
manifest purity/hash validation, thread-safe JSONL logging, retry/circuit-breaker
HTTP execution, model-drift detection, and output manifest sealing.
"""

from __future__ import annotations

import concurrent.futures as cf
import csv
import datetime as dt
import hashlib
import json
import os
import random
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from .prompt import FrozenPromptLoader
from .schema import validate_llm_response

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_ROOT = REPO_ROOT / "results" / "v2_runs"
DATA_MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"


class Abort(RuntimeError):
    """Fail-closed unrecoverable harvest error."""


def build_ssl_context() -> tuple[ssl.SSLContext, str]:
    """Builds a verified TLS context using certifi if available."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where()), "certifi " + certifi.__version__
    except ImportError:
        return ssl.create_default_context(), "system default"


SSL_CTX, CA_SOURCE = build_ssl_context()


def sha256_file(path: Path) -> str:
    """Computes streaming SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_config_sha256(config: dict[str, Any]) -> str:
    """Computes canonical sorted-key SHA-256 hex digest of a call config dict."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Reads non-empty JSON lines from a file; returns [] if missing."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl_locked(path: Path, record: dict[str, Any], lock: threading.Lock) -> None:
    """Appends one JSON line and flushes under a thread lock."""
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


def load_api_key(env_name: str, repo_root: Path = REPO_ROOT) -> str:
    """Reads an API key from process environment or `<repo_root>/.env`."""
    key = os.environ.get(env_name, "").strip()
    if key:
        return key
    env_path = repo_root / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if stripped.startswith(env_name + "="):
                return stripped.split("=", 1)[1].strip().strip("\"'")
    return ""


def load_verified_manifest_rows(
    manifest_path: Path,
    expected_sha256: str,
    expected_split: str,
    expected_rows: int,
    *,
    known_hash_exceptions: dict[str, str] | None = None,
    selection_path: Path | None = None,
    selection_sha256: str | None = None,
    sample_per_cell: int | None = None,
) -> list[dict[str, Any]]:
    """
    Loads and strictly verifies manifest integrity:
      1. File SHA-256 matches `expected_sha256`
      2. Row count equals `expected_rows`
      3. All rows belong to `expected_split`
      4. Per-row SHA-256 of `canonical_url` matches `canonical_url_sha256`
         (modulo optional frozen `known_hash_exceptions`)
      5. Optional verified SHA selection filtering and deterministic smoke sampling
    """
    actual_sha = sha256_file(manifest_path)
    if actual_sha != expected_sha256:
        raise Abort(
            f"manifest SHA-256 mismatch for {manifest_path.name}: {actual_sha} != {expected_sha256}"
        )

    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows: list[dict[str, Any]] = list(csv.DictReader(f))

    if len(rows) != expected_rows:
        raise Abort(f"manifest row count {len(rows)} != {expected_rows}")

    bad_splits = Counter(r["split"] for r in rows if r["split"] != expected_split)
    if bad_splits:
        raise Abort(f"split pollution: expected only {expected_split}, found {dict(bad_splits)}")

    exceptions = known_hash_exceptions or {}
    for row in rows:
        url_sha = hashlib.sha256(row["canonical_url"].encode("utf-8")).hexdigest()
        if url_sha != row["canonical_url_sha256"]:
            if exceptions.get(row["sample_id"]) != row["canonical_url_sha256"]:
                raise Abort(f"canonical_url_sha256 mismatch for {row['sample_id']}")

    if selection_path is not None:
        if not selection_path.exists():
            raise Abort(f"selection file {selection_path.name} missing or altered")
        if selection_sha256 is not None and sha256_file(selection_path) != selection_sha256:
            raise Abort(f"selection file {selection_path.name} missing or altered")
        keep_shas = set(selection_path.read_text(encoding="utf-8").split())
        rows = [r for r in rows if r["canonical_url_sha256"] in keep_shas]

    if sample_per_cell:
        rng = random.Random(42)
        picked: list[dict[str, Any]] = []
        for dataset_name in sorted({r["dataset"] for r in rows}):
            for label in ("0", "1"):
                cell = sorted(
                    (r for r in rows if r["dataset"] == dataset_name and r["y"] == label),
                    key=lambda r: r["sample_id"],
                )
                picked.extend(rng.sample(cell, sample_per_cell))
        rows = picked

    return rows


def ensure_frozen_config(
    out_dir: Path,
    payload: dict[str, Any],
    *,
    dry_run: bool,
    extra_preconditions: dict[str, str] | None = None,
) -> None:
    """
    Writes `run_config.json` on first run and aborts if an existing file has a different `config_sha256`.
    """
    cfg_path = out_dir / "run_config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        if old.get("config_sha256") != payload["config_sha256"]:
            raise Abort(
                f"{cfg_path} holds a different frozen config; refusing to mix configs in one directory"
            )
        return
    if dry_run:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    to_write = dict(payload)
    to_write["frozen_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    if extra_preconditions is not None:
        to_write["preconditions"] = extra_preconditions
    with open(cfg_path, "x", encoding="utf-8") as f:
        json.dump(to_write, f, indent=2)


def verify_smoke_gate_passed(smoke_summary_path: Path, expected_config_sha256: str) -> None:
    """Aborts unless the prerequisite CAL smoke run passed with the same config hash."""
    if not smoke_summary_path.exists():
        raise Abort("run smoke phase first")
    summary = json.loads(smoke_summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("smoke_gate") != "PASS"
        or summary.get("config_sha256") != expected_config_sha256
    ):
        raise Abort(f"smoke gate not passed with this config: {summary.get('smoke_gate')}")


def post_json(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    timeout_s: int = 60,
    ssl_ctx: ssl.SSLContext = SSL_CTX,
) -> tuple[int, dict[str, Any] | None, str, float | None]:
    """Executes an HTTPS JSON POST and returns `(http_status, json_body, error_detail, retry_after)`."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=ssl_ctx) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), "", None
    except urllib.error.HTTPError as err:
        try:
            detail = err.read().decode("utf-8")[:400]
        except Exception:
            detail = ""
        raw_ra = err.headers.get("retry-after") if err.headers else None
        try:
            retry_after = float(raw_ra) if raw_ra is not None else None
        except ValueError:
            retry_after = None
        return err.code, None, detail, retry_after
    except Exception as exc:
        return -1, None, f"{type(exc).__name__}: {exc}"[:300], None


def percentile(sorted_values: Sequence[float], q: float) -> float | None:
    """Returns empirical quantile from an already-sorted list of floats, or None if empty."""
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, int(q * len(sorted_values)))
    return sorted_values[idx]


def write_summary_and_manifest(out_dir: Path, summary: dict[str, Any]) -> None:
    """Writes `summary.json` and updates `manifest.json` with SHA-256 digests of all output files."""
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    files = sorted(p for p in out_dir.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {
        "files": {p.name: sha256_file(p) for p in files},
        "generated_at_utc": summary["generated_at_utc"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


class BaseHarvester:
    """
    Base class providing shared fail-closed runtime mechanics for v2 harvesters:
      - Config consistency checks on existing `cache.jsonl`
      - Retry/backoff HTTP loop with TLS, auth/billing, and quota abort triggers
      - Server-reported model drift abortion
      - Concurrent ThreadPoolExecutor execution with >5% failure circuit breaker
    """

    def __init__(
        self,
        *,
        out_dir: Path,
        cfg_sha: str,
        api_key: str,
        concurrency: int,
        endpoint_override: str | None = None,
        ssl_ctx: ssl.SSLContext = SSL_CTX,
        timeout_s: int = 60,
    ) -> None:
        self.out = out_dir
        self.cfg_sha = cfg_sha
        self.api_key = api_key
        self.concurrency = concurrency
        self.endpoint_override = endpoint_override
        self.ssl_ctx = ssl_ctx
        self.timeout_s = timeout_s

        self.loader = FrozenPromptLoader(version="v2")
        self.lock = threading.Lock()
        self.abort_reason: str | None = None
        self.server_model: str | None = None
        self.done = 0
        self.failed = 0

        self.cache_path = self.out / "cache.jsonl"
        self.ledger_path = self.out / "ledger.jsonl"
        self.failures_path = self.out / "failures.jsonl"

        for rec in read_jsonl(self.cache_path):
            if rec.get("config_sha256") != self.cfg_sha:
                raise Abort("cache.jsonl contains a record from a different config")
            self.server_model = self.server_model or rec.get("server_model")

    def append_jsonl(self, path: Path, rec: dict[str, Any]) -> None:
        append_jsonl_locked(path, rec, self.lock)

    def check_server_model_drift(self, reported_model: str | None) -> None:
        """Records the first seen server model and aborts if it changes mid-run."""
        with self.lock:
            if reported_model:
                if self.server_model is None:
                    self.server_model = reported_model
                elif reported_model != self.server_model:
                    self.abort_reason = f"model drift: {self.server_model} -> {reported_model}"
        if self.abort_reason:
            raise Abort(self.abort_reason)

    def validate_and_populate_verdict(self, rec: dict[str, Any], raw_text: str) -> str:
        """Runs shared JSON schema & agreement validation, mutating verdict fields on success."""
        obj, violation, parse_status = validate_llm_response(raw_text)
        if obj is not None:
            rec.update(
                {
                    "parsed_verdict": obj.verdict,
                    "parsed_probability": obj.probability,
                    "parsed_signals": obj.signals,
                    "agreement_violation": violation,
                }
            )
        return parse_status

    def post_with_retries(
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        max_retries: int,
        retry_statuses: tuple[int, ...],
        abort_statuses: tuple[int, ...],
        honor_retry_after: bool,
        ledger_builder: Callable[[int, int, dict[str, Any] | None, str, float, dt.datetime], dict[str, Any]],
        insufficient_quota_abort: bool = False,
    ) -> tuple[int, dict[str, Any] | None, str, int, float]:
        """Executes HTTP POST with retries, ledger logging, and fail-closed error gates."""
        status, data, detail, attempt, latency_ms = -1, None, "", 0, 0.0
        target_url = self.endpoint_override or url
        for attempt in range(max_retries + 1):
            if self.abort_reason:
                raise Abort(self.abort_reason)
            t_req = dt.datetime.now(dt.timezone.utc)
            t0 = time.perf_counter()
            status, data, detail, retry_after = post_json(
                target_url, headers, body, timeout_s=self.timeout_s, ssl_ctx=self.ssl_ctx
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.append_jsonl(
                self.ledger_path,
                ledger_builder(attempt, status, data, detail, latency_ms, t_req),
            )
            if status == 200:
                break
            if status == -1 and "CERTIFICATE_VERIFY_FAILED" in detail:
                self.abort_reason = (
                    f"TLS certificate verification failed (CA source: {CA_SOURCE}); "
                    f"run with `--with certifi`, or check for an HTTPS-intercepting proxy/VPN: {detail}"
                )
                raise Abort(self.abort_reason)
            if insufficient_quota_abort and status == 429 and "insufficient_quota" in detail:
                self.abort_reason = (
                    f"HTTP 429 insufficient_quota (account balance/billing, not a rate limit): {detail}"
                )
                raise Abort(self.abort_reason)
            if status in abort_statuses:
                self.abort_reason = f"HTTP {status} (request/auth/credit): {detail}"
                raise Abort(self.abort_reason)
            if status in retry_statuses and attempt < max_retries:
                backoff = min(60.0, 2.0 * (2**attempt))
                wait = retry_after if (honor_retry_after and retry_after is not None) else backoff
                time.sleep(wait + random.uniform(0, 1))
                continue
            break
        return status, data, detail, attempt, latency_ms

    def run_concurrent(
        self,
        todo: list[dict[str, Any]],
        classify_fn: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        fail_rate_abort: float = 0.05,
        log_every: int = 250,
        unit_label: str = "URL/s",
    ) -> None:
        """Executes `classify_fn` concurrently with circuit-breaker monitoring."""
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futures = [ex.submit(classify_fn, row) for row in todo]
            try:
                for fut in cf.as_completed(futures):
                    rec = fut.result()
                    with self.lock:
                        self.done += 1
                        if rec["error_status"] != "success":
                            self.failed += 1
                    target = self.cache_path if rec["error_status"] == "success" else self.failures_path
                    self.append_jsonl(target, rec)
                    if self.done % log_every == 0:
                        rate = self.done / max(1e-9, time.time() - t0)
                        print(
                            f"  {self.done}/{len(todo)} done, failures={self.failed}, {rate:.1f} {unit_label}",
                            flush=True,
                        )
                    if self.done >= 50 and (self.failed / self.done) > fail_rate_abort:
                        self.abort_reason = (
                            f"failure rate {self.failed}/{self.done} > {fail_rate_abort:.0%}"
                        )
                        raise Abort(self.abort_reason)
            except Abort:
                for pending in futures:
                    pending.cancel()
                raise
