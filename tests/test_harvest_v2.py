"""Offline unit, dry-run, config-seal, and local-HTTP mock tests for v2 harvesters."""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from experiments import run_anthropic_harvest_v2, run_deepseek_harvest_v2, run_harvest_v2
from src.llm.harvest_common import Abort, read_jsonl
from src.v2.io import RESULTS_V2_DIR, read_json


def _sample_rows(n: int = 2) -> list[dict[str, Any]]:
    rows = []
    for i in range(n):
        url = f"https://example-phish-{i}.test/login"
        sha = hashlib.sha256(url.encode("utf-8")).hexdigest()
        rows.append(
            {
                "sample_id": f"MOCK_{i}",
                "canonical_url": url,
                "canonical_url_sha256": sha,
                "dataset": "MOCK",
                "split": "CAL_ROUTER",
                "manifest_hash": "mock_manifest",
                "y": str(i % 2),
                "repeat_index": 0,
            }
        )
    return rows


def test_harvester_config_sha256_matches_frozen_run_configs() -> None:
    mapping = {
        RESULTS_V2_DIR / "BRIDGE" / "gemini-3.5-flash-lite" / "run_config.json": run_harvest_v2.config_sha("gemini", "bridge"),
        RESULTS_V2_DIR / "E3_BACKFILL" / "gemini-3.5-flash-lite" / "run_config.json": run_harvest_v2.config_sha("gemini", "e3_backfill"),
        RESULTS_V2_DIR / "E8" / "gemini-3.5-flash-lite" / "run_config.json": run_harvest_v2.config_sha("gemini", "e8"),
        RESULTS_V2_DIR / "SMOKE" / "gemini-3.5-flash-lite" / "run_config.json": run_harvest_v2.config_sha("gemini", "smoke"),
        RESULTS_V2_DIR / "SMOKE" / "gpt-5.6-luna" / "run_config.json": run_harvest_v2.config_sha("luna", "smoke"),
        RESULTS_V2_DIR / "E9" / "gpt-5.6-luna" / "run_config.json": run_harvest_v2.config_sha("luna", "e9"),
        RESULTS_V2_DIR / "SMOKE" / "gpt-6-luna" / "run_config.json": run_harvest_v2.config_sha("luna6", "smoke"),
        RESULTS_V2_DIR / "E9" / "gpt-6-luna" / "run_config.json": run_harvest_v2.config_sha("luna6", "e9"),
        RESULTS_V2_DIR / "SMOKE" / "deepseek-flash" / "run_config.json": run_deepseek_harvest_v2.config_sha256(),
        RESULTS_V2_DIR / "E9" / "deepseek-flash" / "run_config.json": run_deepseek_harvest_v2.config_sha256(),
        RESULTS_V2_DIR / "SMOKE" / "claude-haiku-5-5" / "run_config.json": run_anthropic_harvest_v2.config_sha256(),
        RESULTS_V2_DIR / "E9" / "claude-haiku-5-5" / "run_config.json": run_anthropic_harvest_v2.config_sha256(),
    }
    for cfg_path, computed in mapping.items():
        frozen = read_json(cfg_path)["config_sha256"]
        assert computed == frozen, f"config_sha256 drift for {cfg_path}"


@pytest.mark.parametrize(
    ("vendor", "phase"),
    [
        ("gemini", "smoke"),
        ("gemini", "bridge"),
        ("gemini", "e3_backfill"),
        ("gemini", "e8"),
        ("luna", "smoke"),
        ("luna", "e9"),
        ("luna6", "smoke"),
        ("luna6", "e9"),
    ],
)
def test_primary_harvester_dry_run_all_phases(vendor: str, phase: str) -> None:
    rc = run_harvest_v2.main(["--vendor", vendor, "--phase", phase, "--dry-run"])
    assert rc == 0


@pytest.mark.parametrize("phase", ["smoke", "e9"])
def test_vendor_harvesters_dry_run_all_phases(phase: str) -> None:
    assert run_deepseek_harvest_v2.main(["--phase", phase, "--dry-run"]) == 0
    assert run_anthropic_harvest_v2.main(["--phase", phase, "--dry-run"]) == 0


def test_harvest_mock_server_retry_resume_drift_and_auth_abort(tmp_path: Path) -> None:
    lock = threading.Lock()
    state: dict[str, Any] = {"mode": "retry_once_then_ok", "calls": 0}
    verdict_json = json.dumps(
        {"verdict": "phishing", "probability": 0.91, "signals": ["test brand signal"]}
    )

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            with lock:
                state["calls"] += 1
                call_num = state["calls"]
                mode = state["mode"]

            if mode == "auth_401":
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b'{"error": "invalid api key"}')
                return

            if mode == "retry_once_then_ok" and call_num == 1:
                self.send_response(429)
                self.send_header("retry-after", "0")
                self.end_headers()
                self.wfile.write(b'{"error": "rate limit"}')
                return

            if mode == "drift" and call_num >= 2:
                model_name = "gemini-drifted-version"
            else:
                model_name = "gemini-3.5-flash-lite"

            if self.path.endswith("/anthropic"):
                payload = {
                    "id": f"msg_{call_num}",
                    "model": "claude-haiku-5-5",
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": verdict_json}],
                    "usage": {"input_tokens": 120, "output_tokens": 25},
                }
            elif self.path.endswith("/deepseek"):
                payload = {
                    "id": f"ds_{call_num}",
                    "model": "deepseek-flash",
                    "system_fingerprint": "fp_mock",
                    "choices": [{"finish_reason": "stop", "message": {"content": verdict_json}}],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 25},
                }
            else:
                payload = {
                    "modelVersion": model_name,
                    "responseId": f"resp_{call_num}",
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {"parts": [{"text": verdict_json}]},
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 120,
                        "candidatesTokenCount": 25,
                        "thoughtsTokenCount": 0,
                    },
                }

            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        rows = _sample_rows(2)
        base = f"http://127.0.0.1:{port}"

        # 1. Retry on 429 + full completion + resume skips completed rows
        out_root = tmp_path / "gemini_test"
        h1 = run_harvest_v2.Harvester(
            "gemini",
            "smoke",
            api_key="mock_key",
            concurrency=1,
            out_root=out_root,
            endpoint_override=base + "/gemini",
        )
        summary = h1.run(rows)
        assert summary["unique_success"] == 2
        assert summary["http_attempts"] == 3  # 1x 429 + 2x 200
        assert summary["smoke_gate"] == "PASS"

        state["calls"] = 0
        h1_resume = run_harvest_v2.Harvester(
            "gemini",
            "smoke",
            api_key="mock_key",
            concurrency=1,
            out_root=out_root,
            endpoint_override=base + "/gemini",
        )
        summary_resume = h1_resume.run(rows)
        assert summary_resume["unique_success"] == 2
        assert state["calls"] == 0

        # 2. Verify DeepSeek & Anthropic harvester classes against mock server
        ds_out = tmp_path / "deepseek_smoke"
        h_ds = run_deepseek_harvest_v2.Harvester(
            "smoke",
            api_key="mock_key",
            concurrency=1,
            out_dir_override=ds_out,
            endpoint_override=base + "/deepseek",
        )
        state["mode"] = "ok"
        ds_sum = h_ds.run(rows[:1])
        assert ds_sum["unique_success"] == 1
        assert ds_sum["smoke_gate"] == "PASS"

        an_out = tmp_path / "anthropic_smoke"
        h_an = run_anthropic_harvest_v2.Harvester(
            "smoke",
            api_key="mock_key",
            concurrency=1,
            out_dir_override=an_out,
            endpoint_override=base + "/anthropic",
        )
        an_sum = h_an.run(rows[:1])
        assert an_sum["unique_success"] == 1
        assert an_sum["smoke_gate"] == "PASS"

        # 3. Fail-closed model drift abort
        drift_root = tmp_path / "gemini_drift"
        h_drift = run_harvest_v2.Harvester(
            "gemini",
            "smoke",
            api_key="mock_key",
            concurrency=1,
            out_root=drift_root,
            endpoint_override=base + "/gemini",
        )
        state["mode"] = "drift"
        state["calls"] = 0
        with pytest.raises(Abort, match="model drift"):
            h_drift.run(rows)

        # 4. Fail-closed HTTP 401 abort
        auth_root = tmp_path / "gemini_auth"
        h_auth = run_harvest_v2.Harvester(
            "gemini",
            "smoke",
            api_key="bad_key",
            concurrency=1,
            out_root=auth_root,
            endpoint_override=base + "/gemini",
        )
        state["mode"] = "auth_401"
        with pytest.raises(Abort, match="HTTP 401"):
            h_auth.run(rows[:1])
        ledger = read_jsonl(h_auth.ledger_path)
        assert len(ledger) == 1
        assert ledger[0]["http_status"] == 401
    finally:
        server.shutdown()
        server.server_close()
