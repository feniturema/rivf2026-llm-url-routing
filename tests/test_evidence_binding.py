"""
Comprehensive Machine Evidence-Binding Automated Audit (X-000 § 13 / P-33 / P-34 / P-35).
Asserts that every published benchmark figure matches the exact JSON/CSV artifact on disk.
Fails CI if any narrative drift, transcription error, or threshold leakage occurs.
"""

import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.llm.prompt import FrozenPromptLoader, FROZEN_V1_COMBINED_SHA256, FROZEN_V2_COMBINED_SHA256
from src.llm.ledger import TokenLedger


def test_frozen_prompt_v1_and_v2_hashes():
    p1 = FrozenPromptLoader(version="v1")
    assert p1.combined_hash == FROZEN_V1_COMBINED_SHA256
    
    p2 = FrozenPromptLoader(version="v2")
    assert p2.combined_hash == FROZEN_V2_COMBINED_SHA256


def test_artifact_probe_evidence_binding():
    probe_json = Path("results/artifacts/champion_artifact_probe.json")
    assert probe_json.exists(), "results/artifacts/champion_artifact_probe.json must exist"
    
    with open(probe_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    assert len(df) == 3
    
    for _, row in df.iterrows():
        # Majority class baseline must be 0.50 on balanced TEST split
        assert abs(row["majority_class_baseline"] - 0.50) < 1e-4
        # Bias delta must be exactly |accuracy - 0.5|
        assert abs(row["path_bias_abs_delta"] - abs(row["path_strongest_acc"] - 0.5)) < 1e-6
        assert abs(row["scheme_bias_abs_delta"] - abs(row["scheme_strongest_acc"] - 0.5)) < 1e-6
        # Single-feature accuracy must be >= 0.50
        assert row["path_strongest_acc"] >= 0.50
        assert row["scheme_strongest_acc"] >= 0.50


def test_lineage_matrix_evidence_binding():
    mat_json = Path("results/artifacts/dataset_lineage_matrix.json")
    assert mat_json.exists(), "results/artifacts/dataset_lineage_matrix.json must exist"
    
    with open(mat_json, "r", encoding="utf-8") as f:
        matrix = json.load(f)
        
    # URLPhish2025 is virtually disjoint (< 0.2% overlap)
    assert matrix["PWD2021"]["URLPhish2025"] <= 20
    assert matrix["URLPhish2025"]["EBBU2017"] <= 5
    
    # PWD2021 and EBBU2017 share historical lineage (4,835 URLs in manifests)
    assert matrix["PWD2021"]["EBBU2017"] == 4835


def test_policy_freeze_pre_registration():
    freeze_path = Path("results/policies/policy_freeze.json")
    assert freeze_path.exists(), "results/policies/policy_freeze.json must exist"
    
    with open(freeze_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    meta = data["metadata"]
    assert meta["dev_policy_total_samples"] == 13446
    assert meta["primary_policy"] == "R3_entropy"
    assert meta["primary_budget"] == 0.25
    assert meta["primary_cutoff_25"] > 0.40 # Valid entropy cutoff
    
    # Thresholds must be monotonically non-increasing with increasing budget
    r3_thresh = data["thresholds"]["R3_entropy"]
    budgets = sorted([float(b) for b in r3_thresh.keys()])
    vals = [r3_thresh[str(b)] for b in budgets]
    for i in range(1, len(vals)):
        assert vals[i] <= vals[i - 1], f"Thresholds must decrease with budget: {vals[i]} > {vals[i-1]}"


def test_live_ledger_and_cache_integrity():
    for ledger_path in Path("results/pilot").glob("*_ledger.jsonl"):
        ledger = TokenLedger(ledger_path)
        integ = ledger.validate_integrity(max_allowed_failure_rate_pct=5.0)
        assert integ["valid_live_integrity"] is True, f"Integrity failed for {ledger_path}: {integ['violation_reasons']}"
        assert integ["synthetic_entries"] == 0, f"Found synthetic entries in {ledger_path}"
        assert integ["failure_rate_pct"] <= 5.0, f"Failure rate >5% in {ledger_path}"
        
        # Test Cache parity
        cache_path = ledger_path.parent / ledger_path.name.replace("_ledger.jsonl", "_cache.jsonl")
        if cache_path.exists():
            from src.llm.cache import LLMCache
            cache = LLMCache(cache_path)
            successful_ledger_entries = [e for e in ledger.entries if e.error_status == "success"]
            assert len(cache) == len(successful_ledger_entries), f"Cache count mismatch for {cache_path}"
            
            cache_hashes = cache.get_all_hashes()
            ledger_hashes = set(e.canonical_url_sha256 for e in successful_ledger_entries)
            assert cache_hashes == ledger_hashes, f"Hash set mismatch between {cache_path} and {ledger_path}"
            
            # Check non-empty raw response and valid fields
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    assert len(rec["canonical_url_sha256"]) == 64
                    assert len(rec["raw_response"].strip()) > 0
                    assert rec["prompt_version"] == "v2"
                    assert "model_version" in rec


def test_cohort_manifest_purity_and_freeze():
    freeze_path = Path("results/manifests/cohort_manifest_freeze.json")
    assert freeze_path.exists(), "Cohort manifest freeze file missing!"
    with open(freeze_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    cohorts = meta["cohorts"]
    for fname, info in cohorts.items():
        csv_path = Path(f"data/manifests/{fname}")
        assert csv_path.exists(), f"Cohort CSV {csv_path} missing!"
        
        # Verify SHA-256
        actual_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        assert actual_sha == info["sha256"], f"SHA-256 mismatch for {fname}"
        
        df = pd.read_csv(csv_path)
        assert len(df) == info["total_rows"], f"Row count mismatch for {fname}"
        assert (df["split"] == info["split"]).all(), f"Partition impurity in {fname}! Not 100% {info['split']}"
        assert (df["y"] == 0).sum() == info["y_0_legit"], f"Legit label mismatch in {fname}"
        assert (df["y"] == 1).sum() == info["y_1_phish"], f"Phish label mismatch in {fname}"

    # Verify E3 total row count = 26,886
    assert cohorts["e3_test_all_26886.csv"]["total_rows"] == 26886
    # Verify E4 total row count = 13,446
    assert cohorts["e4_cal_all_13446.csv"]["total_rows"] == 13446
    # Verify E9 cohorts (600 and 3,000)
    assert cohorts["e9_shared_test_cohort_600.csv"]["total_rows"] == 600
    assert cohorts["e9_shared_test_cohort_3000.csv"]["total_rows"] == 3000


def test_runner_partition_pollution_guard():
    import pytest
    from experiments.run_pilot import run_pilot_harvest
    import asyncio
    import tempfile

    # Test that passing a mixed manifest with expected_split='TEST' raises ValueError
    with tempfile.TemporaryDirectory() as tmpdir:
        mixed_csv = Path(tmpdir) / "mixed_manifest.csv"
        df_mixed = pd.DataFrame([
            {"canonical_url": "https://a.com", "sample_id": "1", "canonical_url_sha256": "h1", "dataset": "PWD", "split": "TEST", "y": 0},
            {"canonical_url": "https://b.com", "sample_id": "2", "canonical_url_sha256": "h2", "dataset": "PWD", "split": "TRAIN", "y": 1},
        ])
        df_mixed.to_csv(mixed_csv, index=False)

        with pytest.raises(ValueError, match="Partition pollution detected"):
            asyncio.run(run_pilot_harvest(
                manifest_path=mixed_csv,
                model_name="gemini-flash-lite-latest",
                prompt_version="v2",
                api_key="mock_key",
                cache_path=Path(tmpdir) / "test_cache.jsonl",
                ledger_path=Path(tmpdir) / "test_ledger.jsonl",
                expected_split="TEST",
            ))


def test_runner_sha256_gate_positive_and_negative():
    import pytest
    from experiments.run_pilot import run_pilot_harvest
    from experiments.run_openai_harvest import run_openai_harvest
    import asyncio

    manifest_p = Path("data/manifests/e3_test_pwd2021.csv")
    correct_sha = hashlib.sha256(manifest_p.read_bytes()).hexdigest()
    wrong_sha = "0000000000000000000000000000000000000000000000000000000000000000"

    # 1. Negative test: Gemini runner with wrong SHA-256 raises ValueError
    with pytest.raises(ValueError, match="Manifest SHA-256 mismatch"):
        asyncio.run(run_pilot_harvest(
            manifest_path=manifest_p,
            model_name="gemini-flash-lite-latest",
            prompt_version="v2",
            api_key="mock_key",
            cache_path=Path("results/pilot/test_cache.jsonl"),
            ledger_path=Path("results/pilot/test_ledger.jsonl"),
            expected_manifest_sha256=wrong_sha,
        ))

    # 2. Negative test: OpenAI runner with wrong SHA-256 raises ValueError
    with pytest.raises(ValueError, match="Manifest SHA-256 mismatch"):
        asyncio.run(run_openai_harvest(
            manifest_path=manifest_p,
            model_name="gpt-5.6-luna",
            prompt_version="v2",
            api_key="mock_key",
            base_url="http://mock",
            cache_path=Path("results/pilot/test_cache.jsonl"),
            ledger_path=Path("results/pilot/test_ledger.jsonl"),
            expected_manifest_sha256=wrong_sha,
        ))


def test_cache_contextualization_prevents_row_loss():
    from src.llm.cache import LLMCache
    from src.llm.client import GeminiAsyncClient
    import asyncio
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "test_cache.jsonl"
        cache = LLMCache(cache_file)
        
        # Pre-seed cache with a response under sample_id = "PWD_1"
        url = "https://example.com/test"
        url_sha = hashlib.sha256(url.encode()).hexdigest()
        cache.put({
            "canonical_url_sha256": url_sha,
            "sample_id": "PWD_1",
            "dataset": "PWD",
            "model": "gemini-flash-lite-latest",
            "prompt_version": "v2",
            "repeat_index": 0,
            "raw_response": '{"verdict": "legitimate", "probability": 0.05, "signals": []}',
            "parsed_verdict": "legitimate",
            "parsed_probability": 0.05,
            "parsed_signals": [],
            "agreement_violation": False,
            "error_status": "success",
        })

        client = GeminiAsyncClient(api_key="mock", model_name="gemini-flash-lite-latest", prompt_version="v2", cache=cache)
        
        # Query same URL with a different sample_id = "EBBU_2"
        rec = asyncio.run(client.classify_url(
            canonical_url=url,
            sample_id="EBBU_2",
            dataset="EBBU",
            stratum="S-STRICT",
            split="TEST",
        ))

        # Must be contextualized to EBBU_2, NOT stale PWD_1
        assert rec["sample_id"] == "EBBU_2"
        assert rec["dataset"] == "EBBU"


def test_end_to_end_evaluate_pilot_k2_metrics_export():
    from experiments.run_pilot import evaluate_pilot_k2
    from src.llm.ledger import TokenLedger
    import tempfile

    smoke_ledger_p = Path("results/pilot/gemini_smoke_ledger.jsonl")
    if not smoke_ledger_p.exists():
        smoke_ledger_p = Path("results/pilot/champion_pilot_v2_ledger.jsonl")

    ledger = TokenLedger(smoke_ledger_p)
    manifest_p = Path("data/manifests/e3_test_pwd2021.csv")
    df_manifest = pd.read_csv(manifest_p).head(15).copy()

    # Synthetic merged dataframe representing harvest results
    records = []
    for _, row in df_manifest.iterrows():
        records.append({
            "sample_id": row["sample_id"],
            "canonical_url_sha256": row["canonical_url_sha256"],
            "model": "gemini-flash-lite-latest",
            "model_version": "gemini-3.5-flash-lite",
            "prompt_version": "v2",
            "prompt_combined_sha256": "mock_prompt_sha",
            "parsed_verdict": "legitimate",
            "parsed_probability": 0.05,
            "error_status": "success",
        })
    df_res = pd.DataFrame(records)
    merged = df_manifest.merge(df_res, on=["sample_id", "canonical_url_sha256"], how="inner")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_metrics = Path(tmpdir) / "test_metrics.json"
        # Must execute without NameError and write JSON
        evaluate_pilot_k2(merged, ledger, out_report_path=out_metrics)
        assert out_metrics.exists()
        
        with open(out_metrics, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "k2_status" in data["metadata"]
        assert "indeterminate" in data["metadata"]["k2_status"] or "passed" in data["metadata"]["k2_status"]
        assert "two_by_two" in data
        assert "bootstrap_ci_delta_f1" in data


def test_production_experiments_freeze():
    cfg_p = Path("configs/production_experiments_freeze.json")
    assert cfg_p.exists(), "Production experiments freeze missing!"
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    for exp_id, exp_meta in cfg["experiments"].items():
        m_path = Path(exp_meta["manifest"])
        assert m_path.exists(), f"Manifest missing for {exp_id}: {m_path}"
        actual_sha = hashlib.sha256(m_path.read_bytes()).hexdigest()
        assert actual_sha == exp_meta["manifest_sha256"], f"SHA mismatch for {exp_id}"
        
        df = pd.read_csv(m_path)
        assert len(df) == exp_meta["total_samples"], f"Sample count mismatch for {exp_id}"
        assert (df["split"] == exp_meta["expected_split"]).all(), f"Split mismatch for {exp_id}"


def test_e5_frontier_evidence_binding():
    e5_p = Path("results/E5/e5_frontier.json")
    assert e5_p.exists(), "results/E5/e5_frontier.json must exist"
    with open(e5_p, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for r_name in ["R3_entropy", "R2_margin", "R1_max_prob", "R0_random"]:
        assert r_name in data, f"Missing router {r_name} in E5 frontier"
        
    r3 = data["R3_entropy"]
    assert len(r3) == 17
    
    # Baseline at b=0.0
    assert r3[0]["nominal_budget"] == 0.0
    assert r3[0]["escalated_count"] == 0
    assert abs(r3[0]["cascade_f1"] - 0.9537238) < 1e-4
    assert r3[0]["delta_f1"] == 0.0
    
    # Full escalation at b=1.0
    assert r3[-1]["nominal_budget"] == 1.0
    assert r3[-1]["escalated_count"] == 26886
    assert r3[-1]["rescues_b"] == 964
    assert r3[-1]["regressions_c"] == 1553
    assert r3[-1]["net_rescues"] == -589
    assert r3[-1]["delta_f1"] < 0.0  # Must be negative (regression)
    
    # NRR definition consistency check
    for entry in r3:
        n_esc = entry["escalated_count"]
        net = entry["net_rescues"]
        if n_esc > 0:
            assert abs(entry["escalated_nrr"] - (net / n_esc)) < 1e-6
        assert abs(entry["population_nrr"] - (net / 26886)) < 1e-6


def test_e3_dataset_breakdown_evidence_binding():
    d_path = Path("results/E3/e3_dataset_breakdown.json")
    assert d_path.exists(), "results/E3/e3_dataset_breakdown.json must exist"
    with open(d_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for dname in ["EBBU2017", "PWD2021", "URLPhish2025"]:
        assert dname in data, f"Missing dataset {dname}"
        d_info = data[dname]
        # In every dataset, Top 25% routed zone must be strictly positive
        assert d_info["top_25_uncertainty_zone"]["net_advantage"] > 0, f"Top 25% zone must be positive for {dname}"
        # In every dataset, Bot 75% intercepted zone must be strictly negative
        assert d_info["bot_75_certainty_zone"]["net_advantage"] < 0, f"Bot 75% zone must be negative for {dname}"


def test_e9_partition_evidence_binding():
    e9_p = Path("results/E9/e9_openai_luna_partition_and_frontier.json")
    assert e9_p.exists(), "results/E9/e9_openai_luna_partition_and_frontier.json must exist"
    with open(e9_p, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["metadata"]["n_samples"] == 3000
    assert data["metadata"]["model"] == "gpt-5.6-luna"
    
    two_zone = data["two_zone_advantage_partition"]
    assert two_zone["top_25_uncertainty_zone"]["net_advantage"] > 0
    assert two_zone["bot_75_certainty_zone"]["net_advantage"] < 0
    assert data["full_escalation_2x2"]["net_rescues"] < 0


def test_breakeven_inequality_theorem():
    # Proof verification: net > 0 <=> eps_error_on_correct < rho_rescue * (E / (N - E))
    N = 10000
    E = 500  # specialist accuracy = 95%
    N_minus_E = N - E
    
    rho_rescue = 0.80  # LLM rescues 80% of errors -> B = 400
    B = int(rho_rescue * E)
    
    # Breakeven epsilon
    eps_breakeven = rho_rescue * (E / N_minus_E)  # 0.80 * (500 / 9500) = 0.042105 (4.21%)
    
    # Case 1: measured epsilon = 3% (< 4.21%) -> Net should be positive
    eps_1 = 0.03
    C_1 = int(eps_1 * N_minus_E)
    net_1 = B - C_1
    assert net_1 > 0, "Net must be positive when eps < eps_breakeven"
    
    # Case 2: measured epsilon = 6% (> 4.21%) -> Net should be negative
    eps_2 = 0.06
    C_2 = int(eps_2 * N_minus_E)
    net_2 = B - C_2
    assert net_2 < 0, "Net must be negative when eps > eps_breakeven"
