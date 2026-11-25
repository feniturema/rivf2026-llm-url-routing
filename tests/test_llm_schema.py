import json

from src.llm.prompt import FrozenPromptLoader, FROZEN_V1_COMBINED_SHA256, FROZEN_V2_COMBINED_SHA256
from src.llm.schema import validate_llm_response

def test_prompt_hash_integrity():
    loader_v1 = FrozenPromptLoader(version="v1")
    assert loader_v1.combined_hash == FROZEN_V1_COMBINED_SHA256
    loader_v2 = FrozenPromptLoader(version="v2")
    assert loader_v2.combined_hash == FROZEN_V2_COMBINED_SHA256
    rendered = loader_v2.render_user_prompt("https://test.com")
    assert "https://test.com" in rendered

def test_schema_validation_success():
    raw = '{"verdict": "phishing", "probability": 0.85, "signals": ["suspicious TLD"]}'
    obj, viol, status = validate_llm_response(raw)
    assert status == "success"
    assert obj.verdict == "phishing"
    assert obj.probability == 0.85
    assert viol is False

def test_schema_agreement_violation():
    # Verdict is phishing but probability is 0.10 -> violation
    raw = '{"verdict": "phishing", "probability": 0.10, "signals": []}'
    obj, viol, status = validate_llm_response(raw)
    assert status == "success"
    assert viol is True

def test_schema_failure():
    raw = '{"invalid_field": 123}'
    obj, viol, status = validate_llm_response(raw)
    assert status == "schema_fail"
    assert obj is None


def test_schema_four_signals_truncated_to_three():
    raw = json.dumps(
        {
            "verdict": "phishing",
            "probability": 0.91,
            "signals": ["s" * 75, "sig2", "sig3", "sig4"],
        }
    )
    obj, viol, status = validate_llm_response(raw)
    assert status == "success"
    assert viol is False
    assert obj is not None
    assert len(obj.signals) == 3
    assert obj.signals == ["s" * 60, "sig2", "sig3"]

    _, _, bad_status = validate_llm_response(
        '{"verdict": "phishing", "probability": 0.91, "signals": "not-a-list"}'
    )
    assert bad_status == "schema_fail"
