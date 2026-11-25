"""
LLM Structured Output Schema & Agreement Constraint Validator (D-19 / P-19).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class LLMResponseSchema(BaseModel):
    verdict: Literal["phishing", "legitimate"]
    probability: float = Field(..., ge=0.0, le=1.0, description="P(phishing)")
    signals: List[str] = Field(default_factory=list)

    @field_validator("signals", mode="before")
    @classmethod
    def validate_signals_length(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        return [s[:60] if isinstance(s, str) else s for s in v[:3]]


def validate_llm_response(raw_text: str) -> Tuple[Optional[LLMResponseSchema], bool, str]:
    """
    Parses and validates LLM raw response.
    Returns:
      (parsed_obj, agreement_violation, error_status)
      where error_status is 'success', 'json_error', or 'schema_fail'.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None, False, "json_error"
        
    cleaned = raw_text.strip()
    # Strip markdown fences if present
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except Exception:
        return None, False, "json_error"

    try:
        obj = LLMResponseSchema.model_validate(data)
    except Exception:
        return None, False, "schema_fail"

    # Check Agreement Constraint:
    # verdict == "phishing" <==> probability >= 0.5
    is_phish = (obj.verdict == "phishing")
    prob_phish = (obj.probability >= 0.5)
    agreement_violation = (is_phish != prob_phish)

    return obj, agreement_violation, "success"
