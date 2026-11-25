"""
Pure Offline Replay Engine (D-12).
NO network imports. Evaluates cascade predictions deterministically from cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, precision_score, recall_score


@dataclass
class CascadeResult:
    policy: str
    budget_nominal: float
    escalation_rate_realised: float
    n_escalated: int
    cascade_f1: float
    cascade_mcc: float
    specialist_f1: float
    llm_f1: float
    delta_f1: float
    rescues_cell_b: int
    regressions_cell_c: int
    wasted_calls_cell_a: int
    unsolvable_cell_d: int
    total_available_rescues: int
    rescue_precision: float
    rescue_recall: float
    net_rescue_rate_nrr: float
    escalation_efficiency_ee: float


def replay_cascade(
    y_true: np.ndarray,
    y_spec: np.ndarray,
    y_llm: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    policy_name: str = "R4_calibrated_conf",
    budget_nominal: float = 0.0,
    oracle_f1: Optional[float] = None,
    *,
    strict: bool = False,
) -> CascadeResult:
    """
    Pure function: calculates cascade metrics given predictions and threshold.
    """
    n = len(y_true)
    escalate_mask = (scores > threshold) if strict else (scores >= threshold)
    
    # Replacement integration: y_cascade = y_llm if escalate else y_spec
    y_cascade = np.where(escalate_mask, y_llm, y_spec)
    
    # 2x2 Taxonomy
    spec_correct = (y_spec == y_true)
    llm_correct = (y_llm == y_true)
    
    cell_a = spec_correct & llm_correct    # Both correct (wasted call)
    cell_b = (~spec_correct) & llm_correct # Specialist wrong, LLM correct (Rescue)
    cell_c = spec_correct & (~llm_correct) # Specialist correct, LLM wrong (Regression / Harm)
    cell_d = (~spec_correct) & (~llm_correct) # Both wrong
    
    n_escalated = int(np.sum(escalate_mask))
    realised_rate = float(n_escalated / n) if n > 0 else 0.0
    
    n_rescues = int(np.sum(escalate_mask & cell_b))
    n_regressions = int(np.sum(escalate_mask & cell_c))
    n_waste_a = int(np.sum(escalate_mask & cell_a))
    n_waste_d = int(np.sum(escalate_mask & cell_d))
    
    spec_f1 = float(f1_score(y_true, y_spec, zero_division=0))
    llm_f1 = float(f1_score(y_true, y_llm, zero_division=0))
    cascade_f1 = float(f1_score(y_true, y_cascade, zero_division=0))
    cascade_mcc = float(matthews_corrcoef(y_true, y_cascade))
    
    total_rescues = int(np.sum(cell_b))
    rescue_prec = float(n_rescues / n_escalated) if n_escalated > 0 else float("nan")
    rescue_rec = float(n_rescues / total_rescues) if total_rescues > 0 else float("nan")
    nrr = float((n_rescues - n_regressions) / n_escalated) if n_escalated > 0 else float("nan")
    
    # Escalation Efficiency: EE(pi, b) = (Q_pi - Q_spec) / (Q_oracle - Q_spec)
    if oracle_f1 is not None and abs(oracle_f1 - spec_f1) > 1e-6:
        ee = float((cascade_f1 - spec_f1) / (oracle_f1 - spec_f1))
    else:
        ee = float("nan")

    return CascadeResult(
        policy=policy_name,
        budget_nominal=budget_nominal,
        escalation_rate_realised=realised_rate,
        n_escalated=n_escalated,
        cascade_f1=cascade_f1,
        cascade_mcc=cascade_mcc,
        specialist_f1=spec_f1,
        llm_f1=llm_f1,
        delta_f1=cascade_f1 - spec_f1,
        rescues_cell_b=n_rescues,
        regressions_cell_c=n_regressions,
        wasted_calls_cell_a=n_waste_a,
        unsolvable_cell_d=n_waste_d,
        total_available_rescues=total_rescues,
        rescue_precision=rescue_prec,
        rescue_recall=rescue_rec,
        net_rescue_rate_nrr=nrr,
        escalation_efficiency_ee=ee,
    )
