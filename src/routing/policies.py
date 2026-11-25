"""
Routing Policies R0..R6 + Oracle Bound (D-08 / experiment-design § 5).
Scorers return s(x) where higher = more urgent to escalate.
"""

from __future__ import annotations

from typing import Literal, Optional
import numpy as np


PolicyType = Literal[
    "R0_random",
    "R1_max_prob",
    "R2_margin",
    "R3_entropy",
    "R4_calibrated_conf",
    "R5_conformal",
    "R6_disagreement",
    "R_oracle",
]


def compute_routing_scores(
    policy: PolicyType,
    uncal_probs: np.ndarray,
    cal_probs: Optional[np.ndarray] = None,
    ensemble_probs: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None,
    y_spec: Optional[np.ndarray] = None,
    y_llm: Optional[np.ndarray] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Computes continuous urgency score s(x) for policy.
    Higher score indicates higher priority for escalation to LLM tier.
    """
    n = len(uncal_probs)
    
    if policy == "R0_random":
        return np.random.RandomState(seed).uniform(0.0, 1.0, n)
        
    elif policy == "R1_max_prob":
        # 1 - max(p, 1-p)
        return 1.0 - np.maximum(uncal_probs, 1.0 - uncal_probs)
        
    elif policy == "R2_margin":
        # - |p - (1-p)| = - |2p - 1|
        probs = cal_probs if cal_probs is not None else uncal_probs
        return - np.abs(2.0 * probs - 1.0)
        
    elif policy == "R3_entropy":
        probs = cal_probs if cal_probs is not None else uncal_probs
        p = np.clip(probs, 1e-12, 1.0 - 1e-12)
        return - (p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
        
    elif policy == "R4_calibrated_conf":
        probs = cal_probs if cal_probs is not None else uncal_probs
        return 1.0 - np.maximum(probs, 1.0 - probs)
        
    elif policy == "R5_conformal":
        probs = cal_probs if cal_probs is not None else uncal_probs
        # Non-conformity score: 1 - p(y_pred)
        return 1.0 - np.maximum(probs, 1.0 - probs)
        
    elif policy == "R6_disagreement":
        if ensemble_probs is None:
            raise ValueError("Ensemble probabilities required for R6.")
        # Variance across seeds
        return np.var(ensemble_probs, axis=1)
        
    elif policy == "R_oracle":
        if y_true is None or y_spec is None or y_llm is None:
            raise ValueError("Labels required for Oracle computation.")
        # Oracle value: +1 for rescue (spec wrong, LLM right), -1 for regression (spec right, LLM wrong), 0 otherwise
        is_rescue = (y_spec != y_true) & (y_llm == y_true)
        is_regression = (y_spec == y_true) & (y_llm != y_true)
        
        oracle_score = np.zeros(n, dtype=np.float32)
        oracle_score[is_rescue] = 1.0
        oracle_score[is_regression] = -1.0
        return oracle_score
        
    else:
        raise ValueError(f"Unknown policy: {policy}")


def derive_budget_quantiles(scores_cal: np.ndarray, budgets: list[float]) -> dict[float, float]:
    """
    Fits quantile cutoff thresholds on CAL partition for matched budgets b in [0, 1].
    escalate(x) = 1 iff score >= Quantile_{1-b}(scores_cal).
    """
    thresholds = {}
    for b in budgets:
        if b <= 0.0:
            thresholds[b] = float("inf")
        elif b >= 1.0:
            thresholds[b] = -float("inf")
        else:
            thresholds[b] = float(np.quantile(scores_cal, 1.0 - b))
    return thresholds


def select_escalation_mask(
    scores: np.ndarray,
    threshold: float,
    *,
    strict: bool = False,
) -> np.ndarray:
    """Returns boolean array of escalated samples."""
    return (scores > threshold) if strict else (scores >= threshold)
