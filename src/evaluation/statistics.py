"""
Statistical Significance & Pre-Registered Contrasts (D-14).
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
from scipy.stats import binomtest


def bootstrap_confidence_interval(
    data: np.ndarray,
    stat_func=np.mean,
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Computes deterministic percentile bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    n = len(data)
    boot_stats = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        boot_stats[i] = stat_func(data[indices])
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_stats, 100.0 * alpha))
    high = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha)))
    return low, high


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Applies step-down Holm-Bonferroni correction to a family of hypotheses.
    Returns adjusted p-values in original order.
    """
    m = len(p_values)
    if m == 0:
        return []
    
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        adj_p = p_val * (m - rank)
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = min(1.0, running_max)
        
    return adjusted


def exact_mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> float:
    """Exact two-tailed McNemar test for paired binary predictions."""
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    b = int(np.sum(correct_a & (~correct_b)))  # A right, B wrong
    c = int(np.sum((~correct_a) & correct_b))  # A wrong, B right
    n = b + c
    if n == 0:
        return 1.0
    res = binomtest(min(b, c), n, 0.5, alternative="two-sided")
    return float(res.pvalue)
