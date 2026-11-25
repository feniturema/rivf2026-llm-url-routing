"""
Core Evaluation Metrics for RIVF 2026.
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, confusion_matrix


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    return {
        "f1": f1,
        "mcc": mcc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
    }


def compute_taxonomy_counts(y_true: np.ndarray, y_spec: np.ndarray, y_llm: np.ndarray) -> Dict[str, int]:
    spec_right = (y_spec == y_true)
    llm_right = (y_llm == y_true)
    return {
        "cell_a_both_correct": int(np.sum(spec_right & llm_right)),
        "cell_b_rescue": int(np.sum((~spec_right) & llm_right)),
        "cell_c_regression": int(np.sum(spec_right & (~llm_right))),
        "cell_d_both_wrong": int(np.sum((~spec_right) & (~llm_right))),
    }
