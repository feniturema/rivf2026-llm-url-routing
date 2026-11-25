"""Evaluation metrics and statistical verification."""
from .metrics import compute_classification_metrics, compute_taxonomy_counts
from .statistics import bootstrap_confidence_interval, holm_bonferroni_correction, exact_mcnemar_test

__all__ = [
    "compute_classification_metrics",
    "compute_taxonomy_counts",
    "bootstrap_confidence_interval",
    "holm_bonferroni_correction",
    "exact_mcnemar_test",
]
