import numpy as np
import pytest
from src.evaluation.metrics import compute_classification_metrics, compute_taxonomy_counts
from src.evaluation.statistics import holm_bonferroni_correction, exact_mcnemar_test

def test_metrics_and_taxonomy():
    y_true = np.array([1, 1, 0, 0, 1])
    y_spec = np.array([0, 1, 0, 1, 1]) # 1 FN, 1 FP, 3 correct
    y_llm  = np.array([1, 1, 0, 0, 0]) # 1 FN, 4 correct
    
    tax = compute_taxonomy_counts(y_true, y_spec, y_llm)
    assert tax["cell_a_both_correct"] == 2 # (idx 1, 2)
    assert tax["cell_b_rescue"] == 2       # (idx 0: spec 0!=1, llm 1==1; idx 3: spec 1!=0, llm 0==0)
    assert tax["cell_c_regression"] == 1   # (idx 4: spec 1==1, llm 0!=1)
    assert tax["cell_d_both_wrong"] == 0

def test_holm_bonferroni():
    raw_p = [0.01, 0.04, 0.03]
    adj_p = holm_bonferroni_correction(raw_p)
    assert len(adj_p) == 3
    assert adj_p[0] <= adj_p[2] <= adj_p[1]
