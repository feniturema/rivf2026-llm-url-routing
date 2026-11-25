import numpy as np
import pytest
from src.replay.engine import replay_cascade

def test_pure_replay_deterministic():
    n = 100
    y_true = np.random.RandomState(42).randint(0, 2, n)
    y_spec = np.random.RandomState(43).randint(0, 2, n)
    y_llm = np.random.RandomState(44).randint(0, 2, n)
    scores = np.random.RandomState(45).uniform(0, 1, n)
    
    r1 = replay_cascade(y_true, y_spec, y_llm, scores, threshold=0.5, budget_nominal=0.2)
    r2 = replay_cascade(y_true, y_spec, y_llm, scores, threshold=0.5, budget_nominal=0.2)
    
    assert r1.cascade_f1 == r2.cascade_f1
    assert r1.net_rescue_rate_nrr == r2.net_rescue_rate_nrr
    assert r1.rescues_cell_b == r2.rescues_cell_b
