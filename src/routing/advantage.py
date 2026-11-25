"""
Learned Advantage Router R7 (D-08, D-16).
Estimates expected net benefit A = B - C in {-1, 0, +1},
where B = 1{spec wrong ∧ LLM correct} (Rescue)
      C = 1{spec correct ∧ LLM wrong} (Regression).
Trained on CAL_ROUTER partition with cached LLM labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge


class LearnedAdvantageRouter:
    def __init__(self, mode: str = "regression", seed: int = 42):
        self.mode = mode
        if mode == "regression":
            self.model = Ridge(alpha=1.0, random_state=seed)
        else:
            self.model = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
        self.is_fitted = False

    def _extract_features(
        self,
        specialist_probs: np.ndarray,
        url_lengths: np.ndarray,
        digit_ratios: np.ndarray,
        hyphen_counts: np.ndarray,
        entropy: np.ndarray,
    ) -> np.ndarray:
        """Constructs cheap routing feature matrix."""
        p = specialist_probs.reshape(-1, 1)
        margin = np.abs(2.0 * specialist_probs - 1.0).reshape(-1, 1)
        ent = entropy.reshape(-1, 1)
        u_len = url_lengths.reshape(-1, 1)
        d_rat = digit_ratios.reshape(-1, 1)
        h_cnt = hyphen_counts.reshape(-1, 1)
        
        return np.hstack([p, margin, ent, u_len, d_rat, h_cnt])

    def fit(
        self,
        specialist_probs: np.ndarray,
        specialist_preds: np.ndarray,
        llm_preds: np.ndarray,
        y_true: np.ndarray,
        url_lengths: np.ndarray,
        digit_ratios: np.ndarray,
        hyphen_counts: np.ndarray,
        entropy: np.ndarray,
    ) -> LearnedAdvantageRouter:
        """Fits router on CAL_ROUTER partition."""
        X = self._extract_features(specialist_probs, url_lengths, digit_ratios, hyphen_counts, entropy)
        
        # Net advantage target: A = B - C
        is_rescue = (specialist_preds != y_true) & (llm_preds == y_true)
        is_regression = (specialist_preds == y_true) & (llm_preds != y_true)
        
        target_a = np.zeros(len(y_true), dtype=np.float32)
        target_a[is_rescue] = 1.0
        target_a[is_regression] = -1.0

        if self.mode == "regression":
            self.model.fit(X, target_a)
        else:
            # Binary approximation: P(Rescue)
            self.model.fit(X, is_rescue.astype(int))
            
        self.is_fitted = True
        return self

    def score(
        self,
        specialist_probs: np.ndarray,
        url_lengths: np.ndarray,
        digit_ratios: np.ndarray,
        hyphen_counts: np.ndarray,
        entropy: np.ndarray,
    ) -> np.ndarray:
        """Returns predicted net benefit score for escalation ranking."""
        if not self.is_fitted:
            raise RuntimeError("Advantage router is not fitted.")
        X = self._extract_features(specialist_probs, url_lengths, digit_ratios, hyphen_counts, entropy)
        if self.mode == "regression":
            return self.model.predict(X)
        else:
            return self.model.predict_proba(X)[:, 1]
