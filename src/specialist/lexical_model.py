"""
Secondary Specialist Classifier: XGBoost on Lexical URL features.
"""

from __future__ import annotations

import time
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import xgboost as xgb


def extract_lexical_features(urls: pd.Series | list[str]) -> np.ndarray:
    feats = []
    for u in urls:
        if not isinstance(u, str):
            u = str(u) if pd.notna(u) else ""
        u_len = len(u)
        n_digits = sum(c.isdigit() for c in u)
        n_letters = sum(c.isalpha() for c in u)
        n_hyphens = u.count("-")
        n_dots = u.count(".")
        
        # Character entropy
        if u_len > 0:
            counts = pd.Series(list(u)).value_counts().values
            probs = counts / u_len
            ent = float(-np.sum(probs * np.log2(probs)))
        else:
            ent = 0.0
            
        p = urlparse(u)
        host = p.hostname or ""
        subdomain_depth = max(0, host.count("."))
        
        feats.append([
            float(u_len),
            float(n_digits / max(1, u_len)),
            float(n_letters / max(1, u_len)),
            float(n_hyphens),
            float(n_dots),
            float(subdomain_depth),
            float(ent)
        ])
    return np.array(feats, dtype=np.float32)


class LexicalXGBoostSpecialist:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        seed: int = 42,
    ):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
        )
        self.is_fitted = False
        self.fit_time_s = 0.0

    def fit(self, train_urls: pd.Series | list[str], y_train: np.ndarray | list[int]) -> LexicalXGBoostSpecialist:
        t0 = time.perf_counter()
        X_train = extract_lexical_features(train_urls)
        self.model.fit(X_train, np.asarray(y_train))
        self.fit_time_s = time.perf_counter() - t0
        self.is_fitted = True
        return self

    def predict_proba(self, urls: pd.Series | list[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X = extract_lexical_features(urls)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, urls: pd.Series | list[str], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(urls)
        return (probs >= threshold).astype(int)
