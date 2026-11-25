"""
Primary Specialist Classifier: Character n-gram TF-IDF + Logistic Regression.
Inductive serving, microsecond latency (~50 µs on CPU).
"""

from __future__ import annotations

import time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class CharTFIDFSpecialist:
    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 5),
        min_df: int = 3,
        max_features: int = 50000,
        C: float = 1.0,
        max_iter: int = 500,
        seed: int = 42,
    ):
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
        )
        self.classifier = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
        )
        self.is_fitted = False
        self.fit_time_s = 0.0

    def fit(self, train_urls: pd.Series | list[str], y_train: np.ndarray | list[int]) -> CharTFIDFSpecialist:
        t0 = time.perf_counter()
        X_train = self.vectorizer.fit_transform(train_urls)
        self.classifier.fit(X_train, np.asarray(y_train))
        self.fit_time_s = time.perf_counter() - t0
        self.is_fitted = True
        return self

    def predict_proba(self, urls: pd.Series | list[str]) -> np.ndarray:
        """Returns 1D array of P(phishing)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X = self.vectorizer.transform(urls)
        return self.classifier.predict_proba(X)[:, 1]

    def predict(self, urls: pd.Series | list[str], threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions (0 or 1)."""
        probs = self.predict_proba(urls)
        return (probs >= threshold).astype(int)
