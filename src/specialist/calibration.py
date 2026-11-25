"""
Isotonic Probability Calibrator.
Fits on validation (CAL / DEV_POLICY) uncalibrated probabilities.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.is_fitted = False

    def fit(self, uncalibrated_probs: np.ndarray, y_true: np.ndarray) -> IsotonicCalibrator:
        p = np.asarray(uncalibrated_probs, dtype=np.float64)
        y = np.asarray(y_true, dtype=np.float64)
        self.iso.fit(p, y)
        self.is_fitted = True
        return self

    def calibrate(self, uncalibrated_probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Calibrator is not fitted.")
        p = np.asarray(uncalibrated_probs, dtype=np.float64)
        cal_p = self.iso.predict(p)
        return np.clip(cal_p, 0.0, 1.0)
