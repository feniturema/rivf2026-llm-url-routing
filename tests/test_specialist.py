import numpy as np
import pytest
from src.specialist.char_model import CharTFIDFSpecialist
from src.specialist.calibration import IsotonicCalibrator

def test_char_specialist_fit_and_calibrate():
    urls_train = [
        "https://google.com", "https://github.com", "https://apple.com",
        "https://paypa1-secure.top", "https://bank-login.xyz", "https://account-verify.cfd"
    ]
    y_train = [0, 0, 0, 1, 1, 1]
    
    model = CharTFIDFSpecialist(min_df=1, ngram_range=(2, 3))
    model.fit(urls_train, y_train)
    
    probs = model.predict_proba(urls_train)
    assert len(probs) == 6
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    
    calibrator = IsotonicCalibrator()
    calibrator.fit(probs, np.array(y_train))
    cal_p = calibrator.calibrate(probs)
    assert len(cal_p) == 6
    assert cal_p.min() >= 0.0 and cal_p.max() <= 1.0
