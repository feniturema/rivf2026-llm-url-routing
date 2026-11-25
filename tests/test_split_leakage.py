import pandas as pd
import pytest
from src.data.split import create_group_disjoint_split, materialize_split

def test_group_disjoint_split_zero_crossing():
    # Synthetic data with repeated groups
    data = []
    for i in range(100):
        # each group has 2 occurrences
        data.append({"url_hash": f"hash_{i}", "y": i % 2, "URL": f"https://site{i}.com"})
        data.append({"url_hash": f"hash_{i}", "y": i % 2, "URL": f"https://site{i}.com"})
        
    df = pd.DataFrame(data)
    manifest = create_group_disjoint_split(df, seed=42)
    train, cal, dev, test = materialize_split(manifest)
    
    set_train = set(train["url_hash"])
    set_cal = set(cal["url_hash"])
    set_dev = set(dev["url_hash"])
    set_test = set(test["url_hash"])
    
    assert len(set_train.intersection(set_cal)) == 0
    assert len(set_train.intersection(set_dev)) == 0
    assert len(set_train.intersection(set_test)) == 0
    assert len(set_cal.intersection(set_dev)) == 0
    assert len(set_cal.intersection(set_test)) == 0
    assert len(set_dev.intersection(set_test)) == 0
