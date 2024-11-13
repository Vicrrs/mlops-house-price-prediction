# tests/unit/test_data_processing.py
import pytest
import pandas as pd
from src.data_processing import preprocess_data

def test_preprocess_data():
    df = pd.read_csv("data/raw/house_data.csv")
    X_train, X_val, y_train, y_val = preprocess_data(df)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_train.shape[1] == X_val.shape[1]
    assert not pd.isnull(X_train).any().any()
