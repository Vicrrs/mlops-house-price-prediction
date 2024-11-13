# tests/unit/test_data_processing.py
import pytest
from src.data_processing import preprocess_data, load_data


def test_preprocess_data():
    df = load_data("data/raw/real_estate.xlsx")
    X_train, X_val, y_train, y_val = preprocess_data(df)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[1] == X_train.shape[1]
    assert not (X_train is None)
