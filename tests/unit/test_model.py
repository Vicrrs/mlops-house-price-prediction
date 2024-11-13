# tests/unit/test_model.py
import pytest
from sklearn.datasets import make_regression
from src.model import train_model

def test_train_model():
    X_train, y_train = make_regression(n_samples=100, n_features=10)
    X_val, y_val = make_regression(n_samples=20, n_features=10)
    model = train_model(X_train, y_train, X_val, y_val)
    assert model is not None
