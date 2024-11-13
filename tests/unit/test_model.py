# tests/unit/test_model.py
import pytest
from src.model import train_model
import numpy as np

def test_train_model():
    # Dados sintéticos para teste
    X_train = np.random.rand(100, 6)
    y_train = np.random.rand(100)
    X_val = np.random.rand(20, 6)
    y_val = np.random.rand(20)
    model = train_model(X_train, y_train, X_val, y_val)
    assert model is not None
