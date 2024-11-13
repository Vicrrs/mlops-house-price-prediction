# tests/unit/test_evaluation.py
import pytest
from src.evaluation import evaluate_model
from joblib import dump
import numpy as np

def test_evaluate_model():
    # Dados sintéticos para teste
    X_test = np.random.rand(20, 6)
    y_test = np.random.rand(20)
    # Criar e salvar um modelo dummy para teste
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_test, y_test)
    dump(model, "models/test_model.joblib")
    r2, mae = evaluate_model("models/test_model.joblib", X_test, y_test)
    assert r2 <= 1.0
