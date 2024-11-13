# tests/unit/test_evaluation.py
import pytest
from sklearn.datasets import make_regression
from src.evaluation import evaluate_model
from joblib import dump

def test_evaluate_model():
    X_test, y_test = make_regression(n_samples=20, n_features=10)
    # Criar e salvar um modelo dummy para teste
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_test, y_test)
    dump(model, "models/test_model.joblib")
    r2, mae = evaluate_model("models/test_model.joblib", X_test, y_test)
    assert r2 <= 1.0
