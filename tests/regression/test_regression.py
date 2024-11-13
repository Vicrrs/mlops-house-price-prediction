# tests/regression/test_regression.py
import pytest
import json
from src.evaluation import evaluate_model
from src.data_processing import load_data, preprocess_data

def test_regression():
    # Carregar resultados anteriores
    with open('tests/regression/previous_results.json') as f:
        previous_results = json.load(f)
    data = load_data("data/raw/real_estate.xlsx")
    X_train, X_test, y_train, y_test = preprocess_data(data)
    r2, mae = evaluate_model('models/model.joblib', X_test, y_test)
    assert r2 >= previous_results['r2']
    assert mae <= previous_results['mae']
