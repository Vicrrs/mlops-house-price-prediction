# tests/integration/test_full_pipeline.py
import pytest
import os
from src.train import main as train_main
from src.evaluation import evaluate_model
from src.data_processing import load_data, preprocess_data


def test_full_pipeline():
    train_main()
    assert os.path.exists("models/model.joblib")

    data = load_data("data/raw/real_estate.xlsx")
    X_train, X_test, y_train, y_test = preprocess_data(data)
    r2, mae = evaluate_model("models/model.joblib", X_test, y_test)
    assert r2 >= 0
