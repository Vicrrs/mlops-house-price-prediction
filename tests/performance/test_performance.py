# tests/performance/test_performance.py
import pytest
import time
from src.model import train_model
from src.data_processing import preprocess_data, load_data
import numpy as np
from joblib import load

@pytest.fixture
def data():
    df = load_data("data/raw/real_estate.xlsx")
    return preprocess_data(df)

def test_training_time(data):
    X_train, X_val, y_train, y_val = data
    start_time = time.time()
    train_model(X_train, y_train, X_val, y_val)
    end_time = time.time()
    training_time = end_time - start_time
    assert training_time < 60  # Ajuste o tempo conforme necessário

def test_inference_time():
    model = load("models/model.joblib")
    sample = np.random.rand(1, 6)
    start_time = time.time()
    model.predict(sample)
    end_time = time.time()
    inference_time = end_time - start_time
    assert inference_time < 0.1  # A inferência deve ser rápida
