# tests/integration/test_deployment.py
import os
import requests


def test_docker_build():
    result = os.system("docker build -t house-price-prediction .")
    assert result == 0


def test_prediction_service():
    # Certifique-se de que o container está em execução antes de executar este teste
    payload = {
        "transaction_date": 2013.5,
        "house_age": 13.3,
        "distance_MRT": 561.9845,
        "number_convenience_stores": 5,
        "latitude": 24.98746,
        "longitude": 121.54391,
    }
    response = requests.post("http://localhost:5000/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
