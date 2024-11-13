# tests/stress/test_stress.py
import requests
from multiprocessing import Pool


def make_request(_):
    payload = {
        "transaction_date": 2013.5,
        "house_age": 13.3,
        "distance_MRT": 561.9845,
        "number_convenience_stores": 5,
        "latitude": 24.98746,
        "longitude": 121.54391,
    }
    response = requests.post("http://localhost:5000/predict", json=payload)
    return response.status_code


def test_stress():
    # API está em execução antes de executar este teste
    with Pool(100) as pool:  # 100 processos simultâneos
        results = pool.map(make_request, range(1000))  # 1000 requisições
    assert all(status == 200 for status in results)
