# src/api.py
from flask import Flask, request, jsonify
from joblib import load
import numpy as np
from src.logger import logger
import json

app = Flask(__name__)
model = load("models/model.joblib")
scaler = load("models/scaler.joblib")
logger.info("Modelo e scaler carregados para a API.")

# Carregar os nomes das features
with open("models/feature_names.json", "r") as f:
    feature_names = json.load(f)
logger.info("Nomes das features carregados.")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    # Ordenar os dados de entrada de acordo com os nomes das features
    features = [data.get(feature) for feature in feature_names]
    # Converter para DataFrame
    import pandas as pd

    input_data = pd.DataFrame([features], columns=feature_names)
    # Escalonar as features
    features_scaled = scaler.transform(input_data)
    # Fazer a predição
    prediction = model.predict(features_scaled)
    logger.info(f"Recebida requisição de predição: {data}, resposta: {prediction[0]}")
    return jsonify({"prediction": prediction[0]})


if __name__ == "__main__":
    logger.info("API iniciada.")
    app.run(host="0.0.0.0", port=5000)
