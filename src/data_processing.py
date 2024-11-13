# src/data_processing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.logger import logger
from joblib import dump
import os
import json


def load_data(filepath):
    logger.info(f"Carregando dados de {filepath}")
    df = pd.read_excel(filepath, decimal=",")
    return df


def preprocess_data(df):
    logger.info("Iniciando pré-processamento dos dados.")

    df.columns = [
        "No",
        "transaction_date",
        "house_age",
        "distance_MRT",
        "number_convenience_stores",
        "latitude",
        "longitude",
        "house_price_unit_area",
    ]

    df = df.drop("No", axis=1)
    cols_to_convert = df.columns
    for col in cols_to_convert:
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        logger.info("Valores nulos encontrados e removidos.")
    else:
        logger.info("Nenhum valor nulo encontrado.")

    X = df.drop("house_price_unit_area", axis=1)
    y = df["house_price_unit_area"]

    feature_names = X.columns.tolist()

    os.makedirs("models", exist_ok=True)

    with open("models/feature_names.json", "w") as f:
        json.dump(feature_names, f)
    logger.info("Nomes das features salvos em models/feature_names.json")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Features escalonadas.")

    # salvar o scaler para uso na API
    dump(scaler, "models/scaler.joblib")
    logger.info("Scaler salvo em models/scaler.joblib")

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    logger.info("Divisão dos dados em treino e validação concluída.")

    return X_train, X_val, y_train, y_val
