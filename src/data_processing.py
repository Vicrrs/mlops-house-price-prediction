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
    # Ler o Excel considerando a vírgula como separador decimal
    df = pd.read_excel(filepath, decimal=",")
    return df


def preprocess_data(df):
    logger.info("Iniciando pré-processamento dos dados.")

    # Renomear colunas
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

    # Remover a coluna 'No'
    df = df.drop("No", axis=1)

    # Converter colunas numéricas que estão como strings devido à vírgula decimal
    cols_to_convert = df.columns
    for col in cols_to_convert:
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

    # Verificar e remover valores nulos
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        logger.info("Valores nulos encontrados e removidos.")
    else:
        logger.info("Nenhum valor nulo encontrado.")

    # Separar features e target
    X = df.drop("house_price_unit_area", axis=1)
    y = df["house_price_unit_area"]

    # Salvar os nomes das features
    feature_names = X.columns.tolist()

    # Garantir que o diretório 'models/' existe
    os.makedirs("models", exist_ok=True)

    # Salvar os nomes das features em 'models/feature_names.json'
    with open("models/feature_names.json", "w") as f:
        json.dump(feature_names, f)
    logger.info("Nomes das features salvos em models/feature_names.json")

    # Escalonamento das features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Features escalonadas.")

    # Salvar o scaler para uso na API
    dump(scaler, "models/scaler.joblib")
    logger.info("Scaler salvo em models/scaler.joblib")

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    logger.info("Divisão dos dados em treino e validação concluída.")

    return X_train, X_val, y_train, y_val
