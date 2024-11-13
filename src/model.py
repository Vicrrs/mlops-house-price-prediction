# src/model.py
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump
import os
from src.logger import logger
import numpy as np

def train_model(X_train, y_train, X_val, y_val):
    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForestRegressor")
        
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        logger.info("Iniciando o treinamento do modelo RandomForestRegressor.")
        model.fit(X_train, y_train)
        logger.info("Treinamento do modelo concluído.")
        
        predictions = model.predict(X_val)
        mse = mean_squared_error(y_val, predictions)
        rmse = np.sqrt(mse)
        logger.info(f"RMSE no conjunto de validação: {rmse}")
        mlflow.log_metric("rmse", rmse)
        
        # Salvar o modelo
        os.makedirs("models", exist_ok=True)
        dump(model, "models/model.joblib")
        logger.info("Modelo salvo em models/model.joblib")
        
        mlflow.end_run()
        return model
