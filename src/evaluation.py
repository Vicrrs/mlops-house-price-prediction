# src/evaluation.py
import mlflow
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import load
from src.logger import logger


def evaluate_model(model_path, X_test, y_test):
    logger.info(f"Carregando modelo para avaliação: {model_path}")
    model = load(model_path)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    logger.info(f"R2: {r2}, MAE: {mae}")

    with mlflow.start_run():
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.end_run()

    return r2, mae
