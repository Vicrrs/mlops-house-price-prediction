# src/model.py
import mlflow
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from joblib import dump
import os

def train_model(X_train, y_train, X_val, y_val):
    with mlflow.start_run():
        mlflow.log_param("model_type", "XGBRegressor")

        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        rmse = mean_squared_error(y_val, predictions, squared=False)
        mlflow.log_metric("rmse", rmse)

        # Salvar o modelo
        os.makedirs("models", exist_ok=True)
        dump(model, "models/model.joblib")

        mlflow.end_run()
        return model
