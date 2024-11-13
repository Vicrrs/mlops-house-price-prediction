# src/train.py
from src.data_processing import load_data, preprocess_data
from src.model import train_model
from src.evaluation import evaluate_model
from src.logger import logger

def main():
    logger.info("Iniciando o treinamento do modelo.")
    data = load_data("data/raw/real_estate.xlsx")
    X_train, X_val, y_train, y_val = preprocess_data(data)
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Avaliar o modelo após o treinamento
    r2, mae = evaluate_model("models/model.joblib", X_val, y_val)
    logger.info(f"Desempenho do modelo - R2: {r2}, MAE: {mae}")

if __name__ == "__main__":
    main()
