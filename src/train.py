# src/train.py
from src.data_processing import load_data, preprocess_data
from src.model import train_model
from src.logger import logger


def main():
    logger.info("Iniciando o treinamento do modelo.")
    data = load_data("data/raw/real_estate.xlsx")
    X_train, X_val, y_train, y_val = preprocess_data(data)
    train_model(X_train, y_train, X_val, y_val)
    logger.info("Treinamento concluído.")


if __name__ == "__main__":
    main()
