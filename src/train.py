# src/train.py
from src.data_processing import load_data, preprocess_data
from src.model import train_model

def main():
    data = load_data("data/raw/house_data.csv")
    X_train, X_val, y_train, y_val = preprocess_data(data)
    train_model(X_train, y_train, X_val, y_val)

if __name__ == "__main__":
    main()
