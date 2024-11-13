# src/data_processing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Imputação de valores faltantes
    imputer = SimpleImputer(strategy='mean')
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Remoção de outliers simples (por exemplo, usando o IQR)
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Encoding de variáveis categóricas
    df = pd.get_dummies(df, drop_first=True)

    # Escalonamento de features
    scaler = StandardScaler()
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
