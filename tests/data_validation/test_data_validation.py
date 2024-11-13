# tests/data_validation/test_data_validation.py
import pytest
import pandas as pd

def test_data_schema():
    df = pd.read_excel('data/raw/real_estate.xlsx', decimal=',')
    expected_columns = [
        'No',
        'X1 transaction date',
        'X2 house age',
        'X3 distance to the nearest MRT station',
        'X4 number of convenience stores',
        'X5 latitude',
        'X6 longitude',
        'Y house price of unit area'
    ]
    assert all(col in df.columns for col in expected_columns)
    assert df['Y house price of unit area'].dtype in ['float64', 'int64']
