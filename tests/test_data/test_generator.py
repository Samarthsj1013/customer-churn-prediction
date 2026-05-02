import pytest
import pandas as pd
from src.data.generator import generate_churn_data
from src.data.validator import validate_dataframe

def test_generate_returns_dataframe():
    df = generate_churn_data(n_samples=100)
    assert isinstance(df, pd.DataFrame)

def test_generate_correct_row_count():
    df = generate_churn_data(n_samples=500)
    assert len(df) == 500

def test_churn_column_is_binary():
    df = generate_churn_data(n_samples=200)
    assert set(df["churn"].unique()).issubset({0, 1})

def test_no_null_values():
    df = generate_churn_data(n_samples=200)
    assert df.isnull().sum().sum() == 0

def test_validation_passes():
    df = generate_churn_data(n_samples=200)
    assert validate_dataframe(df) == True

def test_churn_rate_reasonable():
    df = generate_churn_data(n_samples=5000)
    churn_rate = df["churn"].mean()
    assert 0.15 <= churn_rate <= 0.65