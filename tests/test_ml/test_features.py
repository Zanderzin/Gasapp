"""Testes do modulo de features."""
import pandas as pd
from src.ml.features import add_time_features, add_lag_features

def make_df():
    dates = pd.date_range("2023-01-01", periods=20, freq="W")
    return pd.DataFrame({
        "semana": dates,
        "estado": ["SP"] * 20,
        "produto": ["GASOLINA COMUM"] * 20,
        "preco_medio": [5.5 + i * 0.05 for i in range(20)],
    })

def test_time_features():
    result = add_time_features(make_df())
    for col in ["mes", "trimestre", "semana_ano", "ano"]:
        assert col in result.columns

def test_lag_features():
    result = add_lag_features(make_df(), lags=[1, 2])
    assert "preco_medio_lag_1" in result.columns
    assert "preco_medio_lag_2" in result.columns
