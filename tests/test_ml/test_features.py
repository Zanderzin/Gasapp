"""Testes do modulo de feature engineering."""
import pandas as pd
import numpy as np
import pytest
from src.ml.features import (
    add_time_features,
    add_lag_features,
    add_rolling_features,
    add_regional_features,
    add_target,
    build_features,
)


def make_df():
    """DataFrame minimo para testar features."""
    datas = pd.date_range("2024-01-01", periods=20, freq="W")
    return pd.DataFrame({
        "cnpj":         ["123"] * 20,
        "produto":      ["GASOLINA COMUM"] * 20,
        "regiao":       ["TAGUATINGA"] * 20,
        "data_coleta":  datas,
        "preco_revenda": np.linspace(5.5, 6.5, 20),
    })


def test_time_features():
    df = add_time_features(make_df())
    for col in ["mes", "trimestre", "semana_ano", "ano", "mes_sin", "mes_cos"]:
        assert col in df.columns


def test_lag_features():
    df = add_lag_features(make_df())
    assert "preco_revenda_lag_1" in df.columns
    assert "preco_revenda_lag_4" in df.columns
    assert df["preco_revenda_lag_1"].isna().sum() >= 1


def test_rolling_features():
    df = add_rolling_features(make_df())
    assert "preco_revenda_ma_4" in df.columns
    assert "preco_revenda_std_4" in df.columns


def test_regional_features():
    df = add_regional_features(make_df())
    assert "preco_medio_regiao" in df.columns
    assert "diff_regiao" in df.columns
    assert df["diff_regiao"].abs().max() < 0.01


def test_add_target():
    df = add_target(make_df())
    assert "target" in df.columns
    assert df["target"].iloc[-1] != df["target"].iloc[-1]  # ultimo deve ser NaN


def test_build_features_shape():
    df = build_features(make_df())
    assert len(df) > 0
    assert "target" in df.columns
    assert df["target"].isna().sum() == 0
