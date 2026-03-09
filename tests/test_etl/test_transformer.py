"""Testes do modulo de transformacao."""
import pandas as pd
import pytest
from src.etl.transformer import clean, run_transform_all
from pathlib import Path


def make_df():
    return pd.DataFrame({
        "CNPJ": ["12345678000100", "12345678000100"],
        "RAZÃO": ["POSTO TESTE LTDA", "POSTO TESTE LTDA"],
        "FANTASIA": ["POSTO TESTE", "NAN"],
        "ENDEREÇO": ["RUA A", "RUA A"],
        "NÚMERO": ["100", "100"],
        "COMPLEMENTO": ["", ""],
        "BAIRRO": ["TAGUATINGA NORTE (TAGUATINGA)", "ASA NORTE"],
        "CEP": ["72000000", "70000000"],
        "MUNICÍPIO": ["BRASILIA", "BRASILIA"],
        "ESTADO": ["DISTRITO FEDERAL", "DISTRITO FEDERAL"],
        "BANDEIRA": ["BRANCA", "IPIRANGA"],
        "PRODUTO": ["GASOLINA COMUM", "ETANOL"],
        "UNIDADE DE MEDIDA": ["R$ / LITRO", "R$ / LITRO"],
        "PREÇO DE REVENDA": ["5,99", "3,89"],
        "DATA DA COLETA": ["01/01/2024", "08/01/2024"],
    })


def test_clean_returns_dataframe():
    result = clean(make_df())
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_clean_price_is_float():
    result = clean(make_df())
    assert result["preco_revenda"].dtype == float


def test_clean_filters_df_only():
    result = clean(make_df())
    assert result["estado"].unique().tolist() == ["DISTRITO FEDERAL"]


def test_clean_extrai_regiao():
    result = clean(make_df())
    assert "regiao" in result.columns
    assert "TAGUATINGA" in result["regiao"].values


def test_clean_substitui_nan_nome_fantasia():
    result = clean(make_df())
    assert "NAN" not in result["nome_fantasia"].str.upper().values


def test_clean_marca_glp():
    df = make_df()
    df.loc[0, "PRODUTO"] = "GLP"
    result = clean(df)
    assert "is_botijao" in result.columns




