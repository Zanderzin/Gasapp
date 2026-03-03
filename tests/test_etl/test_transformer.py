"""Testes do modulo de transformacao."""
import pandas as pd
from src.etl.transformer import clean, aggregate_weekly

def make_df():
    return pd.DataFrame({
        "Produto": ["GASOLINA COMUM", "GASOLINA COMUM"],
        "Data da Coleta": ["01/01/2024", "08/01/2024"],
        "Valor de Venda": ["5,79", "5,85"],
        "Valor de Compra": ["5,20", "5,25"],
        "Estado - Sigla": ["SP", "SP"],
        "Municipio": ["SAO PAULO", "SAO PAULO"],
        "Regiao - Sigla": ["SE", "SE"],
        "Revenda": ["Posto A", "Posto A"],
        "CNPJ da Revenda": ["00000000000000", "00000000000000"],
        "Unidade de Medida": ["R$/l", "R$/l"],
        "Bandeira": ["BRANCA", "BRANCA"],
    })

def test_clean_returns_dataframe():
    assert isinstance(clean(make_df()), pd.DataFrame)

def test_clean_price_is_float():
    assert clean(make_df())["preco_venda"].dtype == float

def test_aggregate_has_semana():
    df = aggregate_weekly(clean(make_df()))
    assert "semana" in df.columns and "preco_medio" in df.columns
