"""Limpeza e transformacao dos dados brutos da ANP."""
import pandas as pd
from loguru import logger
from pathlib import Path

COLUMNS_MAP = {
    "Regiao - Sigla": "regiao",
    "Estado - Sigla": "estado",
    "Municipio": "municipio",
    "Revenda": "posto",
    "CNPJ da Revenda": "cnpj",
    "Produto": "produto",
    "Data da Coleta": "data_coleta",
    "Valor de Venda": "preco_venda",
    "Valor de Compra": "preco_compra",
    "Unidade de Medida": "unidade",
    "Bandeira": "bandeira",
}

def load_raw(filepath):
    suffix = Path(filepath).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath, dtype=str)
    else:
        df = pd.read_csv(filepath, sep=";", encoding="latin-1", dtype=str)
    logger.info(f"Arquivo carregado: {len(df)} linhas")
    return df

def clean(df):
    df = df.rename(columns=COLUMNS_MAP)
    df = df[[c for c in COLUMNS_MAP.values() if c in df.columns]]
    for col in ["preco_venda", "preco_compra"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors="coerce")
    df["data_coleta"] = pd.to_datetime(df["data_coleta"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["preco_venda", "data_coleta"])
    df = df[df["preco_venda"] > 0]
    logger.info(f"Apos limpeza: {len(df)} linhas")
    return df

def aggregate_weekly(df):
    df["semana"] = df["data_coleta"].dt.to_period("W").dt.start_time
    agg = df.groupby(["semana", "estado", "municipio", "produto"]).agg(
        preco_medio=("preco_venda", "mean"),
        preco_min=("preco_venda", "min"),
        preco_max=("preco_venda", "max"),
        preco_mediana=("preco_venda", "median"),
        n_postos=("cnpj", "nunique"),
    ).reset_index()
    logger.info(f"Agregacao semanal: {len(agg)} registros")
    return agg

def run_transform(filepath):
    df = load_raw(filepath)
    df = clean(df)
    df = aggregate_weekly(df)
    return df
