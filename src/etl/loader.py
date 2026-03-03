"""Carrega dados no PostgreSQL e em Parquet."""
import os
import pandas as pd
from sqlalchemy import create_engine
from loguru import logger

def get_engine():
    url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/fuel_predictor")
    return create_engine(url)

def load_to_db(df, table="precos_semanais", if_exists="append"):
    engine = get_engine()
    df.to_sql(table, engine, if_exists=if_exists, index=False, method="multi", chunksize=1000)
    logger.success(f"{len(df)} registros inseridos em '{table}'")

def load_to_parquet(df, path):
    df.to_parquet(path, index=False)
    logger.success(f"Parquet salvo: {path}")
