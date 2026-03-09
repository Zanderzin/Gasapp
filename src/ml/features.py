"""Feature engineering para series temporais de precos de combustiveis."""
import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame, date_col: str = "data_coleta") -> pd.DataFrame:
    """Features de calendario."""
    df = df.copy()
    df["mes"] = df[date_col].dt.month
    df["trimestre"] = df[date_col].dt.quarter
    df["semana_ano"] = df[date_col].dt.isocalendar().week.fillna(0).astype(int)
    df["ano"] = df[date_col].dt.year
    df["dia_semana"] = df[date_col].dt.dayofweek
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    df["semana_sin"] = np.sin(2 * np.pi * df["semana_ano"] / 52)
    df["semana_cos"] = np.cos(2 * np.pi * df["semana_ano"] / 52)
    df["delta_lag_1"] = df.groupby(["cnpj", "produto"])["preco_revenda"].diff(1)
    df["delta_lag_2"] = df.groupby(["cnpj", "produto"])["preco_revenda"].diff(2)
    df["delta_4s"] = df.groupby(["cnpj", "produto"])["preco_revenda"].diff(4)
    return df


def add_lag_features(
    df: pd.DataFrame,
    target: str = "preco_revenda",
    lags: list = None,
    group_cols: list = None,
) -> pd.DataFrame:
    """Lags do preco — por posto e produto."""
    if lags is None:
        lags = [1, 2, 4, 8, 12]
    if group_cols is None:
        group_cols = ["cnpj", "produto"]
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df.groupby(group_cols)[target].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target: str = "preco_revenda",
    windows: list = None,
    group_cols: list = None,
) -> pd.DataFrame:
    """Medias moveis e desvio padrao — por posto e produto."""
    if windows is None:
        windows = [4, 8, 12]
    if group_cols is None:
        group_cols = ["cnpj", "produto"]
    df = df.copy()
    for w in windows:
        rolled = df.groupby(group_cols)[target].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).mean()
        )
        df[f"{target}_ma_{w}"] = rolled
        rolled_std = df.groupby(group_cols)[target].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).std()
        )
        df[f"{target}_std_{w}"] = rolled_std
    return df


def add_regional_features(df: pd.DataFrame, target: str = "preco_revenda") -> pd.DataFrame:
    """Preco medio da regiao e diferenca do posto em relacao a media regional."""
    df = df.copy()
    media_regional = (
        df.groupby(["regiao", "produto", "data_coleta"])[target]
        .mean()
        .reset_index()
        .rename(columns={target: "preco_medio_regiao"})
    )
    df = df.merge(media_regional, on=["regiao", "produto", "data_coleta"], how="left")
    df["diff_regiao"] = df[target] - df["preco_medio_regiao"]
    # Preco medio do DF na semana anterior (proxy da tendencia macro)
    media_df = (
        df.groupby(["produto", "data_coleta"])["preco_revenda"]
        .mean()
        .reset_index()
        .rename(columns={"preco_revenda": "preco_medio_df"})
    )
    media_df["preco_medio_df_lag1"] = media_df.groupby("produto")["preco_medio_df"].shift(1)
    df = df.merge(media_df[["produto", "data_coleta", "preco_medio_df_lag1"]], on=["produto", "data_coleta"], how="left")
    return df


def add_target(df: pd.DataFrame, target: str = "preco_revenda", horizonte: int = 1) -> pd.DataFrame:
    """Cria coluna alvo: preco do posto na proxima semana."""
    df = df.copy()
    df["target"] = df.groupby(["cnpj", "produto"])[target].shift(-horizonte)
    return df


def build_features(
    df: pd.DataFrame,
    target: str = "preco_revenda",
    horizonte: int = 1,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Pipeline completo de feature engineering."""
    df = add_time_features(df)
    df = add_lag_features(df, target=target)
    df = add_rolling_features(df, target=target)
    df = add_regional_features(df, target=target)
    df = add_target(df, target=target, horizonte=horizonte)
    if drop_na:
        df = df.dropna(subset=["target"])
    return df



def add_delta_target(df: pd.DataFrame, target: str = "preco_revenda", horizonte: int = 1) -> pd.DataFrame:
    """Target como variacao de preco em vez de valor absoluto."""
    df = df.copy()
    prox = df.groupby(["cnpj", "produto"])[target].shift(-horizonte)
    df["target_delta"] = prox - df[target]
    df["target"] = prox
    return df

