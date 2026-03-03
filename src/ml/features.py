"""Feature engineering para series temporais."""
import pandas as pd

def add_time_features(df, date_col="semana"):
    df = df.copy()
    df["mes"] = df[date_col].dt.month
    df["trimestre"] = df[date_col].dt.quarter
    df["semana_ano"] = df[date_col].dt.isocalendar().week.astype(int)
    df["ano"] = df[date_col].dt.year
    return df

def add_lag_features(df, target="preco_medio", lags=None):
    if lags is None:
        lags = [1, 2, 4, 8]
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df.groupby(["estado", "produto"])[target].shift(lag)
    return df

def add_rolling_features(df, target="preco_medio", windows=None):
    if windows is None:
        windows = [4, 8, 12]
    df = df.copy()
    for w in windows:
        df[f"{target}_ma_{w}"] = df.groupby(["estado", "produto"])[target].transform(
            lambda x: x.shift(1).rolling(w).mean()
        )
    return df

def build_features(df):
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    return df.dropna()
