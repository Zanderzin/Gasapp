"""Modelos de previsao de precos de combustiveis."""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger


def evaluate(y_true, y_pred, nome="modelo"):
    """Calcula MAE, RMSE e R2."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    logger.info(f"[{nome}] MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")
    return {"modelo": nome, "mae": mae, "rmse": rmse, "r2": r2}


class BaselineMA:
    """Baseline: media movel das ultimas N semanas por posto+produto."""

    def __init__(self, window: int = 4):
        self.window = window
        self.name = f"baseline_ma{window}"

    def predict(self, df: pd.DataFrame, target: str = "preco_revenda") -> pd.Series:
        col = f"{target}_ma_{self.window}"
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' nao encontrada. Rode build_features antes.")
        return df[col]

    def evaluate(self, df: pd.DataFrame, target: str = "preco_revenda") -> dict:
        df_valid = df.dropna(subset=[f"{target}_ma_{self.window}", "target"])
        y_true = df_valid["target"]
        y_pred = self.predict(df_valid, target)
        return evaluate(y_true, y_pred, self.name)


class BaselineLast:
    """Baseline ingênuo: repete o ultimo preco observado (random walk)."""

    def __init__(self):
        self.name = "baseline_last"

    def predict(self, df: pd.DataFrame, target: str = "preco_revenda") -> pd.Series:
        col = f"{target}_lag_1"
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' nao encontrada. Rode build_features antes.")
        return df[col]

    def evaluate(self, df: pd.DataFrame, target: str = "preco_revenda") -> dict:
        df_valid = df.dropna(subset=[f"{target}_lag_1", "target"])
        y_true = df_valid["target"]
        y_pred = self.predict(df_valid, target)
        return evaluate(y_true, y_pred, self.name)


def train_test_split_temporal(
    df: pd.DataFrame,
    date_col: str = "data_coleta",
    test_months: int = 3,
):
    """Split temporal — ultimos N meses como teste, resto como treino."""
    cutoff = df[date_col].max() - pd.DateOffset(months=test_months)
    train = df[df[date_col] <= cutoff].copy()
    test = df[df[date_col] > cutoff].copy()
    logger.info(
        f"Split temporal | treino: {len(train):,} ({train[date_col].min().date()} a {train[date_col].max().date()})"
        f" | teste: {len(test):,} ({test[date_col].min().date()} a {test[date_col].max().date()})"
    )
    return train, test
