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


class XGBoostModel:
    """XGBoost com features de serie temporal."""

    def __init__(self, params: dict = None):
        from xgboost import XGBRegressor
        self.name = "xgboost"
        self.params = params or {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }
        self.model = XGBRegressor(**self.params)
        self.feature_cols = None

    FEATURE_COLS = [
        "mes_sin", "mes_cos", "semana_sin", "semana_cos",
        "mes", "trimestre", "semana_ano",
        "preco_revenda_lag_1", "preco_revenda_lag_2",
        "preco_revenda_lag_4", "preco_revenda_lag_8", "preco_revenda_lag_12",
        "preco_revenda_ma_4", "preco_revenda_ma_8", "preco_revenda_ma_12",
        "preco_revenda_std_4", "preco_revenda_std_8", "preco_revenda_std_12",
        "preco_medio_regiao", "diff_regiao",
        "delta_lag_1", "delta_lag_2", "delta_4s",
        "preco_medio_df_lag1",
    ]

    def fit(self, train: pd.DataFrame) -> None:
        self.feature_cols = [c for c in self.FEATURE_COLS if c in train.columns]
        X = train[self.feature_cols]
        y = train["target"]
        self.model.fit(X, y, verbose=False)
        logger.info(f"XGBoost treinado | features: {len(self.feature_cols)} | amostras: {len(train):,}")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = df[self.feature_cols]
        return pd.Series(self.model.predict(X), index=df.index)

    def evaluate(self, test: pd.DataFrame) -> dict:
        df_valid = test.dropna(subset=self.feature_cols + ["target"])
        y_true = df_valid["target"]
        y_pred = self.predict(df_valid)
        return evaluate(y_true, y_pred, self.name)

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)



