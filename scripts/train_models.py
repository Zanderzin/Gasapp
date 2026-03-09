"""
Script de treinamento e avaliacao de modelos por regiao.
Uso: py scripts/train_models.py [--produto GASOLINA COMUM] [--salvar]
"""
import sys
import argparse
import pickle
from pathlib import Path
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from loguru import logger

from src.etl.loader import load_parquet


# ── Configuracao ─────────────────────────────────────────────────────────────

PRODUTOS = ["GASOLINA COMUM", "GASOLINA ADITIVADA", "ETANOL", "DIESEL S10", "DIESEL S500"]

FEATURE_COLS = [
    "lag_1", "lag_2", "lag_4", "lag_8",
    "ma_4", "ma_8", "ma_12",
    "std_4", "std_8", "std_12",
    "delta_1", "delta_4",
    "mes_sin", "mes_cos", "mes", "trimestre", "semana_ano",
    "preco_df_lag1", "n_postos",
]

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}


# ── Preparacao de dados ───────────────────────────────────────────────────────

def agregar_por_regiao(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["semana"] = df["data_coleta"].dt.to_period("W").dt.start_time
    return (
        df.groupby(["regiao", "produto", "semana"])
        .agg(preco_revenda=("preco_revenda", "mean"), n_postos=("cnpj", "nunique"))
        .reset_index()
        .rename(columns={"semana": "data_coleta"})
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["regiao", "produto", "data_coleta"])
    g = ["regiao", "produto"]

    for lag in [1, 2, 4, 8]:
        df[f"lag_{lag}"] = df.groupby(g)["preco_revenda"].shift(lag)

    for w in [4, 8, 12]:
        df[f"ma_{w}"]  = df.groupby(g)["preco_revenda"].transform(lambda x: x.shift(1).rolling(w, min_periods=2).mean())
        df[f"std_{w}"] = df.groupby(g)["preco_revenda"].transform(lambda x: x.shift(1).rolling(w, min_periods=2).std())

    df["delta_1"] = df.groupby(g)["preco_revenda"].diff(1)
    df["delta_4"] = df.groupby(g)["preco_revenda"].diff(4)

    df["mes"]       = df["data_coleta"].dt.month
    df["trimestre"] = df["data_coleta"].dt.quarter
    df["semana_ano"]= df["data_coleta"].dt.isocalendar().week.fillna(0).astype(int)
    df["mes_sin"]   = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"]   = np.cos(2 * np.pi * df["mes"] / 12)

    media_df = df.groupby(["produto", "data_coleta"])["preco_revenda"].mean().reset_index()
    media_df["preco_df_lag1"] = media_df.groupby("produto")["preco_revenda"].shift(1)
    df = df.merge(media_df[["produto", "data_coleta", "preco_df_lag1"]], on=["produto", "data_coleta"], how="left")

    prox = df.groupby(g)["preco_revenda"].shift(-1)
    df["target_abs"] = prox
    df["target"]     = prox - df["preco_revenda"]

    return df.dropna(subset=["target"])


def split_temporal(df: pd.DataFrame, meses_teste: int = 6):
    cutoff = df["data_coleta"].max() - pd.DateOffset(months=meses_teste)
    return df[df["data_coleta"] <= cutoff].copy(), df[df["data_coleta"] > cutoff].copy()


# ── Metricas ──────────────────────────────────────────────────────────────────

def metricas(y_true, y_pred, nome="modelo") -> dict:
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.mean((y_true - y_pred) ** 2) ** 0.5
    r2   = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    logger.info(f"[{nome}] MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")
    return {"modelo": nome, "mae": mae, "rmse": rmse, "r2": r2}


# ── Treinamento ───────────────────────────────────────────────────────────────

def treinar_produto(df_feat: pd.DataFrame, produto: str, salvar: bool = False) -> dict:
    logger.info(f"=== {produto} ===")
    df_prod = df_feat[df_feat["produto"] == produto].copy()
    train, test = split_temporal(df_prod)

    feat_cols = [c for c in FEATURE_COLS if c in train.columns]
    train_v = train.dropna(subset=feat_cols + ["target"])
    test_v  = test.dropna(subset=feat_cols + ["target_abs"])

    logger.info(f"Treino: {len(train_v):,} | Teste: {len(test_v):,} | Features: {len(feat_cols)}")

    # Baseline MA4
    test_bl = test_v.dropna(subset=["ma_4"])
    res_bl = metricas(test_bl["target_abs"].values, test_bl["ma_4"].values, "baseline_ma4")

    # XGBoost
    xgb = XGBRegressor(**XGB_PARAMS)
    xgb.fit(train_v[feat_cols], train_v["target"], verbose=False)
    delta_pred = xgb.predict(test_v[feat_cols])
    y_pred = test_v["preco_revenda"].values + delta_pred
    y_true = test_v["target_abs"].values
    res_xgb = metricas(y_true, y_pred, "xgb_regiao")

    # Feature importance
    imp = pd.DataFrame({"feature": feat_cols, "importance": xgb.feature_importances_})
    logger.info(f"Top 5 features:\n{imp.sort_values('importance', ascending=False).head(5).to_string(index=False)}")

    # Salva modelo
    if salvar:
        path = Path("models") / f"xgb_{produto.lower().replace(' ', '_')}.pkl"
        path.parent.mkdir(exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": xgb, "feature_cols": feat_cols, "produto": produto}, f)
        logger.success(f"Modelo salvo: {path}")

    return {
        "produto": produto,
        "baseline": res_bl,
        "xgboost": res_xgb,
        "feature_importance": imp.sort_values("importance", ascending=False),
        "model": xgb,
        "feature_cols": feat_cols,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--produto", nargs="+", default=PRODUTOS)
    parser.add_argument("--salvar", action="store_true")
    args = parser.parse_args()

    df = load_parquet()
    df_reg = agregar_por_regiao(df)
    df_feat = build_features(df_reg)

    resultados = []
    for produto in args.produto:
        if produto not in df_feat["produto"].unique():
            logger.warning(f"Produto nao encontrado: {produto}")
            continue
        res = treinar_produto(df_feat, produto, salvar=args.salvar)
        resultados.append({"produto": produto, **res["baseline"], "tipo": "baseline_ma4"})
        resultados.append({"produto": produto, **res["xgboost"], "tipo": "xgb_regiao"})

    print("\n=== RESUMO FINAL ===")
    resumo = pd.DataFrame(resultados)[["produto", "tipo", "mae", "rmse", "r2"]]
    print(resumo.sort_values(["produto", "mae"]).to_string(index=False))


if __name__ == "__main__":
    main()
