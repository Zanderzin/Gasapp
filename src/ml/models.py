"""Modelos de previsao: ARIMA, Prophet, interface LSTM."""
import numpy as np
from loguru import logger

def train_arima(series, order=(1, 1, 1)):
    from statsmodels.tsa.arima.model import ARIMA
    result = ARIMA(series, order=order).fit()
    logger.info(f"ARIMA{order} | AIC={result.aic:.2f}")
    return result

def train_prophet(df, date_col="semana", target_col="preco_medio"):
    from prophet import Prophet
    prophet_df = df[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"})
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    model.fit(prophet_df)
    logger.info("Prophet treinado")
    return model

def evaluate(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    metrics = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 2)}
    logger.info(f"Metricas: {metrics}")
    return metrics
