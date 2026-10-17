from src.forecasting.machine_learning.random_forest import RandomForestForecaster
from src.forecasting.machine_learning.xgboost_model import XGBoostForecaster
from src.forecasting.machine_learning.lightgbm_model import LightGBMForecaster
from src.forecasting.deep_learning.lstm_model import LSTMForecaster
from src.forecasting.deep_learning.gru_model import GRUForecaster
from src.forecasting.deep_learning.transformer_model import TransformerForecaster
from src.forecasting.deep_learning.hybrid_models.generic_hybrid import build_default_hybrid

TABULAR_FORECAST_MODES = [
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "LSTM",
    "GRU",
    "Transformer",
    "Hybrid (RF + XGBoost)",
]

SEQUENCE_FORECAST_MODES = {"LSTM", "GRU", "Transformer"}


def build_tabular_forecaster(mode, n_features=None):
    if mode == "Random Forest":
        return RandomForestForecaster()
    if mode == "XGBoost":
        return XGBoostForecaster()
    if mode == "LightGBM":
        return LightGBMForecaster()
    if mode == "LSTM":
        return LSTMForecaster(n_features=n_features)
    if mode == "GRU":
        return GRUForecaster(n_features=n_features)
    if mode == "Transformer":
        return TransformerForecaster(n_features=n_features)
    if mode == "Hybrid (RF + XGBoost)":
        return build_default_hybrid()
    raise ValueError(f"Unsupported tabular forecast mode: {mode}")
