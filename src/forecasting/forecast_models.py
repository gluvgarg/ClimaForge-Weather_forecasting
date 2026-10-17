from src.forecasting.statistical.moving_average import MovingAverageForecaster
from src.forecasting.statistical.regression_forecast import LinearRegressionForecaster
from src.forecasting.statistical.arima_model import ARIMAForecaster
from src.forecasting.statistical.sarimax_forecast import SARIMAXForecaster

SUPPORTED_MODES = ["Linear Regression", "Moving Average", "ARIMA", "SARIMAX"]


def build_forecaster(mode):
    if mode == "Moving Average":
        return MovingAverageForecaster()
    if mode == "Linear Regression":
        return LinearRegressionForecaster()
    if mode == "ARIMA":
        return ARIMAForecaster()
    if mode == "SARIMAX":
        return SARIMAXForecaster()
    raise ValueError(f"Unsupported Phase 1 forecast mode: {mode}")
