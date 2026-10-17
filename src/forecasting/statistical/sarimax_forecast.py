from pathlib import Path

import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.core.base_forecaster import BaseForecaster


class SARIMAXForecaster(BaseForecaster):
    """Seasonal ARIMA baseline. Seasonal component is off by default so it
    degrades safely on the short series typical of the GUI's regional forecasts."""

    name = "SARIMAX"
    WINDOWED = False

    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.results = None

    def fit(self, X, y):
        series = np.asarray(y, dtype=float)
        self.results = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def predict(self, X):
        steps = len(np.asarray(X))
        if steps == 0 or self.results is None:
            return np.asarray([])
        return np.asarray(self.results.forecast(steps=steps))

    def forecast_next(self, history, steps=1, window_size=3):
        series = np.asarray(history, dtype=float)
        results = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return np.asarray(results.forecast(steps=steps))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.results.save(str(path))
        return path

    @classmethod
    def load(cls, path):
        instance = cls()
        instance.results = sm.load(str(path))
        return instance
