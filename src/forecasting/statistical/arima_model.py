from pathlib import Path

import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

from src.core.base_forecaster import BaseForecaster


class ARIMAForecaster(BaseForecaster):
    """Univariate ARIMA baseline fit on the full chronological series."""

    name = "ARIMA"
    WINDOWED = False

    def __init__(self, order=(2, 1, 2)):
        self.order = order
        self.results = None

    def fit(self, X, y):
        series = np.asarray(y, dtype=float)
        self.results = ARIMA(series, order=self.order).fit()
        return self

    def predict(self, X):
        steps = len(np.asarray(X))
        if steps == 0 or self.results is None:
            return np.asarray([])
        return np.asarray(self.results.forecast(steps=steps))

    def forecast_next(self, history, steps=1, window_size=3):
        series = np.asarray(history, dtype=float)
        # Refit on the full available history so the future forecast starts
        # from the true end of history rather than the train-only split used by fit().
        results = ARIMA(series, order=self.order).fit()
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
