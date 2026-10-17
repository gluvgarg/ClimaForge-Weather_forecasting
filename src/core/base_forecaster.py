from abc import ABC, abstractmethod
from pathlib import Path

import joblib

from src.evaluation.regression_metrics import ForecastEvaluator


class BaseForecaster(ABC):
    """Common interface every forecasting model in this package must implement.

    WINDOWED models consume sliding-window (X, y) pairs built by WindowGenerator
    or the tabular feature CSVs. Non-windowed models (ARIMA/SARIMAX) consume the
    contiguous chronological series directly - see ForecastManager for the branch.
    """

    name = "Base Forecaster"
    WINDOWED = True

    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def predict(self, X):
        ...

    def evaluate(self, actual, predicted):
        return ForecastEvaluator().evaluate(actual, predicted)

    def forecast_next(self, history, steps=1, window_size=3):
        raise NotImplementedError(
            f"{self.name} does not support recursive future forecasting."
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path):
        return joblib.load(Path(path))
