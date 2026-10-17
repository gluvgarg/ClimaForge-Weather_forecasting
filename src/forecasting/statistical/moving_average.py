import numpy as np

from src.core.base_forecaster import BaseForecaster


class MovingAverageForecaster(BaseForecaster):
    """Simple moving-average baseline for univariate forecasting."""

    name = "Moving Average"

    def fit(self, X, y):
        return self

    def predict(self, X):
        if len(X) == 0:
            return np.asarray([])
        return np.mean(X, axis=1)

    def forecast_next(self, history, steps=1, window_size=3):
        values = list(np.asarray(history, dtype=float))
        predictions = []
        for _ in range(steps):
            window = values[-window_size:] if len(values) >= window_size else values
            prediction = float(np.mean(window))
            predictions.append(prediction)
            values.append(prediction)
        return np.asarray(predictions, dtype=float)
