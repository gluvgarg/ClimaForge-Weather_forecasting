import numpy as np
from sklearn.linear_model import LinearRegression

from src.core.base_forecaster import BaseForecaster


class LinearRegressionForecaster(BaseForecaster):
    """Linear regression baseline over rolling lag windows."""

    name = "Linear Regression"

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        if len(X) == 0:
            return self
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if len(X) == 0:
            return np.asarray([])
        return self.model.predict(X)

    def forecast_next(self, history, steps=1, window_size=3):
        values = list(np.asarray(history, dtype=float))
        predictions = []
        for _ in range(steps):
            if len(values) < window_size:
                prediction = float(np.mean(values))
            else:
                window = np.asarray(values[-window_size:], dtype=float).reshape(1, -1)
                prediction = float(self.model.predict(window)[0])
            predictions.append(prediction)
            values.append(prediction)
        return np.asarray(predictions, dtype=float)
