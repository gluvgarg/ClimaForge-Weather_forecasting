import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.core.base_forecaster import BaseForecaster


class RandomForestForecaster(BaseForecaster):
    """Random Forest regressor for tabular rainfall-target forecasting."""

    name = "Random Forest"

    def __init__(self, n_estimators=300, max_depth=None, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X, y):
        self.model.fit(X, np.ravel(y))
        return self

    def predict(self, X):
        return self.model.predict(X)
