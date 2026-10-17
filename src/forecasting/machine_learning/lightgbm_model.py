import numpy as np
from lightgbm import LGBMRegressor

from src.core.base_forecaster import BaseForecaster


class LightGBMForecaster(BaseForecaster):
    """LightGBM gradient-boosted regressor for tabular rainfall-target forecasting."""

    name = "LightGBM"

    def __init__(self, n_estimators=400, max_depth=-1, learning_rate=0.05, random_state=42):
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    def fit(self, X, y):
        self.model.fit(X, np.ravel(y))
        return self

    def predict(self, X):
        return self.model.predict(X)
