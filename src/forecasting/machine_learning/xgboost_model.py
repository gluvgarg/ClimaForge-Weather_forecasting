import numpy as np
from xgboost import XGBRegressor

from src.core.base_forecaster import BaseForecaster


class XGBoostForecaster(BaseForecaster):
    """XGBoost gradient-boosted regressor for tabular rainfall-target forecasting."""

    name = "XGBoost"

    def __init__(self, n_estimators=400, max_depth=6, learning_rate=0.05, random_state=42):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X, y):
        self.model.fit(X, np.ravel(y))
        return self

    def predict(self, X):
        return self.model.predict(X)
