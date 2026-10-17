from src.core.base_forecaster import BaseForecaster
from src.forecasting.deep_learning.hybrid_models.combiners import AverageCombiner


class HybridForecaster(BaseForecaster):
    """Generic hybrid forecaster: fits N component forecasters on the same
    input and combines their predictions with a pluggable combiner.

    New hybrid variants are built by instantiating this class with a
    different component list/combiner - the framework itself never changes.
    Components must accept the same X shape (e.g. all tabular or all
    sequence-shaped); mixing shapes is the caller's responsibility.
    """

    def __init__(self, components, combiner=None, name="Hybrid"):
        self.components = components
        self.combiner = combiner or AverageCombiner()
        self.name = name

    def fit(self, X, y):
        for component in self.components:
            component.fit(X, y)
        return self

    def predict(self, X):
        predictions = [component.predict(X) for component in self.components]
        return self.combiner.combine(predictions)


def build_default_hybrid():
    from src.forecasting.machine_learning.random_forest import RandomForestForecaster
    from src.forecasting.machine_learning.xgboost_model import XGBoostForecaster

    return HybridForecaster(
        components=[RandomForestForecaster(), XGBoostForecaster()],
        name="Hybrid (RF + XGBoost)",
    )
