from pathlib import Path

import numpy as np
import torch

from src.core.base_forecaster import BaseForecaster
from src.forecasting.deep_learning.torch_modules import LSTMRegressor, predict as torch_predict, train_loop


class LSTMForecaster(BaseForecaster):
    """LSTM sequence regressor for tabular rainfall-target forecasting."""

    name = "LSTM"

    def __init__(self, n_features=None, hidden_size=32, num_layers=1, epochs=20, lr=1e-3):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if self.n_features is None:
            self.n_features = X.shape[-1]
        self.model = LSTMRegressor(self.n_features, self.hidden_size, self.num_layers)
        train_loop(self.model, X, y, epochs=self.epochs, lr=self.lr)
        return self

    def predict(self, X):
        if self.model is None:
            return np.asarray([])
        return torch_predict(self.model, np.asarray(X, dtype=float))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "n_features": self.n_features,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location="cpu")
        instance = cls(
            n_features=checkpoint["n_features"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
        )
        instance.model = LSTMRegressor(instance.n_features, instance.hidden_size, instance.num_layers)
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.model.eval()
        return instance
