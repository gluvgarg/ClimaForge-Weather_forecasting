from pathlib import Path

import numpy as np
import torch

from src.core.base_forecaster import BaseForecaster
from src.forecasting.deep_learning.torch_modules import (
    TransformerRegressor,
    predict as torch_predict,
    train_loop,
)


class TransformerForecaster(BaseForecaster):
    """Transformer-encoder sequence regressor for tabular rainfall-target forecasting."""

    name = "Transformer"

    def __init__(self, n_features=None, d_model=32, nhead=4, num_layers=1, epochs=20, lr=1e-3):
        self.n_features = n_features
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if self.n_features is None:
            self.n_features = X.shape[-1]
        self.model = TransformerRegressor(self.n_features, self.d_model, self.nhead, self.num_layers)
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
                "d_model": self.d_model,
                "nhead": self.nhead,
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
            d_model=checkpoint["d_model"],
            nhead=checkpoint["nhead"],
            num_layers=checkpoint["num_layers"],
        )
        instance.model = TransformerRegressor(
            instance.n_features, instance.d_model, instance.nhead, instance.num_layers
        )
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.model.eval()
        return instance
