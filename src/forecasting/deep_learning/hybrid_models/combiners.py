import numpy as np


class AverageCombiner:
    """Combine component predictions via an unweighted mean."""

    def combine(self, predictions_list):
        stacked = np.vstack([np.asarray(predictions, dtype=float) for predictions in predictions_list])
        return np.mean(stacked, axis=0)


class WeightedCombiner:
    """Combine component predictions via a fixed weight per component."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def combine(self, predictions_list):
        stacked = np.vstack([np.asarray(predictions, dtype=float) for predictions in predictions_list])
        weights = self.weights / self.weights.sum()
        return np.tensordot(weights, stacked, axes=([0], [0]))
