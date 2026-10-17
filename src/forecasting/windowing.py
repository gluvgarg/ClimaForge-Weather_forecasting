import numpy as np


class WindowGenerator:
    """Generate chronological supervised windows from a univariate series."""

    def make_windows(self, values, window_size=3, horizon=1):
        values = np.asarray(values, dtype=float)
        X, y = [], []
        max_start = len(values) - window_size - horizon + 1
        for start in range(max(0, max_start)):
            end = start + window_size
            X.append(values[start:end])
            y.append(values[end + horizon - 1])
        return np.asarray(X, dtype=float), np.asarray(y, dtype=float)

    def chronological_split(self, X, y, train_ratio=0.8):
        if len(X) == 0:
            return X, X, y, y
        split = int(len(X) * train_ratio)
        split = max(1, min(split, len(X) - 1)) if len(X) > 1 else len(X)
        return X[:split], X[split:], y[:split], y[split:]
