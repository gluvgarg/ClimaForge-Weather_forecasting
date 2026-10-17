import numpy as np
import pandas as pd


class TimeSeriesPreprocessor:
    """Clean and scale one-dimensional weather time series."""

    def clean(self, values):
        series = pd.Series(np.asarray(values, dtype=float))
        series = series.interpolate(limit_direction="both")
        series = series.ffill().bfill()
        return series.to_numpy(dtype=float)

    def smooth(self, values, window=1):
        values = np.asarray(values, dtype=float)
        if window <= 1 or values.size < window:
            return values
        return pd.Series(values).rolling(window=window, min_periods=1, center=True).mean().to_numpy()

    def scale(self, values):
        values = np.asarray(values, dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std == 0:
            std = 1.0
        return (values - mean) / std, {"mean": mean, "std": std}

    def inverse_scale(self, values, state):
        return np.asarray(values, dtype=float) * state["std"] + state["mean"]
