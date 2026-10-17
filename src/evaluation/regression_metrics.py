import numpy as np


class ForecastEvaluator:
    """Evaluate forecast outputs with baseline regression metrics."""

    def evaluate(self, actual, predicted):
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        if actual.size == 0 or predicted.size == 0:
            return {"mse": "N/A", "rmse": "N/A", "mae": "N/A", "mape": "N/A", "r2": "N/A"}
        error = predicted - actual
        mse = float(np.mean(error**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(error)))

        nonzero = actual != 0
        if np.any(nonzero):
            mape = float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100)
            mape_text = f"{mape:.4f}"
        else:
            mape_text = "N/A"

        total_variance = float(np.sum((actual - np.mean(actual)) ** 2))
        if total_variance == 0:
            r2_text = "N/A"
        else:
            r2 = 1.0 - float(np.sum(error**2)) / total_variance
            r2_text = f"{r2:.4f}"

        return {
            "mse": f"{mse:.4f}",
            "rmse": f"{rmse:.4f}",
            "mae": f"{mae:.4f}",
            "mape": mape_text,
            "r2": r2_text,
        }
