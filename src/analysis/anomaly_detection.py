import numpy as np


class AnomalyDetection:
    """
    Simple anomaly detection using Z-score.
    """

    def __init__(self, threshold=2.0):
        self.threshold = threshold

    # -----------------------------------------------------

    def analyze(self, dataset):

        results = {}

        for variable_name, variable in dataset.data_vars.items():

            try:
                results[variable_name] = self._analyze_variable(variable)

            except Exception as e:

                results[variable_name] = {
                    "status": "Failed",
                    "message": str(e)
                }

        return results

    # -----------------------------------------------------

    def _analyze_variable(self, variable):

        values = np.asarray(variable.values, dtype=float)

        values = values[np.isfinite(values)]

        if len(values) == 0:

            return {
                "status": "No Data"
            }

        mean = np.mean(values)
        std = np.std(values)

        if std == 0:

            return {
                "status": "Constant Data",
                "count": 0,
                "percentage": 0.0
            }

        z_scores = np.abs((values - mean) / std)

        count = int(np.sum(z_scores > self.threshold))

        return {

            "status": "Success",

            "threshold": self.threshold,

            "mean": round(float(mean), 4),

            "std": round(float(std), 4),

            "count": count,

            "percentage": round(
                (count / len(values)) * 100,
                2
            )
        }