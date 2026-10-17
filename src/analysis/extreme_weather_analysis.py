import numpy as np


class ExtremeWeatherAnalysis:
    """
    Detect extreme weather events using threshold-based analysis.
    """

    THRESHOLDS = {
        "temperature": {
            "high": 310.0,     # Kelvin (~37°C)
            "low": 273.15      # 0°C
        },

        "pressure": {
            "high": 1025.0,
            "low": 980.0
        },

        "humidity": {
            "high": 90.0,
            "low": 20.0
        },

        "wind_speed": {
            "high": 15.0
        },

        "precipitation": {
            "high": 20.0
        }
    }

    def analyze(self, dataset):

        report = {}

        for variable_name, variable in dataset.data_vars.items():

            report[variable_name] = self._analyze_variable(
                variable_name,
                variable,
            )

        return report

    # --------------------------------------------------

    def _analyze_variable(self, name, variable):

        values = np.asarray(variable.values, dtype=float)

        values = values[np.isfinite(values)]

        if values.size == 0:

            return {
                "status": "No Data"
            }

        info = {

            "status": "Success",

            "minimum": round(float(np.min(values)), 4),

            "maximum": round(float(np.max(values)), 4),

            "mean": round(float(np.mean(values)), 4),

            "extreme_events": []
        }

        threshold = self.THRESHOLDS.get(name)

        if threshold is None:

            return info

        if "high" in threshold:

            count = int(np.sum(values > threshold["high"]))

            info["extreme_events"].append({

                "type": "High",

                "threshold": threshold["high"],

                "count": count
            })

        if "low" in threshold:

            count = int(np.sum(values < threshold["low"]))

            info["extreme_events"].append({

                "type": "Low",

                "threshold": threshold["low"],

                "count": count
            })

        return info