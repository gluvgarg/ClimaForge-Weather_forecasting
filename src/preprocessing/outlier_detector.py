import numpy as np

from src.data.weather_variables import WeatherVariableNormalizer


class OutlierHandler:
    """
    Detect and optionally treat statistical outliers.

    Parameters
    ----------
    action : str
        "flag"   -> Detect only (default)
        "remove" -> Replace outliers with NaN
        "clip"   -> Clip outliers to lower/upper thresholds

    Notes
    -----
    The default behaviour is 'flag' because weather extremes
    often represent real meteorological events.
    """

    OUTLIER_CONFIG = {

        "temperature": {
            "method": "IQR",
            "multiplier": 1.5,
        },

        "pressure": {
            "method": "IQR",
            "multiplier": 2.0,
        },

        "humidity": {
            "method": "IQR",
            "multiplier": 2.0,
        },

        "wind": {
            "method": "IQR",
            "multiplier": 3.0,
        },

        "precipitation": {
            "method": "IQR",
            "multiplier": 5.0,
        },

        "generic": {
            "method": "IQR",
            "multiplier": 1.5,
        },
    }
 

    def __init__(self, action="flag"):

        self.action = action.lower()

        self.normalizer = WeatherVariableNormalizer()
 

    def process(self, dataset, report):

        cleaned = dataset.copy(deep=True)

        report_data = {}

        for variable_name, variable in cleaned.data_vars.items():

            if not np.issubdtype(variable.dtype, np.number):
                continue

            concept = self.normalizer.concept_for_variable(
                variable_name,
                variable,
            )

            config = self.OUTLIER_CONFIG.get(
                concept,
                self.OUTLIER_CONFIG["generic"],
            )

            multiplier = config["multiplier"]

            values = variable.values.astype(float)

            valid_values = values[np.isfinite(values)]

            if valid_values.size == 0:
                continue

            q1 = np.percentile(valid_values, 25)

            q3 = np.percentile(valid_values, 75)

            iqr = q3 - q1

            lower = q1 - multiplier * iqr

            upper = q3 + multiplier * iqr

            mask = (
                (variable < lower)
                |
                (variable > upper)
            )

            outlier_count = int(mask.sum())

            percentage = (
                outlier_count / valid_values.size
            ) * 100

            severity = self._severity(
                percentage
            )

            recommendation = self._recommendation(
                concept,
                severity,
                outlier_count,
            )

            # --------------------------------------------------
            # Optional treatment
            # --------------------------------------------------

            if self.action == "remove":

                variable = variable.where(~mask)

            elif self.action == "clip":

                variable = variable.clip(
                    min=lower,
                    max=upper,
                )

            # flag -> no modification

            cleaned[variable_name] = variable

            report_data[variable_name] = {

                "concept": concept,

                "method": config["method"],

                "action": self.action,

                "multiplier": multiplier,

                "minimum": float(np.min(valid_values)),

                "maximum": float(np.max(valid_values)),

                "mean": float(np.mean(valid_values)),

                "median": float(np.median(valid_values)),

                "std": float(np.std(valid_values)),

                "q1": float(q1),

                "q3": float(q3),

                "iqr": float(iqr),

                "lower_threshold": float(lower),

                "upper_threshold": float(upper),

                "outlier_count": outlier_count,

                "outlier_percentage": round(
                    percentage,
                    4,
                ),

                "severity": severity,

                "recommendation": recommendation,
            }

        report.add(
            "outliers",
            "analysis",
            report_data,
        )

        return cleaned
 

    def _severity(self, percentage):

        if percentage < 1:

            return "Low"

        elif percentage < 5:

            return "Moderate"

        return "High"
 

    def _recommendation(self, concept, severity, outlier_count):

        if outlier_count == 0:

            return "No statistical outliers detected."

        if concept == "precipitation":

            return (
                "Extreme rainfall values may represent "
                "real weather events. Review before "
                "removing or clipping."
            )

        if concept == "pressure":

            return (
                "Pressure values exceed statistical "
                "thresholds. Verify against physical "
                "limits before modification."
            )

        if concept == "wind":

            return (
                "Extreme wind speeds may occur during "
                "storms or cyclones. Review before "
                "modification."
            )

        if severity == "Low":

            return "No action required."

        elif severity == "Moderate":

            return (
                "Review flagged values before "
                "feature engineering."
            )

        return (
            "Large number of statistical outliers. "
            "Investigate dataset before modelling."
        )