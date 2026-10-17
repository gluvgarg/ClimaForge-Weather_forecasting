import numpy as np


class DataScaler:
    """
    Scale numeric meteorological variables.

    Supported methods
    -----------------
    - standard : (x - mean) / std
    - minmax   : (x - min) / (max - min)

    Coordinates are never scaled.
    """

    EXCLUDED_VARIABLES = {
        "TIME",
        "time",
        "valid_time",
        "forecast_time",
        "LATITUDE",
        "latitude",
        "lat",
        "LONGITUDE",
        "longitude",
        "lon",
    }

    def __init__(self, method="standard"):

        self.method = method.lower()

    def process(self, dataset, report):

        scaled = dataset.copy(deep=True)

        analysis = {}

        for variable_name in scaled.data_vars:

            if variable_name in self.EXCLUDED_VARIABLES:
                continue

            variable = scaled[variable_name]

            if not np.issubdtype(variable.dtype, np.number):
                continue

            values = variable.values.astype(float)

            if self.method == "standard":

                scaled_values, info = self._standard_scale(values)

            elif self.method == "minmax":

                scaled_values, info = self._minmax_scale(values)

            else:

                raise ValueError(
                    f"Unsupported scaling method: {self.method}"
                )

            scaled[variable_name].values = scaled_values

            scaled[variable_name].attrs["scaled"] = True
            scaled[variable_name].attrs["scaling_method"] = self.method

            analysis[variable_name] = info

        report.add(
            "scaling",
            "analysis",
            analysis,
        )

        return scaled

    def _standard_scale(self, values):

        mean = np.nanmean(values)

        std = np.nanstd(values)

        if std == 0 or np.isnan(std):

            return values, {
                "method": "standard",
                "scaled": False,
                "reason": "Zero standard deviation",
            }

        scaled = (values - mean) / std

        return scaled, {

            "method": "standard",

            "scaled": True,

            "mean": float(mean),

            "std": float(std),

        }

    def _minmax_scale(self, values):

        minimum = np.nanmin(values)

        maximum = np.nanmax(values)

        denominator = maximum - minimum

        if denominator == 0 or np.isnan(denominator):

            return values, {
                "method": "minmax",
                "scaled": False,
                "reason": "Zero data range",
            }

        scaled = (values - minimum) / denominator

        return scaled, {

            "method": "minmax",

            "scaled": True,

            "minimum": float(minimum),

            "maximum": float(maximum),

        }