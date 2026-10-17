import numpy as np


class TrendAnalysis:
    """
    Performs temporal trend analysis on weather variables.

    Features
    --------
    1. Temporal mean series
    2. Moving average
    3. Linear trend estimation
    4. Trend direction
    """

    TIME_NAMES = {
        "time",
        "valid_time",
        "forecast_time",
    }

    def analyze(self, dataset):

        report = {}

        for variable_name, variable in dataset.data_vars.items():

            try:
                report[variable_name] = self._analyze_variable(variable)

            except Exception as e:

                report[variable_name] = {
                    "status": "Failed",
                    "message": str(e),
                }

        return report
 

    def _analyze_variable(self, variable):

        time_dim = self._find_time_dimension(variable)

        if time_dim is None:

            return {
                "status": "No Time Dimension",
                "trend": "Unavailable",
            }

        axis = variable.get_axis_num(time_dim)

        values = np.asarray(variable.values, dtype=float)

        # Spatial mean for every timestamp
        series = np.nanmean(
            values,
            axis=tuple(
                i
                for i in range(values.ndim)
                if i != axis
            ),
        )

        series = np.asarray(series, dtype=float)

        if series.size < 2:

            return {
                "status": "Insufficient Data",
                "trend": "Unavailable",
                "series": series.tolist(),
                "moving_average": [],
                "slope": 0.0,
            }

        moving_average = self._moving_average(series)

        slope = self._trend_slope(series)

        direction = self._trend_direction(slope)

        return {

            "status": "Success",

            "trend": direction,

            "slope": round(float(slope), 6),

            "series": [
                round(float(v), 4)
                for v in series
            ],

            "moving_average": [
                round(float(v), 4)
                for v in moving_average
            ],
        }
 

    def _moving_average(self, series, window=3):

        window = min(window, len(series))

        kernel = np.ones(window) / window

        return np.convolve(
            series,
            kernel,
            mode="same",
        )
 

    def _trend_slope(self, series):

        x = np.arange(len(series))

        slope, _ = np.polyfit(
            x,
            series,
            1,
        )

        return slope
 

    def _trend_direction(self, slope):

        if slope > 0.001:
            return "Increasing"

        elif slope < -0.001:
            return "Decreasing"

        return "Stable"
 

    def _find_time_dimension(self, variable):

        for dim in variable.dims:

            if dim.lower() in self.TIME_NAMES:
                return dim

        return None