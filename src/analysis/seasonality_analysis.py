import numpy as np
import pandas as pd


class SeasonalityAnalysis:
    """
    Performs basic seasonal analysis on weather variables.
    """

    TIME_NAMES = {
        "time",
        "valid_time",
        "forecast_time",
    }

    def analyze(self, dataset):

        report = {}

        time_dim = self._find_time_dimension(dataset)

        if time_dim is None:
            return {
                "status": "No Time Dimension"
            }

        timestamps = pd.to_datetime(dataset[time_dim].values)

        for variable_name, variable in dataset.data_vars.items():

            try:
                report[variable_name] = self._analyze_variable(
                    variable,
                    timestamps,
                    time_dim,
                )

            except Exception as e:

                report[variable_name] = {
                    "status": "Failed",
                    "message": str(e),
                }

        return report
 

    def _analyze_variable(
        self,
        variable,
        timestamps,
        time_dim,
    ):

        axis = variable.get_axis_num(time_dim)

        values = np.asarray(variable.values, dtype=float)

        series = np.nanmean(
            values,
            axis=tuple(
                i
                for i in range(values.ndim)
                if i != axis
            ),
        )

        dataframe = pd.DataFrame({
            "time": timestamps,
            "value": series,
        })

        monthly = (
            dataframe
            .groupby(dataframe["time"].dt.month)["value"]
            .mean()
            .to_dict()
        )

        daily = (
            dataframe
            .groupby(dataframe["time"].dt.day)["value"]
            .mean()
            .to_dict()
        )

        hourly = (
            dataframe
            .groupby(dataframe["time"].dt.hour)["value"]
            .mean()
            .to_dict()
        )

        return {

            "status": "Success",

            "observations": len(series),

            "monthly_pattern": {
                int(k): round(float(v), 4)
                for k, v in monthly.items()
            },

            "daily_pattern": {
                int(k): round(float(v), 4)
                for k, v in daily.items()
            },

            "hourly_pattern": {
                int(k): round(float(v), 4)
                for k, v in hourly.items()
            },
        }
 

    def _find_time_dimension(self, dataset):

        for dim in dataset.dims:

            if dim.lower() in self.TIME_NAMES:
                return dim

        return None