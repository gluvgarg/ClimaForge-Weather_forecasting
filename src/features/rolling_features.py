import numpy as np
import xarray as xr


class RollingFeatureGenerator:
    """
    Generate rolling statistical features for
    meteorological variables.

    Rolling statistics are computed only along
    the temporal dimension.
    """

    TIME_DIMENSIONS = {
        "time",
        "valid_time",
        "forecast_time",
        "TIME",
    }

    WINDOW = 3

    EXCLUDED_VARIABLES = {

        "month",
        "day_of_month",
        "day_of_year",
        "week_of_year",
        "quarter",
        "hour",
        "season",

    }

    def process(self, dataset: xr.Dataset, report):

        featured = dataset.copy(deep=True)

        time_dim = self._get_time_dimension(featured)

        if time_dim is None:
            raise ValueError("No valid time dimension found.")

        generated = []

        for variable_name, variable in featured.data_vars.items():

            if variable_name in self.EXCLUDED_VARIABLES:
                continue

            if not np.issubdtype(variable.dtype, np.number):
                continue

            if time_dim not in variable.dims:
                continue

            rolling = variable.rolling(
                {time_dim: self.WINDOW},
                min_periods=1,
            )

            featured[f"{variable_name}_roll_mean"] = rolling.mean()

            featured[f"{variable_name}_roll_std"] = rolling.std()

            featured[f"{variable_name}_roll_min"] = rolling.min()

            featured[f"{variable_name}_roll_max"] = rolling.max()

            featured[f"{variable_name}_roll_mean"].attrs.update({
                "long_name": f"{variable_name} Rolling Mean",
                "window": self.WINDOW,
            })

            featured[f"{variable_name}_roll_std"].attrs.update({
                "long_name": f"{variable_name} Rolling Std",
                "window": self.WINDOW,
            })

            featured[f"{variable_name}_roll_min"].attrs.update({
                "long_name": f"{variable_name} Rolling Min",
                "window": self.WINDOW,
            })

            featured[f"{variable_name}_roll_max"].attrs.update({
                "long_name": f"{variable_name} Rolling Max",
                "window": self.WINDOW,
            })

            generated.extend([
                f"{variable_name}_roll_mean",
                f"{variable_name}_roll_std",
                f"{variable_name}_roll_min",
                f"{variable_name}_roll_max",
            ])

        report.add(
            "rolling_features",
            "generated_features",
            generated,
        )

        return featured

    def _get_time_dimension(self, dataset):

        for dim in dataset.dims:

            if dim.lower() in self.TIME_DIMENSIONS:
                return dim

        return None