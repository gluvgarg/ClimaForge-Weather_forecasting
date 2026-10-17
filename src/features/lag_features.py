import numpy as np
import xarray as xr


class LagFeatureGenerator:
    """
    Generate lag features for meteorological variables.

    Lag features are generated only along the
    temporal dimension.
    """

    TIME_DIMENSIONS = {
        "time",
        "valid_time",
        "forecast_time",
        "TIME",
    }

    LAGS = [1, 3, 7]

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

            for lag in self.LAGS:

                new_name = f"{variable_name}_lag{lag}"

                featured[new_name] = variable.shift(
                    {time_dim: lag}
                )

                featured[new_name].attrs.update({
                    "long_name": f"{variable_name} Lag {lag}",
                    "lag_steps": lag,
                })

                generated.append(new_name)

        report.add(
            "lag_features",
            "generated_features",
            generated,
        )

        return featured

    def _get_time_dimension(self, dataset):

        for dim in dataset.dims:

            if dim.lower() in self.TIME_DIMENSIONS:
                return dim

        return None