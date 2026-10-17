import xarray as xr


class TargetGenerator:
    """
    Generate rainfall prediction target.

    Target:
        rainfall at next forecast timestep.

    Example:
        t  -> rainfall(t+1)
    """

    TIME_DIMENSIONS = {
        "time",
        "valid_time",
        "forecast_time",
        "TIME",
    }

    RAINFALL_VARIABLES = {
        "RAINFALL",
        "rainfall",
        "precipitation",
        "precip",
        "tp",
    }

    def process(self, dataset: xr.Dataset, report):

        featured = dataset.copy(deep=True)

        time_dim = self._get_time_dimension(featured)

        if time_dim is None:
            raise ValueError("No valid time dimension found.")

        rainfall_variable = self._get_rainfall_variable(featured)

        if rainfall_variable is None:

            report.add(
                "target_generation",
                "status",
                "Rainfall variable not found.",
            )

            return featured

        featured["rainfall_target_24h"] = (
            featured[rainfall_variable]
            .shift({time_dim: -1})
        )

        featured["rainfall_target_24h"].attrs.update({

            "long_name": "Rainfall Target (+1 timestep)",

            "description": "Forecast target generated from rainfall shifted one timestep ahead.",

        })

        report.add(
            "target_generation",
            "generated_target",
            "rainfall_target_24h",
        )

        return featured

    def _get_time_dimension(self, dataset):

        for dim in dataset.dims:

            if dim.lower() in self.TIME_DIMENSIONS:
                return dim

        return None

    def _get_rainfall_variable(self, dataset):

        for variable in dataset.data_vars:

            if variable in self.RAINFALL_VARIABLES:
                return variable

        return None