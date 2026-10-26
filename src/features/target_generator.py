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

        # Determine actual time interval
        interval_desc = "+1 timestep"
        is_24h = False
        if time_dim in featured.coords and len(featured[time_dim]) > 1:
            try:
                import pandas as pd
                times = pd.to_datetime(featured[time_dim].values[:2])
                delta_sec = abs((times[1] - times[0]).total_seconds())
                hours = delta_sec / 3600.0
                if abs(hours - 24.0) < 1e-3:
                    is_24h = True
                    interval_desc = "24h ahead (+1 daily step)"
                elif hours == int(hours):
                    interval_desc = f"{int(hours)}h ahead (+1 step)"
                else:
                    interval_desc = f"{hours:.1f}h ahead (+1 step)"
            except Exception:
                pass

        featured["rainfall_target_24h"] = (
            featured[rainfall_variable]
            .shift({time_dim: -1})
        )

        featured["rainfall_target_24h"].attrs.update({
            "long_name": f"Rainfall Target ({interval_desc})",
            "lead_time": interval_desc,
            "lead_time_steps": 1,
            "is_24h_interval": int(is_24h),
            "units": featured[rainfall_variable].attrs.get("units", "mm"),
            "description": f"Forecast target generated from rainfall shifted 1 timestep ahead ({interval_desc}).",
        })

        report.add(
            "target_generation",
            "generated_target",
            "rainfall_target_24h",
        )
        report.add(
            "target_generation",
            "lead_time",
            interval_desc,
        )
        report.add(
            "target_generation",
            "lead_time_steps",
            1,
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