import numpy as np
import xarray as xr


class TemporalFeatureGenerator:
    """
    Generate temporal features from dataset timestamps.
    """

    TIME_DIMENSIONS = {
        "time",
        "valid_time",
        "forecast_time",
        "TIME",
    }

    def process(self, dataset: xr.Dataset, report):

        featured = dataset.copy(deep=True)

        time_dim = self._get_time_dimension(featured)

        if time_dim is None:
            raise ValueError("No valid time dimension found.")

        # Convert to pandas DatetimeIndex
        time = featured[time_dim].to_index()

        featured["month"] = (
            time_dim,
            time.month.to_numpy(),
        )

        featured["day_of_month"] = (
            time_dim,
            time.day.to_numpy(),
        )

        featured["day_of_year"] = (
            time_dim,
            time.dayofyear.to_numpy(),
        )

        featured["week_of_year"] = (
            time_dim,
            time.isocalendar().week.astype(int).to_numpy(),
        )

        featured["quarter"] = (
            time_dim,
            time.quarter.to_numpy(),
        )

        featured["hour"] = (
            time_dim,
            time.hour.to_numpy(),
        )

        season = np.select(
            [
                np.isin(time.month, [12, 1, 2]),
                np.isin(time.month, [3, 4, 5]),
                np.isin(time.month, [6, 7, 8, 9]),
            ],
            [
                0,   # Winter
                1,   # Pre-Monsoon
                2,   # Monsoon
            ],
            default=3,   # Post-Monsoon
        )

        featured["season"] = (
            time_dim,
            season,
        )

        report.add(
            "temporal_features",
            "generated_features",
            [
                "month",
                "day_of_month",
                "day_of_year",
                "week_of_year",
                "quarter",
                "hour",
                "season",
            ],
        )

        return featured

    def _get_time_dimension(self, dataset):

        for dim in dataset.dims:

            if dim.lower() in self.TIME_DIMENSIONS:
                return dim

        return None