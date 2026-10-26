import numpy as np
import pandas as pd
import xarray as xr


class TimestampValidator:
    """
    Validate and clean temporal coordinates.

    Responsibilities
    ----------------
    1. Detect time coordinate.
    2. Sort timestamps.
    3. Remove duplicate timestamps.
    4. Analyse temporal continuity.
    """

    TIME_NAMES = {
        "time",
        "valid_time",
        "forecast_time",
    }

    def process(self, dataset: xr.Dataset, report):
        cleaned = dataset.copy(deep=True)
        time_dim = self._find_time_dimension(cleaned)
        if time_dim is None:
            report.add(
                "timestamps",
                "status",
                {
                    "found": False,
                    "message": "No time coordinate found."
                },
            )

            return cleaned

        time_values = pd.to_datetime(cleaned[time_dim].values)
        before_count = len(time_values)

        if before_count == 0:
            report.add(
                "timestamps",
                "analysis",
                {
                    "found": True,
                    "time_dimension": time_dim,
                    "timestamps_before": 0,
                    "timestamps_after": 0,
                    "duplicates_removed": 0,
                    "sorted": True,
                    "frequency": "Unknown",
                    "start": "",
                    "end": "",
                },
            )
            return cleaned

        sorted_flag = bool(np.all(time_values[:-1] <= time_values[1:]))

        if not sorted_flag:

            cleaned = cleaned.sortby(time_dim)

            time_values = pd.to_datetime(cleaned[time_dim].values)
 

        duplicated = pd.Index(time_values).duplicated()

        duplicate_count = int(duplicated.sum())

        if duplicate_count > 0:

            cleaned = cleaned.isel(
                {
                    time_dim: ~duplicated
                }
            )

            time_values = pd.to_datetime(
                cleaned[time_dim].values
            )

        frequency = self._infer_frequency(time_values)

        # Preserve metadata
        cleaned.attrs = dict(dataset.attrs)
        for var_name in cleaned.data_vars:
            if var_name in dataset.data_vars:
                cleaned[var_name].attrs = dict(dataset[var_name].attrs)
        for coord_name in cleaned.coords:
            if coord_name in dataset.coords:
                cleaned[coord_name].attrs = dict(dataset[coord_name].attrs)

        report.add(
            "timestamps",
            "analysis",
            {
                "found": True,
                "time_dimension": time_dim,
                "timestamps_before": before_count,
                "timestamps_after": len(time_values),
                "duplicates_removed": duplicate_count,
                "sorted": sorted_flag,
                "frequency": frequency,
                "start": str(time_values[0]) if len(time_values) else "",
                "end": str(time_values[-1]) if len(time_values) else "",
            },
        )

        return cleaned
 

    def _find_time_dimension(self, dataset):

        for coord in dataset.coords:

            if coord.lower() in self.TIME_NAMES:

                return coord

        return None
 

    def _infer_frequency(self, time_values):

        try:

            freq = pd.infer_freq(time_values)

            if freq is None:

                return "Irregular"

            return freq

        except Exception:

            return "Unknown"