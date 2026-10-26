import numpy as np
import xarray as xr


class MissingValueHandler:
    """
    Analyse and handle missing values in weather datasets.

    1. Analyse missing values.
    2. Preserve permanently masked spatial regions.
    3. Repair only valid temporal gaps.
    4. Generate preprocessing statistics.

    Notes
    - Grid cells that are completely NaN across the entire time axis are
      considered permanent spatial masks and are NOT modified.
    - Temporal interpolation is applied only to valid time series.
    """

    TIME_DIMENSIONS = {
        "time",
        "valid_time",
        "forecast_time",
    }

    def process(self, dataset: xr.Dataset, report):
        """
        Process all numeric variables in the dataset.
        Parameters 
        dataset : xr.Dataset
            Input weather dataset.
        report : PreprocessingSummary
            Shared preprocessing report object.
        Returns 
        xr.Dataset
            Cleaned dataset.
        """

        cleaned = dataset.copy(deep=True)
        report_data = {}

        for variable_name in cleaned.data_vars:
            variable = cleaned[variable_name]
            # Skip non-numeric variables
            if not np.issubdtype(variable.dtype, np.number):
                continue
            missing_before = int(variable.isnull().sum())
            repaired_values = 0
            permanent_mask = 0
            method = "none"

            if missing_before > 0:
                if self._has_time_dimension(variable):
                    variable, repaired_values, permanent_mask = self._repair_temporal_gaps(variable)

                    if repaired_values > 0 and permanent_mask > 0:
                        method = "temporal interpolation + preserved permanent spatial mask"
                    elif repaired_values > 0:
                        method = "temporal interpolation"
                    elif permanent_mask > 0:
                        method = "preserved permanent spatial mask"
                    else:
                        method = "none"

                else:
                    variable = self._fill_mean(variable)
                    repaired_values = missing_before - int(variable.isnull().sum())
                    method = "mean" if repaired_values > 0 else "none"

            missing_after = int(variable.isnull().sum())
            cleaned[variable_name] = variable
            cleaned[variable_name].attrs = dict(dataset[variable_name].attrs)

            if variable.size > 0:
                missing_percentage = round(
                    (missing_before / variable.size) * 100,
                    2,
                )

            else:
                missing_percentage = 0.0

            report_data[variable_name] = {
                "missing_before": missing_before,
                "missing_after": missing_after,
                "repaired_values": repaired_values,
                "permanent_mask": permanent_mask,
                "missing_percentage": missing_percentage,
                "method": method,
            }

        cleaned.attrs = dict(dataset.attrs)

        report.add(
            "missing_values",
            "analysis",
            report_data,
        )
        return cleaned 

    def _has_time_dimension(self, variable):

        return any(
            dim.lower() in self.TIME_DIMENSIONS
            for dim in variable.dims
        ) 

    def _repair_temporal_gaps(self, variable):
        """
        Interpolate only valid temporal gaps while preserving
        permanently masked grid cells.
        """

        time_dim = next(
            dim
            for dim in variable.dims
            if dim.lower() in self.TIME_DIMENSIONS
        )

        # Grid cells that contain NaN for every timestamp
        permanent_mask = variable.isnull().all(dim=time_dim)

        permanent_values = (
            int(permanent_mask.sum())
            * variable.sizes[time_dim]
        )

        if variable.sizes[time_dim] < 2:
            return variable, 0, permanent_values

        var_to_interp = variable
        if time_dim in var_to_interp.coords:
            coord_vals = var_to_interp[time_dim].values
            import pandas as pd
            idx = pd.Index(coord_vals)
            if idx.duplicated().any():
                var_to_interp = var_to_interp.isel({time_dim: ~idx.duplicated()})
            if len(var_to_interp[time_dim]) > 1:
                vals = var_to_interp[time_dim].values
                try:
                    if not np.all(vals[:-1] <= vals[1:]):
                        var_to_interp = var_to_interp.sortby(time_dim)
                except Exception:
                    pass

        try:
            # Interpolate temporal gaps
            interpolated = var_to_interp.interpolate_na(
                dim=time_dim,
                method="linear",
            )
        except Exception:
            # Fallback to ffill and bfill along time dimension if linear interp fails
            interpolated = var_to_interp.ffill(dim=time_dim).bfill(dim=time_dim)

        # Restore permanently masked locations
        interpolated = interpolated.where(
            ~permanent_mask,
            np.nan,
        )

        repaired_values = max(
            0,
            int(variable.isnull().sum())
            - int(interpolated.isnull().sum())
        )

        interpolated.attrs = dict(variable.attrs)

        return (
            interpolated,
            repaired_values,
            permanent_values,
        )
 

    def _fill_mean(self, variable):
        """
        Fill missing values using mean.
        Used only when no time dimension exists.
        """

        mean_value = variable.mean(skipna=True)

        return variable.fillna(mean_value)