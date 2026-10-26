import numpy as np
import pandas as pd
import xarray as xr


class DuplicateHandler:
    """
    Detect and remove duplicate weather observations.

    Duplicate observations are identified using the spatial-temporal
    coordinates (time, latitude, longitude).

    Attributes, variable units, and coordinate metadata are preserved.
    """

    TIME_COLUMNS = {
        "time",
        "valid_time",
        "forecast_time",
        "TIME",
    }

    LAT_COLUMNS = {
        "latitude",
        "lat",
        "LATITUDE",
    }

    LON_COLUMNS = {
        "longitude",
        "lon",
        "LONGITUDE",
    }

    def process(self, dataset: xr.Dataset, report):
        cleaned = dataset.copy(deep=True)

        coordinate_dims = []
        for dim in cleaned.dims:
            dim_lower = dim.lower()
            if (
                dim_lower in self.TIME_COLUMNS
                or dim_lower in self.LAT_COLUMNS
                or dim_lower in self.LON_COLUMNS
            ):
                coordinate_dims.append(dim)

        rows_before = int(np.prod(list(cleaned.sizes.values()))) if cleaned.sizes else 0
        duplicates_removed = 0

        # Check each coordinate dimension for duplicates
        has_coord_duplicates = False
        for dim in coordinate_dims:
            if dim in cleaned.coords:
                idx = pd.Index(cleaned[dim].values)
                duplicated_mask = idx.duplicated()
                dup_count = int(duplicated_mask.sum())
                if dup_count > 0:
                    has_coord_duplicates = True
                    cleaned = cleaned.isel({dim: ~duplicated_mask})

        if has_coord_duplicates:
            rows_after = int(np.prod(list(cleaned.sizes.values()))) if cleaned.sizes else 0
            duplicates_removed = max(0, rows_before - rows_after)
        else:
            rows_after = rows_before

        percentage = 0.0
        if rows_before > 0:
            percentage = (duplicates_removed / rows_before) * 100

        # Preserve all attributes and units
        cleaned.attrs = dict(dataset.attrs)
        for var_name in cleaned.data_vars:
            if var_name in dataset.data_vars:
                cleaned[var_name].attrs = dict(dataset[var_name].attrs)
        for coord_name in cleaned.coords:
            if coord_name in dataset.coords:
                cleaned[coord_name].attrs = dict(dataset[coord_name].attrs)

        report.add(
            "duplicates",
            "analysis",
            {
                "rows_before": rows_before,
                "rows_after": rows_after,
                "duplicates_removed": duplicates_removed,
                "duplicate_percentage": round(percentage, 4),
                "coordinate_columns": coordinate_dims,
            },
        )

        return cleaned