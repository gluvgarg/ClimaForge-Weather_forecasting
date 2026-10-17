import numpy as np


class SpatialTemporalAligner:
    """
    Standardize spatial and temporal coordinate ordering.

    Responsibilities
    ----------------
    1. Sort time coordinate.
    2. Sort latitude coordinate.
    3. Sort longitude coordinate.
    4. Reorder dimensions into a consistent order.

    This stage does NOT:
    - Reproject coordinates
    - Change spatial resolution
    - Resample data
    """

    TIME_NAMES = {"time", "valid_time", "forecast_time"}
    LAT_NAMES = {"lat", "latitude", "y"}
    LON_NAMES = {"lon", "longitude", "x"}

    def process(self, dataset, report):

        cleaned = dataset.copy(deep=True)

        report_data = {
            "time_modified": False,
            "latitude_modified": False,
            "longitude_modified": False,
            "dimensions_reordered": False,
            "final_dimension_order": [],
        }

        time_dim = self._find_dimension(cleaned, self.TIME_NAMES)
        lat_dim = self._find_dimension(cleaned, self.LAT_NAMES)
        lon_dim = self._find_dimension(cleaned, self.LON_NAMES) 

        if time_dim is not None:
            values = cleaned[time_dim].values
            if not np.all(values[:-1] <= values[1:]):
                cleaned = cleaned.sortby(time_dim)
                report_data["time_modified"] = True
 

        if lat_dim is not None:
            values = cleaned[lat_dim].values
            if not np.all(values[:-1] <= values[1:]):
                cleaned = cleaned.sortby(lat_dim)
                report_data["latitude_modified"] = True
 

        if lon_dim is not None:
            values = cleaned[lon_dim].values
            if not np.all(values[:-1] <= values[1:]):
                cleaned = cleaned.sortby(lon_dim)
                report_data["longitude_modified"] = True  

        desired_order = []
        if time_dim:
            desired_order.append(time_dim)
        if lat_dim:
            desired_order.append(lat_dim)
        if lon_dim:
            desired_order.append(lon_dim)

        for dim in cleaned.dims:

            if dim not in desired_order:

                desired_order.append(dim)

        if tuple(cleaned.dims) != tuple(desired_order):

            cleaned = cleaned.transpose(*desired_order)

            report_data["dimensions_reordered"] = True

        report_data["final_dimension_order"] = list(cleaned.dims)

        report.add(
            "alignment",
            "analysis",
            report_data,
        )

        return cleaned
 

    def _find_dimension(self, dataset, aliases):
        for dim in dataset.dims:
            if dim.lower() in aliases:
                return dim
        for coord in dataset.coords:
            if coord.lower() in aliases:
                return coord

        return None