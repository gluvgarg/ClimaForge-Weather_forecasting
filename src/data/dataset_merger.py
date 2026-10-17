import numpy as np
import xarray as xr


class DatasetMerger:
    """Merge multiple same-grid yearly xarray Datasets into one, preserving
    coordinates, metadata, and timestamps, before the merged Dataset enters
    the existing Preprocessing/Analysis/Feature Engineering pipeline."""

    TIME_NAMES = {"time", "valid_time", "forecast_time", "TIME"}
    LAT_NAMES = {"lat", "latitude", "LATITUDE"}
    LON_NAMES = {"lon", "longitude", "LONGITUDE"}

    def merge(self, datasets, source_paths):
        if not datasets:
            raise ValueError("No datasets provided to merge.")
        if len(datasets) == 1:
            return datasets[0]

        time_dim = self._shared_dimension(datasets[0], self.TIME_NAMES)
        if time_dim is None:
            raise ValueError("No shared time dimension found for merging.")

        reference = datasets[0]
        lat_name = self._coord_name(reference, self.LAT_NAMES)
        lon_name = self._coord_name(reference, self.LON_NAMES)

        for dataset, path in zip(datasets[1:], source_paths[1:]):
            self._validate_grid_match(reference, dataset, lat_name, lon_name, path)

        merged = xr.concat(datasets, dim=time_dim, join="exact")

        if time_dim in merged.coords:
            merged = merged.sortby(time_dim)
            _, unique_index = np.unique(merged[time_dim].values, return_index=True)
            if len(unique_index) != merged.sizes[time_dim]:
                merged = merged.isel({time_dim: unique_index})

        merged.attrs["merged_from"] = [str(path) for path in source_paths]

        return merged

    def _shared_dimension(self, dataset, names):
        lowered = {name.lower() for name in names}
        for dim in dataset.dims:
            if dim.lower() in lowered:
                return dim
        return None

    def _coord_name(self, dataset, names):
        lowered = {name.lower() for name in names}
        for coord_name in dataset.coords:
            if coord_name.lower() in lowered:
                return coord_name
        return None

    def _validate_grid_match(self, reference, dataset, lat_name, lon_name, path):
        for coord_name in (lat_name, lon_name):
            if coord_name is None:
                continue
            if coord_name not in dataset.coords:
                raise ValueError(
                    f"'{path}' is missing coordinate '{coord_name}' required to merge "
                    "with the reference dataset."
                )
            if not np.array_equal(reference[coord_name].values, dataset[coord_name].values):
                raise ValueError(
                    f"'{path}' has a different '{coord_name}' grid than the reference "
                    "dataset - yearly datasets must share the same spatial grid to be merged."
                )
