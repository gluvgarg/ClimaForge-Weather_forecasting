import numpy as np


class MetadataExtractor:
    """Create GUI-friendly metadata from an xarray Dataset."""

    def extract(self, dataset):
        if dataset is None:
            return {}

        return {
            "dimensions": {
                name: int(size) for name, size in dataset.sizes.items()
            },
            "coordinates": {
                name: {
                    "dims": list(coord.dims),
                    "shape": list(coord.shape),
                    "dtype": str(coord.dtype),
                    "range": self._range(coord),
                    "attrs": dict(coord.attrs),
                }
                for name, coord in dataset.coords.items()
            },
            "variables": {
                name: {
                    "dims": list(variable.dims),
                    "shape": list(variable.shape),
                    "dtype": str(variable.dtype),
                    "units": variable.attrs.get("units", ""),
                    "long_name": variable.attrs.get("long_name", ""),
                    "range": self._range(variable),
                    "statistics": self._statistics(variable),
                    "attrs": dict(variable.attrs),
                }
                for name, variable in dataset.data_vars.items()
            },
            "attributes": dict(dataset.attrs),
        }

    def _range(self, data_array):
        values = self._numeric_values(data_array)
        if values is None or values.size == 0 or np.all(np.isnan(values)):
            raw_values = data_array.values
            if raw_values.size == 0:
                return {"min": "", "max": ""}
            return {"min": str(raw_values.min()), "max": str(raw_values.max())}

        return {
            "min": self._format_number(np.nanmin(values)),
            "max": self._format_number(np.nanmax(values)),
        }

    def _statistics(self, data_array):
        values = self._numeric_values(data_array)
        if values is None or values.size == 0:
            return {
                "count": 0,
                "missing": 0,
                "mean": "N/A",
                "std": "N/A",
                "min": "N/A",
                "max": "N/A",
            }

        return {
            "count": int(np.count_nonzero(~np.isnan(values))),
            "missing": int(np.count_nonzero(np.isnan(values))),
            "mean": self._format_number(np.nanmean(values)),
            "std": self._format_number(np.nanstd(values)),
            "min": self._format_number(np.nanmin(values)),
            "max": self._format_number(np.nanmax(values)),
        }

    def _numeric_values(self, data_array):
        try:
            return np.asarray(data_array.values, dtype=float)
        except (TypeError, ValueError):
            return None

    def _format_number(self, value):
        if np.isnan(value):
            return "NaN"
        return f"{value:.4f}"
