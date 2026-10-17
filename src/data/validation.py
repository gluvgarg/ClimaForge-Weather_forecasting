import numpy as np


class DatasetValidator:
    """Validate whether an xarray Dataset can support scientific weather workflows."""

    LAT_NAMES = {"lat", "latitude", "y", "nav_lat"}
    LON_NAMES = {"lon", "longitude", "x", "nav_lon"}
    TIME_NAMES = {"time", "valid_time", "forecast_time"}

    def validate(self, dataset):
        issues = []
        if dataset is None:
            return {"valid": False, "errors": ["No dataset loaded."], "warnings": []}

        if not dataset.data_vars:
            issues.append(("error", "Dataset contains no data variables."))

        lat_name = self._coord_name(dataset, self.LAT_NAMES, "latitude", "degrees_north")
        lon_name = self._coord_name(dataset, self.LON_NAMES, "longitude", "degrees_east")
        time_name = self._time_coord_name(dataset)

        if lat_name is None:
            issues.append(("error", "Missing latitude coordinate."))
        if lon_name is None:
            issues.append(("error", "Missing longitude coordinate."))
        if time_name is None:
            issues.append(("warning", "No explicit time coordinate found; static map workflows only."))

        for name, variable in dataset.data_vars.items():
            if variable.size == 0:
                issues.append(("error", f"Variable '{name}' is empty."))
                continue
            if lat_name and lat_name not in variable.dims and lat_name not in variable.coords:
                issues.append(("warning", f"Variable '{name}' is not directly georeferenced by latitude."))
            if lon_name and lon_name not in variable.dims and lon_name not in variable.coords:
                issues.append(("warning", f"Variable '{name}' is not directly georeferenced by longitude."))
            if "units" not in variable.attrs:
                issues.append(("warning", f"Variable '{name}' is missing units metadata."))
            try:
                values = np.asarray(variable.values, dtype=float)
                if values.size and np.all(np.isnan(values)):
                    issues.append(("error", f"Variable '{name}' contains only missing values."))
            except (TypeError, ValueError):
                issues.append(("warning", f"Variable '{name}' is non-numeric and cannot be mapped."))

        errors = [message for level, message in issues if level == "error"]
        warnings = [message for level, message in issues if level == "warning"]
        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "lat_name": lat_name or "",
            "lon_name": lon_name or "",
            "time_name": time_name or "",
        }

    def _coord_name(self, dataset, names, standard_name, units):
        for coord_name, coord in dataset.coords.items():
            lowered = coord_name.lower()
            coord_standard = coord.attrs.get("standard_name", "").lower()
            coord_units = coord.attrs.get("units", "").lower()
            if lowered in names or coord_standard == standard_name or coord_units == units:
                return coord_name
        return None

    def _time_coord_name(self, dataset):
        for coord_name in dataset.coords:
            if coord_name.lower() in self.TIME_NAMES:
                return coord_name
        return None
