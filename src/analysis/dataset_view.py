import numpy as np

from src.data.weather_variables import WeatherVariableNormalizer


class DatasetViewBuilder:
    """Builds view-ready scientific data from xarray datasets."""

    LAT_NAMES = {"lat", "latitude", "y", "nav_lat"}
    LON_NAMES = {"lon", "longitude", "x", "nav_lon"}
    TIME_NAMES = {"time", "valid_time", "forecast_time"}
    NORMALIZER = WeatherVariableNormalizer()

    def time_labels(self, dataset):
        if dataset is None:
            return []

        time_coord = self._time_coord(dataset)
        if time_coord is None:
            return []

        return [str(value) for value in time_coord.values]

    def spatial_view(self, dataset, variable_name, time_index=0):
        variable = self._data_array_for(dataset, variable_name)
        sliced = self._time_slice(variable, time_index)
        lat_name, lon_name = self._lat_lon_names(sliced)

        if lat_name is None or lon_name is None:
            raise ValueError("Selected variable does not expose latitude/longitude coordinates.")

        reduced = sliced.squeeze(drop=True)
        reduced = self._lat_lon_ordered(reduced, lat_name, lon_name)
        if reduced.ndim != 2:
            raise ValueError(f"Selected slice must be 2D for map display. Got shape {tuple(reduced.shape)}.")

        lat = np.asarray(reduced[lat_name].values, dtype=float)
        lon = np.asarray(reduced[lon_name].values, dtype=float)
        values = np.asarray(reduced.values, dtype=float)
        timestamp = self._timestamp_for(dataset, time_index)
        units = variable.attrs.get("units", "")
        long_name = variable.attrs.get("long_name", variable_name)

        category = self.variable_category(variable_name, variable)
        view = {
            "variable": variable_name,
            "long_name": long_name,
            "units": units,
            "timestamp": timestamp,
            "time_index": int(time_index),
            "lat_name": lat_name,
            "lon_name": lon_name,
            "lat": lat,
            "lon": lon,
            "values": values,
            "category": category,
            "title": self._title(long_name, units, timestamp),
            "coordinate_ranges": {
                lat_name: {"min": self._format(lat.min()), "max": self._format(lat.max())},
                lon_name: {"min": self._format(lon.min()), "max": self._format(lon.max())},
            },
        }
        if category == "wind":
            view["wind_vectors"] = self.wind_vectors(dataset, time_index, lat_name, lon_name)
        return view

    def variable_statistics(self, dataset, variable_name, time_index=None):
        data_array = self._data_array_for(dataset, variable_name)
        if time_index is not None:
            data_array = self._time_slice(data_array, time_index)

        values = self._numeric_values(data_array)
        if values is None or values.size == 0:
            return {}

        return {
            "count": int(np.count_nonzero(~np.isnan(values))),
            "missing": int(np.count_nonzero(np.isnan(values))),
            "mean": self._format(np.nanmean(values)),
            "std": self._format(np.nanstd(values)),
            "min": self._format(np.nanmin(values)),
            "max": self._format(np.nanmax(values)),
            "units": data_array.attrs.get("units", ""),
            "dims": ", ".join(data_array.dims),
            "shape": " x ".join(str(item) for item in data_array.shape),
        }

    def metric_summary(self, dataset, time_index=0):
        if dataset is None:
            return []

        metrics = []
        normalized = self.NORMALIZER.resolve(dataset)
        for category in self.NORMALIZER.PRIMARY_ORDER:
            concept = normalized[category]
            variable_name = concept.get("scalar")
            if category == "wind" and not variable_name and concept.get("vector_available"):
                variable_name = "wind_speed"
            if not variable_name:
                continue

            variable = self._time_slice(self._data_array_for(dataset, variable_name), time_index)
            values = self._numeric_values(variable)
            if values is None or values.size == 0 or np.all(np.isnan(values)):
                continue

            units = self._data_array_for(dataset, variable_name).attrs.get("units", "")
            metrics.append(
                {
                    "label": concept["label"],
                    "value": self._format(np.nanmean(values)),
                    "units": units,
                    "min": self._format(np.nanmin(values)),
                    "max": self._format(np.nanmax(values)),
                    "variable": variable_name,
                }
            )

        return metrics

    def grid_rows(self, dataset, variable_name, time_index=0, max_rows=500, region_bounds=None):
        view = self.spatial_view(dataset, variable_name, time_index)
        lat = view["lat"]
        lon = view["lon"]
        values = view["values"]
        mean_value = np.nanmean(values)
        rows = []

        if lat.ndim == 1 and lon.ndim == 1:
            for lat_index, lat_value in enumerate(lat):
                for lon_index, lon_value in enumerate(lon):
                    value = values[lat_index, lon_index]
                    if self._inside_region(lat_value, lon_value, region_bounds):
                        rows.append(self._grid_row(view, lat_value, lon_value, value, mean_value))
                    if len(rows) >= max_rows:
                        return rows
        else:
            for flat_index, value in enumerate(np.ravel(values)):
                lat_value = np.ravel(lat)[flat_index]
                lon_value = np.ravel(lon)[flat_index]
                if self._inside_region(lat_value, lon_value, region_bounds):
                    rows.append(self._grid_row(view, lat_value, lon_value, value, mean_value))
                if len(rows) >= max_rows:
                    return rows
        return rows

    def _grid_row(self, view, lat_value, lon_value, value, mean_value):
        return [
            view["timestamp"],
            self._format(lat_value),
            self._format(lon_value),
            self._format(value),
            view["units"],
            self._format(value - mean_value),
        ]

    def comparison_series(self, dataset, preferred_variable=None, max_variables=5):
        if dataset is None:
            return {"time": [], "series": []}

        time_coord = self._time_coord(dataset)
        labels = self.time_labels(dataset)
        series = []
        names = []
        if preferred_variable:
            names.append(preferred_variable)

        for category in self.NORMALIZER.PRIMARY_ORDER:
            variable_name = self.find_variable_for_category(dataset, category)
            if variable_name and variable_name not in names:
                names.append(variable_name)

        for variable_name in list(dataset.data_vars):
            if len(names) >= max_variables:
                break
            if variable_name not in names:
                names.append(variable_name)

        for variable_name in names[:max_variables]:
            data_array = self._data_array_for(dataset, variable_name)
            time_dim = self._time_dim(data_array)
            values = self._numeric_values(data_array)
            if values is None or values.size == 0:
                continue
            if time_dim is None:
                y_values = [float(np.nanmean(values))]
                x_labels = ["static"]
            else:
                axis = data_array.get_axis_num(time_dim)
                reduced = np.nanmean(values, axis=tuple(index for index in range(values.ndim) if index != axis))
                y_values = [float(item) if not np.isnan(item) else np.nan for item in np.ravel(reduced)]
                x_labels = labels if labels else [str(index) for index in range(len(y_values))]

            series.append(
                {
                    "name": variable_name,
                    "label": data_array.attrs.get("long_name", variable_name),
                    "units": data_array.attrs.get("units", ""),
                    "values": y_values,
                }
            )

        return {
            "time": labels if labels else ([str(value) for value in time_coord.values] if time_coord is not None else []),
            "series": series,
        }

    def multi_layer_view(self, dataset, primary_variable, time_index=0, active_layers=None, opacity=None):
        view = self.spatial_view(dataset, primary_variable, time_index)
        active_layers = active_layers or []
        opacity = opacity or {}
        overlays = []
        for concept in active_layers:
            variable_name = self.find_variable_for_category(dataset, concept)
            if not variable_name or variable_name == primary_variable:
                continue
            try:
                overlay = self.spatial_view(dataset, variable_name, time_index)
            except ValueError:
                continue
            overlay["opacity"] = float(opacity.get(concept, 0.45))
            overlays.append(overlay)
        view["overlays"] = overlays
        view["active_layers"] = active_layers
        return view

    def regional_statistics(self, dataset, variable_name, time_index=0, region_bounds=None):
        view = self.spatial_view(dataset, variable_name, time_index)
        lat = view["lat"]
        lon = view["lon"]
        values = view["values"]
        if region_bounds is None:
            selected = values
        elif lat.ndim == 1 and lon.ndim == 1:
            lat_mask = (lat >= region_bounds["lat_min"]) & (lat <= region_bounds["lat_max"])
            lon_mask = (lon >= region_bounds["lon_min"]) & (lon <= region_bounds["lon_max"])
            selected = values[np.ix_(lat_mask, lon_mask)]
        else:
            mask = (
                (lat >= region_bounds["lat_min"])
                & (lat <= region_bounds["lat_max"])
                & (lon >= region_bounds["lon_min"])
                & (lon <= region_bounds["lon_max"])
            )
            selected = values[mask]
        if selected.size == 0:
            return {"region_count": 0}
        return {
            "region_count": int(np.count_nonzero(~np.isnan(selected))),
            "region_mean": self._format(np.nanmean(selected)),
            "region_min": self._format(np.nanmin(selected)),
            "region_max": self._format(np.nanmax(selected)),
            "region_bounds": region_bounds or {},
        }

    def weather_layer_options(self, dataset):
        if dataset is None:
            return []
        normalized = self.NORMALIZER.resolve(dataset)
        layers = []
        for category in self.NORMALIZER.PRIMARY_ORDER:
            concept = normalized[category]
            if concept["available"]:
                layers.append(
                    {
                        "category": category,
                        "label": concept["button"],
                        "name": concept["label"],
                        "variable": concept.get("scalar") or f"{concept.get('u')}/{concept.get('v')}",
                        "vector_available": bool(concept.get("vector_available")),
                    }
                )
        return layers

    def wind_vectors(self, dataset, time_index=0, lat_name=None, lon_name=None):
        wind = self.NORMALIZER.resolve(dataset)["wind"]
        u_name = wind.get("u")
        v_name = wind.get("v")
        if not u_name or not v_name:
            return None

        u_array = self._time_slice(dataset[u_name], time_index).squeeze(drop=True)
        v_array = self._time_slice(dataset[v_name], time_index).squeeze(drop=True)
        lat_name = lat_name or self._lat_lon_names(u_array)[0]
        lon_name = lon_name or self._lat_lon_names(u_array)[1]
        if not lat_name or not lon_name:
            return None

        u_array = self._lat_lon_ordered(u_array, lat_name, lon_name)
        v_array = self._lat_lon_ordered(v_array, lat_name, lon_name)
        if u_array.ndim != 2 or v_array.ndim != 2 or u_array.shape != v_array.shape:
            return None

        return {
            "u": np.asarray(u_array.values, dtype=float),
            "v": np.asarray(v_array.values, dtype=float),
            "u_name": u_name,
            "v_name": v_name,
        }

    def find_variable_for_category(self, dataset, category):
        concept = self.NORMALIZER.resolve(dataset).get(category, {})
        if category == "wind" and concept.get("vector_available") and not concept.get("scalar"):
            return "wind_speed"
        return concept.get("scalar")

    def _find_by_aliases(self, dataset, aliases):
        for variable_name, variable in dataset.data_vars.items():
            candidates = {
                variable_name.lower(),
                variable.attrs.get("standard_name", "").lower(),
                variable.attrs.get("long_name", "").lower().replace(" ", "_"),
            }
            if candidates & aliases:
                return variable_name
        return None

    def variable_category(self, variable_name, data_array):
        return self.NORMALIZER.concept_for_variable(variable_name, data_array)

    def _inside_region(self, lat_value, lon_value, region_bounds):
        if not region_bounds:
            return True
        return (
            region_bounds["lat_min"] <= float(lat_value) <= region_bounds["lat_max"]
            and region_bounds["lon_min"] <= float(lon_value) <= region_bounds["lon_max"]
        )

    def _time_slice(self, data_array, time_index):
        time_dim = self._time_dim(data_array)
        if time_dim is None:
            return data_array

        safe_index = max(0, min(int(time_index), data_array.sizes[time_dim] - 1))
        return data_array.isel({time_dim: safe_index})

    def _data_array_for(self, dataset, variable_name):
        if variable_name in dataset.data_vars:
            return dataset[variable_name]
        if variable_name == "wind_speed":
            wind = self.NORMALIZER.resolve(dataset)["wind"]
            if wind.get("vector_available"):
                u = dataset[wind["u"]]
                v = dataset[wind["v"]]
                speed = np.sqrt(u**2 + v**2)
                speed.name = "wind_speed"
                speed.attrs["long_name"] = "Derived wind speed"
                speed.attrs["units"] = u.attrs.get("units", "m/s")
                speed.attrs["derived_from"] = f"{wind['u']}, {wind['v']}"
                return speed
        raise KeyError(f"Variable not found in dataset: {variable_name}")

    def _time_dim(self, data_array):
        for dim_name in data_array.dims:
            if dim_name.lower() in self.TIME_NAMES:
                return dim_name
        return None

    def _time_coord(self, dataset):
        for coord_name in dataset.coords:
            if coord_name.lower() in self.TIME_NAMES:
                return dataset.coords[coord_name]
        return None

    def _timestamp_for(self, dataset, time_index):
        labels = self.time_labels(dataset)
        if not labels:
            return "No time coordinate"
        safe_index = max(0, min(int(time_index), len(labels) - 1))
        return labels[safe_index]

    def _lat_lon_names(self, data_array):
        lat_name = None
        lon_name = None

        for name in data_array.coords:
            lowered = name.lower()
            coord = data_array.coords[name]
            standard_name = coord.attrs.get("standard_name", "").lower()
            units = coord.attrs.get("units", "").lower()
            if lowered in self.LAT_NAMES or standard_name == "latitude" or units == "degrees_north":
                lat_name = name
            if lowered in self.LON_NAMES or standard_name == "longitude" or units == "degrees_east":
                lon_name = name

        for name in data_array.dims:
            lowered = name.lower()
            if lat_name is None and lowered in self.LAT_NAMES:
                lat_name = name
            if lon_name is None and lowered in self.LON_NAMES:
                lon_name = name

        return lat_name, lon_name

    def _lat_lon_ordered(self, data_array, lat_name, lon_name):
        dims = list(data_array.dims)
        if lat_name in dims and lon_name in dims:
            return data_array.transpose(lat_name, lon_name, ...)
        return data_array

    def _numeric_values(self, data_array):
        try:
            return np.asarray(data_array.values, dtype=float)
        except (TypeError, ValueError):
            return None

    def _title(self, long_name, units, timestamp):
        suffix = f" ({units})" if units else ""
        return f"{long_name}{suffix} at {timestamp}"

    def _format(self, value):
        if np.isnan(value):
            return "NaN"
        return f"{float(value):.4f}"
