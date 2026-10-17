import numpy as np


class AnalysisEngine:
    """Small first-pass statistics engine for selected variables."""

    def summarize(self, data_array):
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

        missing = int(np.count_nonzero(np.isnan(values)))
        valid_count = int(np.count_nonzero(~np.isnan(values)))

        valid_values = values[~np.isnan(values)]

        return {
            "count": valid_count,
            "missing": missing,

            "mean": self._format_number(np.nanmean(valid_values)),
            "median": self._format_number(np.nanmedian(valid_values)),

            "std": self._format_number(np.nanstd(valid_values)),
            "variance": self._format_number(np.nanvar(valid_values)),

            "min": self._format_number(np.nanmin(valid_values)),
            "max": self._format_number(np.nanmax(valid_values)),

            "q1": self._format_number(np.nanpercentile(valid_values, 25)),
            "q3": self._format_number(np.nanpercentile(valid_values, 75)),

            "iqr": self._format_number(
                np.nanpercentile(valid_values, 75)
                - np.nanpercentile(valid_values, 25)
            ),
        }

    def variable_range(self, data_array):
        values = self._numeric_values(data_array)
        if values is None or values.size == 0 or np.all(np.isnan(values)):
            return {"min": "N/A", "max": "N/A"}
        return {
            "min": self._format_number(np.nanmin(values)),
            "max": self._format_number(np.nanmax(values)),
        }

    def distribution(self, data_array, bins=20):
        values = self._valid_values(data_array)
        if values.size == 0:
            return {"histogram": [], "bin_edges": [], "pdf": [], "cdf": []}
        histogram, edges = np.histogram(values, bins=bins)
        widths = np.diff(edges)
        pdf = histogram / max(1, histogram.sum()) / widths
        cdf = np.cumsum(histogram) / max(1, histogram.sum())
        return {
            "histogram": histogram.tolist(),
            "bin_edges": edges.tolist(),
            "pdf": pdf.tolist(),
            "cdf": cdf.tolist(),
        }

    def temporal_spatial_mean(self, data_array):
        time_dim = self._time_dimension(data_array)
        values = self._numeric_values(data_array)
        if values is None or values.size == 0:
            return []
        if time_dim is None:
            return [float(np.nanmean(values))]
        axis = data_array.get_axis_num(time_dim)
        reduced = np.nanmean(values, axis=tuple(index for index in range(values.ndim) if index != axis))
        return [float(value) for value in np.ravel(reduced)]

    def rolling_average(self, data_array, window=3):
        series = np.asarray(self.temporal_spatial_mean(data_array), dtype=float)
        if series.size == 0:
            return []
        window = max(1, min(int(window), series.size))
        kernel = np.ones(window) / window
        return np.convolve(series, kernel, mode="same").tolist()

    def anomaly_mask(self, data_array, z_threshold=2.0, lower=None, upper=None):
        values = self._numeric_values(data_array)
        if values is None or values.size == 0:
            return None
        mean = np.nanmean(values)
        std = np.nanstd(values)
        if std == 0 or np.isnan(std):
            z_mask = np.zeros(values.shape, dtype=bool)
        else:
            z_mask = np.abs((values - mean) / std) >= z_threshold
        threshold_mask = np.zeros(values.shape, dtype=bool)
        if lower is not None:
            threshold_mask |= values < lower
        if upper is not None:
            threshold_mask |= values > upper
        return z_mask | threshold_mask

    def anomaly_summary(self, data_array, z_threshold=2.0):
        mask = self.anomaly_mask(data_array, z_threshold=z_threshold)
        values = self._numeric_values(data_array)
        if mask is None or values is None:
            return {"count": 0, "percent": "0.00"}
        valid = int(np.count_nonzero(~np.isnan(values)))
        count = int(np.count_nonzero(mask))
        percent = (count / valid * 100) if valid else 0
        return {"count": count, "percent": f"{percent:.2f}", "z_threshold": z_threshold}

    def _numeric_values(self, data_array):
        try:
            return np.asarray(data_array.values, dtype=float)
        except (TypeError, ValueError):
            return None

    def _valid_values(self, data_array):
        values = self._numeric_values(data_array)
        if values is None:
            return np.asarray([])
        return values[~np.isnan(values)]

    def _time_dimension(self, data_array):
        for dim_name in data_array.dims:
            if dim_name.lower() in {"time", "valid_time", "forecast_time"}:
                return dim_name
        return None

    def _format_number(self, value):
        if np.isnan(value):
            return "NaN"
        return f"{value:.4f}"
