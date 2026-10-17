import numpy as np

from src.evaluation.regression_metrics import ForecastEvaluator
from src.forecasting.forecast_models import build_forecaster
from src.forecasting.forecast_view_model import ForecastViewModel
from src.forecasting.preprocessing import TimeSeriesPreprocessor
from src.forecasting.windowing import WindowGenerator


class ForecastManager:
    """Phase 1 univariate regional time-series forecasting workflow."""

    TIME_NAMES = {"time", "valid_time", "forecast_time"}
    LAT_NAMES = {"lat", "latitude", "y"}
    LON_NAMES = {"lon", "longitude", "x"}

    def __init__(self):
        self.preprocessor = TimeSeriesPreprocessor()
        self.windowing = WindowGenerator()
        self.evaluator = ForecastEvaluator()

    def run_forecast(
        self,
        data_array,
        variable_name,
        region_bounds=None,
        mode="Linear Regression",
        window_size=3,
        horizon=1,
        future_steps=3,
        smoothing_window=1,
        progress_callback=None,
    ):
        progress = progress_callback or (lambda message: None)
        progress(f"Training {mode}...")
        series_array = self.extract_regional_series(data_array, region_bounds)
        values = self.preprocessor.clean(series_array.values)
        values = self.preprocessor.smooth(values, smoothing_window)

        if values.size < 3:
            raise ValueError("Forecasting needs at least 3 time steps after preprocessing.")

        window_size = min(int(window_size), max(1, values.size - 2))
        scaled, scale_state = self.preprocessor.scale(values)
        forecaster = build_forecaster(mode)

        if forecaster.WINDOWED:
            X, y = self.windowing.make_windows(scaled, window_size=window_size, horizon=horizon)
            if len(X) == 0:
                raise ValueError("Unable to generate forecasting windows from this time series.")
            X_train, X_test, y_train, y_test = self.windowing.chronological_split(X, y)
            forecaster.fit(X_train, y_train)
            predicted_scaled = forecaster.predict(X_test)
            actual_scaled = y_test
            start_index = window_size + horizon - 1 + len(y_train)
        else:
            split = self._series_split_index(scaled)
            train_series, test_series = scaled[:split], scaled[split:]
            if len(test_series) == 0:
                raise ValueError("Not enough data to hold out a test segment for this model.")
            forecaster.fit(train_series, train_series)
            predicted_scaled = forecaster.predict(test_series)
            actual_scaled = test_series
            start_index = split

        future_scaled = forecaster.forecast_next(scaled, future_steps, window_size)

        actual = self.preprocessor.inverse_scale(actual_scaled, scale_state)
        predicted = self.preprocessor.inverse_scale(predicted_scaled, scale_state)
        future = self.preprocessor.inverse_scale(future_scaled, scale_state)
        metrics = self.evaluator.evaluate(actual, predicted)

        test_index = list(range(start_index, start_index + len(actual)))
        future_index = list(range(len(values), len(values) + future_steps))

        return ForecastViewModel(
            variable=variable_name,
            units=data_array.attrs.get("units", ""),
            model_name=forecaster.name,
            time_labels=[str(value) for value in series_array.coords[self._time_dim(series_array)].values],
            series=[float(value) for value in values],
            test_index=test_index,
            actual=[float(value) for value in actual],
            predicted=[float(value) for value in predicted],
            future_index=future_index,
            future=[float(value) for value in future],
            metrics=metrics,
            region_bounds=region_bounds or {},
        )

    def _series_split_index(self, series, train_ratio=0.8):
        if len(series) <= 1:
            return len(series)
        split = int(len(series) * train_ratio)
        return max(1, min(split, len(series) - 1))

    def extract_regional_series(self, data_array, region_bounds=None):
        time_dim = self._time_dim(data_array)
        if time_dim is None:
            raise ValueError("Forecasting requires a time dimension.")

        working = data_array
        lat_name = self._coord_name(working, self.LAT_NAMES)
        lon_name = self._coord_name(working, self.LON_NAMES)
        if region_bounds and lat_name and lon_name:
            working = working.where(
                (working[lat_name] >= region_bounds["lat_min"])
                & (working[lat_name] <= region_bounds["lat_max"])
                & (working[lon_name] >= region_bounds["lon_min"])
                & (working[lon_name] <= region_bounds["lon_max"]),
                drop=True,
            )

        reduce_dims = [dim for dim in working.dims if dim != time_dim]
        if reduce_dims:
            working = working.mean(dim=reduce_dims, skipna=True)
        return working

    def _time_dim(self, data_array):
        for dim_name in data_array.dims:
            if dim_name.lower() in self.TIME_NAMES:
                return dim_name
        return None

    def _coord_name(self, data_array, names):
        for coord_name in data_array.coords:
            if coord_name.lower() in names:
                return coord_name
        return None
