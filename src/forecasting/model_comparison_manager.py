from src.forecasting.forecast_manager import ForecastManager
from src.forecasting.forecast_models import SUPPORTED_MODES
from src.evaluation.model_comparator import ModelComparisonReport
from src.forecasting.tabular_forecast_manager import TabularForecastManager
from src.forecasting.tabular_forecast_models import TABULAR_FORECAST_MODES


class ModelComparisonManager:
    """Runs multiple forecasters (spanning both the univariate statistical
    registry and the tabular ML/DL registry) and collects their evaluation
    metrics into one ModelComparisonReport."""

    def __init__(self):
        self.forecast_manager = ForecastManager()
        self.tabular_forecast_manager = TabularForecastManager()

    def compare(
        self,
        modes,
        data_array=None,
        variable_name=None,
        region_bounds=None,
        processed_dir=None,
        progress_callback=None,
        **univariate_kwargs,
    ):
        progress = progress_callback or (lambda message: None)
        report = ModelComparisonReport()
        errors = {}

        for mode in modes:
            try:
                progress(f"Comparing: {mode}...")
                if mode in SUPPORTED_MODES:
                    if data_array is None:
                        raise ValueError(f"'{mode}' requires an open dataset and variable.")
                    forecast = self.forecast_manager.run_forecast(
                        data_array,
                        variable_name,
                        region_bounds=region_bounds,
                        mode=mode,
                        **univariate_kwargs,
                    )
                elif mode in TABULAR_FORECAST_MODES:
                    forecast = self.tabular_forecast_manager.run_forecast(
                        mode, processed_dir=processed_dir, progress_callback=progress
                    )
                else:
                    raise ValueError(f"Unknown forecast mode: {mode}")
                report.add_result(mode, forecast.metrics)
            except Exception as error:
                errors[mode] = str(error)

        if errors:
            report.add("errors", "failed_models", errors)

        return report
