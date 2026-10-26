from PyQt5.QtCore import QTimer

from src.analysis.dataset_view import DatasetViewBuilder
from src.analysis.descriptive_statistics import AnalysisEngine
from src.analysis.metadata import MetadataExtractor
from src.data.dataset_manager import DatasetManager
from src.forecasting.forecast_manager import ForecastManager
from src.forecasting.forecast_models import SUPPORTED_MODES
from src.forecasting.model_comparison_manager import ModelComparisonManager
from src.forecasting.tabular_forecast_manager import TabularForecastManager
from src.forecasting.tabular_forecast_models import TABULAR_FORECAST_MODES
from src.reporting.export_manager import ExportManager
from src.reporting.report_generator import ReportManager
from src.visualization.background_task import BackgroundTask


class MainController:
    """Coordinates GUI events with the model layer."""

    def __init__(self, view):
        self.view = view
        self.dataset_manager = DatasetManager()
        self.metadata_extractor = MetadataExtractor()
        self.analysis_engine = AnalysisEngine()
        self.dataset_view_builder = DatasetViewBuilder()
        self.forecast_manager = ForecastManager()
        self.tabular_forecast_manager = TabularForecastManager()
        self.comparison_manager = ModelComparisonManager()
        self.export_manager = ExportManager()
        self.report_manager = ReportManager()
        self.metadata = {}
        self.current_variable = None
        self.current_rows = []
        self.current_statistics = {}
        self.active_overlay_layers = []
        self.layer_opacity = 0.45
        self.region_bounds = None
        self._active_task = None
        self._pending_time_index = 0
        self._refresh_timer = QTimer(self.view)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._flush_refresh)

        self.view.open_dataset_requested.connect(self.open_dataset)
        self.view.open_multi_dataset_requested.connect(self.open_multiple_datasets)
        self.view.open_dataset_folder_requested.connect(self.open_dataset_folder)
        self.view.close_dataset_requested.connect(self.close_dataset)
        self.view.variable_selected.connect(self.select_variable)
        self.view.weather_layer_selected.connect(self.select_weather_layer)
        self.view.weather_layer_toggled.connect(self.toggle_weather_layer)
        self.view.layer_opacity_changed.connect(self.set_layer_opacity)
        self.view.region_selected.connect(self.select_region)
        self.view.time_slice_changed.connect(self.refresh_current_variable)
        self.view.export_metadata_requested.connect(self.export_metadata)
        self.view.export_telemetry_requested.connect(self.export_telemetry)
        self.view.export_dataset_requested.connect(self.export_dataset)
        self.view.generate_report_requested.connect(self.generate_report)
        self.view.run_forecast_requested.connect(self.run_forecast)
        self.view.compare_models_requested.connect(self.compare_models)

    def open_dataset(self):
        file_path = self.view.choose_dataset_file()
        if not file_path:
            return

        self._run_in_background(
            self.dataset_manager.load_dataset,
            lambda dataset: self._apply_loaded_dataset(dataset, f"Loaded dataset: {file_path}"),
            file_path,
            busy_message=f"Loading {file_path}...",
        )

    def open_multiple_datasets(self):
        self._load_and_apply_datasets(self.view.choose_dataset_files())

    def open_dataset_folder(self):
        self._load_and_apply_datasets(self.view.choose_dataset_folder())

    def _load_and_apply_datasets(self, file_paths):
        if not file_paths:
            return

        def on_success(dataset):
            self._apply_loaded_dataset(
                dataset,
                f"Loaded merged dataset from {len(file_paths)} files: {self.dataset_manager.processed_dataset_key}",
            )

        self._run_in_background(
            self.dataset_manager.load_datasets,
            on_success,
            file_paths,
            busy_message=f"Loading {len(file_paths)} datasets...",
        )

    def _run_in_background(self, func, on_success, *args, busy_message, **kwargs):
        if self._active_task is not None:
            return  # a task is already running - ignore re-entrant clicks
        self.view.set_busy(True, busy_message)
        task = BackgroundTask(func, *args, **kwargs)
        task.progress.connect(self.view.set_status)
        task.finished_ok.connect(lambda result: self._on_task_finished(result, on_success))
        task.failed.connect(self._on_task_failed)
        self._active_task = task
        task.start()

    def _on_task_finished(self, result, on_success):
        self.view.set_busy(False)
        self._active_task = None
        on_success(result)

    def _on_task_failed(self, message):
        self.view.set_busy(False)
        self._active_task = None
        self.view.show_error("Operation Failed", message)
        self.view.set_status("Operation failed.")

    def _apply_loaded_dataset(self, dataset, status_message):
        self.metadata = self.metadata_extractor.extract(dataset)
        self.view.set_metadata(self.metadata)
        self.view.set_dataset_summary(self.dataset_manager.get_dataset_info())
        self.view.set_analysis_reports(
        self.dataset_manager.get_analysis_report(),
        self.dataset_manager.get_preprocessing_report(),
        self.dataset_manager.get_dataset_info(),
)
        self.view.set_time_labels(self.dataset_view_builder.time_labels(dataset))
        self.view.set_weather_layers(self.dataset_view_builder.weather_layer_options(dataset))
        self.view.set_variables(self.dataset_manager.get_variable_names())
        self.view.set_status(status_message)

    def select_variable(self, variable_name):
        if not self.dataset_manager.has_variable(variable_name):
            return

        try:
            self.current_variable = variable_name
            current_time = self.view.current_time_slice()
            self.view.configure_time_slice(
                self.dataset_manager.get_time_dimension_size(variable_name),
                current_time,
            )
            self._schedule_refresh(variable_name, self.view.current_time_slice())
        except Exception as error:
            self.view.show_error("Variable Preview Failed", str(error))

    def refresh_current_variable(self, time_index):
        if not self.current_variable or not self.dataset_manager.has_variable(self.current_variable):
            return

        try:
            self._schedule_refresh(self.current_variable, time_index)
        except Exception as error:
            self.view.show_error("Variable Preview Failed", str(error))

    def select_weather_layer(self, category):
        dataset = self.dataset_manager.get_dataset()
        if dataset is None:
            return
        variable_name = self.dataset_view_builder.find_variable_for_category(dataset, category)
        if not variable_name:
            return
        self.view.set_current_variable(variable_name)
        self.select_variable(variable_name)

    def toggle_weather_layer(self, category, enabled):
        if enabled and category not in self.active_overlay_layers:
            self.active_overlay_layers.append(category)
        if not enabled and category in self.active_overlay_layers:
            self.active_overlay_layers.remove(category)
        if self.current_variable:
            self._schedule_refresh(self.current_variable, self.view.current_time_slice())

    def set_layer_opacity(self, opacity):
        self.layer_opacity = opacity
        if self.current_variable:
            self._schedule_refresh(self.current_variable, self.view.current_time_slice())

    def select_region(self, bounds):
        self.region_bounds = bounds
        self.view.set_region_bounds(bounds)
        if self.current_variable:
            self._schedule_refresh(self.current_variable, self.view.current_time_slice())

    def close_dataset(self):
        self.dataset_manager.close_dataset()
        self.metadata = {}
        self.current_variable = None
        self.view.reset_dataset_view()
        self.active_overlay_layers = []
        self.region_bounds = None
        self.view.clear_region_bounds()
        self.view.set_status("Dataset closed.")

    def _refresh_all(self, variable_name, time_index):
        dataset = self.dataset_manager.get_dataset()
        data_array = self.dataset_manager.get_variable(variable_name)
        spatial_view = self.dataset_view_builder.multi_layer_view(
            dataset,
            variable_name,
            time_index,
            self.active_overlay_layers,
            {category: self.layer_opacity for category in self.active_overlay_layers},
        )
        slice_stats = self.dataset_view_builder.variable_statistics(dataset, variable_name, time_index)
        if self.region_bounds:
            slice_stats.update(
                self.dataset_view_builder.regional_statistics(
                    dataset,
                    variable_name,
                    time_index,
                    self.region_bounds,
                )
            )
        metrics = self.dataset_view_builder.metric_summary(dataset, time_index)
        rows = self.dataset_view_builder.grid_rows(
            dataset,
            variable_name,
            time_index,
            region_bounds=self.region_bounds,
        )
        comparison_series = self.dataset_view_builder.comparison_series(dataset, variable_name)
        anomaly_summary = self.analysis_engine.anomaly_summary(
            self.dataset_view_builder._time_slice(data_array, time_index)
        )
        slice_stats["anomaly_count"] = anomaly_summary.get("count", 0)
        slice_stats["anomaly_percent"] = anomaly_summary.get("percent", "0.00")
        self.current_rows = rows
        self.current_statistics = slice_stats

        self.view.set_map_data(spatial_view)
        self.view.set_variable_plot(data_array, time_index, comparison_series)
        self.view.set_slice_statistics(slice_stats)
        self.view.set_metrics(metrics)
        self.view.set_grid_rows(rows)
        self.view.set_status(
            f"Showing {variable_name} at {spatial_view['timestamp']} "
            f"({len(rows)} inspected grid cells"
            f"{'; regional selection active' if self.region_bounds else ''})"
        )

    def _schedule_refresh(self, variable_name, time_index):
        self._pending_variable = variable_name
        self._pending_time_index = int(time_index)
        self._refresh_timer.start(120)

    def _flush_refresh(self):
        variable_name = getattr(self, "_pending_variable", None)
        if not variable_name:
            return
        if not self.dataset_manager.has_variable(variable_name):
            return
        self._refresh_all(variable_name, self._pending_time_index)

    def export_metadata(self):
        if not self.metadata:
            self.view.show_info("Nothing to Export", "Open a dataset before exporting metadata.")
            return

        try:
            output_path = self.export_manager.export_metadata(
                self.metadata,
                self.dataset_manager.file_path,
            )
            self.view.show_info("Metadata Exported", f"Saved metadata to:\n{output_path}")
            self.view.set_status(f"Metadata exported: {output_path}")
        except Exception as error:
            self.view.show_error("Export Failed", str(error))

    def export_telemetry(self):
        if not self.current_rows:
            self.view.show_info("Nothing to Export", "Load a dataset and select a variable first.")
            return
        try:
            output_path = self.export_manager.export_telemetry_csv(self.current_rows)
            self.view.show_info("Telemetry Exported", f"Saved telemetry to:\n{output_path}")
        except Exception as error:
            self.view.show_error("Export Failed", str(error))

    def export_dataset(self):
        dataset = self.dataset_manager.get_dataset()
        if dataset is None:
            self.view.show_info("Nothing to Export", "Open a dataset first.")
            return
        try:
            output_path = self.export_manager.export_dataset_netcdf(dataset)
            self.view.show_info("Dataset Exported", f"Saved processed dataset to:\n{output_path}")
        except Exception as error:
            self.view.show_error("Export Failed", str(error))

    def generate_report(self):
        dataset = self.dataset_manager.get_dataset()
        if dataset is None or not self.current_variable:
            self.view.show_info("Nothing to Report", "Open a dataset and select a variable first.")
            return
        try:
            summary = self.dataset_manager.get_dataset_info()
            html_path = self.report_manager.generate_html_report(
                self.metadata,
                summary,
                self.current_variable,
                self.current_statistics,
                self.current_rows,
            )
            pdf_path = self.report_manager.generate_pdf_report(
                dataset,
                summary,
                self.current_variable,
                self.current_statistics,
            )
            self.view.show_info("Reports Generated", f"HTML:\n{html_path}\n\nPDF:\n{pdf_path}")
        except Exception as error:
            self.view.show_error("Report Failed", str(error))

    def run_forecast(self, config):
        mode = config.get("mode", "Linear Regression") if isinstance(config, dict) else str(config)

        if mode in TABULAR_FORECAST_MODES:
            self._run_tabular_forecast(mode)
            return

        dataset = self.dataset_manager.get_dataset()
        variable_name = config.get("variable") if isinstance(config, dict) else self.current_variable
        if dataset is None or not variable_name:
            self.view.show_info("Nothing to Forecast", "Open a dataset and select a weather variable first.")
            return
        if not self.dataset_manager.has_variable(variable_name):
            self.view.show_info("Unsupported Forecast Variable", "Select a variable that exists in the loaded dataset.")
            return

        window_size = int(config.get("window_size", 3)) if isinstance(config, dict) else 3
        horizon = int(config.get("horizon", 1)) if isinstance(config, dict) else 1
        future_steps = int(config.get("future_steps", 3)) if isinstance(config, dict) else 3
        data_array = self.dataset_manager.get_variable(variable_name)

        self._run_in_background(
            self.forecast_manager.run_forecast,
            self._on_forecast_ready,
            data_array,
            variable_name,
            region_bounds=self.region_bounds,
            mode=mode,
            window_size=window_size,
            horizon=horizon,
            future_steps=future_steps,
            busy_message=f"Forecasting {variable_name} with {mode}...",
        )

    def _run_tabular_forecast(self, mode):
        if self.dataset_manager.processed_directory is None:
            self.view.show_info(
                "Nothing to Forecast",
                "Open a dataset first so Feature Engineering can run.",
            )
            return

        self._run_in_background(
            self.tabular_forecast_manager.run_forecast,
            self._on_forecast_ready,
            mode,
            processed_dir=self.dataset_manager.processed_directory,
            busy_message=f"Training {mode}...",
        )

    def _on_forecast_ready(self, forecast):
        self.view.set_forecast(forecast)
        self.view.set_status(
            f"Forecasted {forecast.variable} with {forecast.model_name}; "
            f"RMSE={forecast.metrics.get('rmse', 'n/a')}"
        )

    def compare_models(self, config):
        variable_name = config.get("variable") if isinstance(config, dict) else self.current_variable
        dataset = self.dataset_manager.get_dataset()

        data_array = None
        univariate_kwargs = {}
        if dataset is not None and variable_name and self.dataset_manager.has_variable(variable_name):
            data_array = self.dataset_manager.get_variable(variable_name)
            univariate_kwargs = {
                "window_size": int(config.get("window_size", 3)) if isinstance(config, dict) else 3,
                "horizon": int(config.get("horizon", 1)) if isinstance(config, dict) else 1,
                "future_steps": int(config.get("future_steps", 3)) if isinstance(config, dict) else 3,
            }

        modes = (SUPPORTED_MODES if data_array is not None else []) + TABULAR_FORECAST_MODES

        def on_success(report):
            self.view.set_comparison(report)
            self.view.set_status(f"Compared {len(report.to_rows())} model(s).")

        self._run_in_background(
            self.comparison_manager.compare,
            on_success,
            modes,
            data_array=data_array,
            variable_name=variable_name,
            region_bounds=self.region_bounds,
            processed_dir=self.dataset_manager.processed_directory,
            busy_message=f"Comparing {len(modes)} model(s)...",
            **univariate_kwargs,
        )
