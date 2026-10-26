from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.visualization.widgets.analysis_panel import DataAnalysisPanel
from src.visualization.widgets.dashboard_visuals import C, STYLE, Sidebar, apply_palette, font
from src.visualization.widgets.data_grid import DataSliceGrid
from src.visualization.widgets.dataset_summary import DatasetSummaryPanel
from src.visualization.widgets.metadata_viewer import MetadataViewer
from src.visualization.widgets.metrics_panel import MetricsPanel
from src.visualization.widgets.satellite_panel import SatelliteLayerPanel
from src.visualization.forecast_plotter import ForecastPanel, ModelComparisonPanel
from src.visualization.spatial_plotter import ScientificMapPanel
from src.visualization.time_series_plotter import PlotPanel
from src.forecasting.forecast_models import SUPPORTED_MODES
from src.forecasting.tabular_forecast_models import TABULAR_FORECAST_MODES
from src.utils.constants import SUPPORTED_DATASET_EXTENSIONS


class ExpandableSection(QFrame):
    def __init__(self, title, widget):
        super().__init__()
        self.widget = widget
        self.setObjectName("sectionFrame")
        self.setStyleSheet(
            f"""
            QFrame#sectionFrame{{
                background:{C['panel']};
                border:1px solid {C['soft_border']};
                border-radius:6px;
            }}
            QLabel{{background:transparent;border:none;color:{C['txt']};}}
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        title_row = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setFont(font(11, True))
        expand_button = QPushButton("Expand")
        expand_button.setFixedWidth(70)
        expand_button.setStyleSheet(
            f"QPushButton{{color:{C['txt']};background:{C['card']};border:1px solid {C['soft_border']};"
            "border-radius:4px;padding:3px;font-size:10px;font-weight:600;}}"
            f"QPushButton:hover{{border-color:{C['cyan']};}}"
        )
        expand_button.clicked.connect(lambda: self.open_expanded(title))
        title_row.addWidget(title_label)
        title_row.addStretch()
        title_row.addWidget(expand_button)
        layout.addLayout(title_row)
        layout.addWidget(widget)

    def open_expanded(self, title):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(980, 680)
        layout = QVBoxLayout(dialog)
        if hasattr(self.widget, "expanded_copy"):
            expanded = self.widget.expanded_copy()
            layout.addWidget(expanded)
        else:
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setPixmap(self.widget.grab().scaled(920, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(label)
            layout.addWidget(scroll)
        dialog.exec_()


class MainWindow(QMainWindow):
    open_dataset_requested = pyqtSignal()
    open_multi_dataset_requested = pyqtSignal()
    open_dataset_folder_requested = pyqtSignal()
    close_dataset_requested = pyqtSignal()
    export_metadata_requested = pyqtSignal()
    export_telemetry_requested = pyqtSignal()
    export_dataset_requested = pyqtSignal()
    generate_report_requested = pyqtSignal()
    engineer_features_requested = pyqtSignal()
    run_forecast_requested = pyqtSignal(dict)
    compare_models_requested = pyqtSignal(dict)
    variable_selected = pyqtSignal(str)
    weather_layer_selected = pyqtSignal(str)
    weather_layer_toggled = pyqtSignal(str, bool)
    layer_opacity_changed = pyqtSignal(float)
    region_selected = pyqtSignal(dict)
    projection_selected = pyqtSignal(str)
    time_slice_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("WEATHER DATA ANALYSIS SYSTEM v2.1")
        self.resize(1500, 860)
        self.setStyleSheet(STYLE)

        app = QApplication.instance()
        if app is not None:
            app.setFont(font(10))
            apply_palette(app)

        self.map_panel = ScientificMapPanel()
        self.map_panel.layer_requested.connect(self.weather_layer_selected.emit)
        self.map_panel.layer_toggled.connect(self.weather_layer_toggled.emit)
        self.map_panel.opacity_changed.connect(self.layer_opacity_changed.emit)
        self.map_panel.region_selected.connect(self.region_selected.emit)
        self.data_grid = DataSliceGrid()
        self.metrics_panel = MetricsPanel()
        self.metadata_viewer = MetadataViewer()
        self.dataset_summary = DatasetSummaryPanel()
        self.plot_panel = PlotPanel()
        self.plot_panel.set_compact_mode(True)
        self.forecast_panel = ForecastPanel()
        self.comparison_panel = ModelComparisonPanel()
        self.satellite_panel = SatelliteLayerPanel()
        self.analysis_panel = DataAnalysisPanel()

        self.variable_combo = QComboBox()
        self.variable_combo.setMinimumWidth(240)
        self.variable_combo.setStyleSheet(f"color:{C['txt']};background:{C['card']};")
        self.variable_combo.currentTextChanged.connect(self._emit_variable_selection)
        self.projection_combo = QComboBox()
        self.projection_combo.addItems(["Auto", "PlateCarree", "Mercator", "Orthographic"])
        self.projection_combo.setStyleSheet(f"color:{C['txt']};background:{C['card']};")
        self.projection_combo.currentTextChanged.connect(self._emit_projection_selection)
        self.forecast_combo = QComboBox()
        self.forecast_combo.addItems(SUPPORTED_MODES + TABULAR_FORECAST_MODES)
        self.forecast_combo.setMinimumWidth(180)
        self.forecast_combo.setStyleSheet(f"color:{C['txt']};background:{C['card']};")
        self.forecast_variable_combo = QComboBox()
        self.forecast_variable_combo.setMinimumWidth(220)
        self.forecast_variable_combo.setStyleSheet(f"color:{C['txt']};background:{C['card']};")
        self.forecast_window_spin = QSpinBox()
        self.forecast_window_spin.setRange(1, 96)
        self.forecast_window_spin.setValue(3)
        self.forecast_window_spin.setFixedWidth(72)
        self.forecast_horizon_spin = QSpinBox()
        self.forecast_horizon_spin.setRange(1, 24)
        self.forecast_horizon_spin.setValue(1)
        self.forecast_horizon_spin.setFixedWidth(72)
        self.forecast_steps_spin = QSpinBox()
        self.forecast_steps_spin.setRange(1, 48)
        self.forecast_steps_spin.setValue(3)
        self.forecast_steps_spin.setFixedWidth(72)
        for spin in (self.forecast_window_spin, self.forecast_horizon_spin, self.forecast_steps_spin):
            spin.setStyleSheet(f"color:{C['txt']};background:{C['card']};")

        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setEnabled(False)
        self.time_slider.valueChanged.connect(self._time_control_changed)

        self.time_slice_spin = QSpinBox()
        self.time_slice_spin.setMinimum(0)
        self.time_slice_spin.setMaximum(0)
        self.time_slice_spin.setEnabled(False)
        self.time_slice_spin.setStyleSheet(f"color:{C['txt']};background:{C['card']};")
        self.time_slice_spin.valueChanged.connect(self._time_control_changed)

        self.time_label = QLabel("No dataset time coordinate")
        self.time_label.setStyleSheet(f"color:{C['txt']};font-weight:600;")

        self._build_menu()
        self._build_layout()

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready. Open a NetCDF dataset to begin.")

        self.status_progress = QProgressBar()
        self.status_progress.setRange(0, 0)
        self.status_progress.setMaximumWidth(160)
        self.status_progress.setMaximumHeight(14)
        self.status_progress.setTextVisible(False)
        self.status_progress.hide()
        self.statusBar().addPermanentWidget(self.status_progress)

        self._busy_controls = [
            self.open_action,
            self.open_multi_action,
            self.open_folder_action,
            self.close_action,
            self.open_button,
            self.close_button,
            self.engineer_button,
            self.run_button,
            self.compare_button,
        ]

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("FILE")

        self.open_action = QAction("Open Dataset", self)
        self.open_action.triggered.connect(self.open_dataset_requested.emit)
        file_menu.addAction(self.open_action)

        self.open_multi_action = QAction("Open Multiple Datasets (Multi-Year)", self)
        self.open_multi_action.triggered.connect(self.open_multi_dataset_requested.emit)
        file_menu.addAction(self.open_multi_action)

        self.open_folder_action = QAction("Open Dataset Folder (Multi-Year)", self)
        self.open_folder_action.triggered.connect(self.open_dataset_folder_requested.emit)
        file_menu.addAction(self.open_folder_action)

        export_action = QAction("Export Metadata", self)
        export_action.triggered.connect(self.export_metadata_requested.emit)
        file_menu.addAction(export_action)

        telemetry_action = QAction("Export Telemetry CSV", self)
        telemetry_action.triggered.connect(self.export_telemetry_requested.emit)
        file_menu.addAction(telemetry_action)

        dataset_action = QAction("Export Processed NetCDF", self)
        dataset_action.triggered.connect(self.export_dataset_requested.emit)
        file_menu.addAction(dataset_action)

        report_action = QAction("Generate HTML/PDF Report", self)
        report_action.triggered.connect(self.generate_report_requested.emit)
        file_menu.addAction(report_action)

        self.close_action = QAction("Close Dataset", self)
        self.close_action.triggered.connect(self.close_dataset_requested.emit)
        file_menu.addAction(self.close_action)

        for menu_name in ["VIEW", "TOOLS", "HELP"]:
            self.menuBar().addMenu(menu_name)

    def _build_layout(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.buttons["dashboard"].clicked.connect(lambda: self.show_module("gis"))
        self.sidebar.buttons["gis globe"].clicked.connect(lambda: self.show_module("gis"))
        self.sidebar.buttons["time-series"].clicked.connect(lambda: self.show_module("analysis"))
        self.sidebar.buttons["predictive"].clicked.connect(lambda: self.show_module("forecast"))
        self.sidebar.buttons["forecast"].clicked.connect(lambda: self.show_module("forecast"))
        self.sidebar.buttons["export"].clicked.connect(self.export_metadata_requested.emit)

        self.center_stack = QStackedWidget()
        self.gis_page = self._gis_page()
        self.analysis_page = self._analysis_page()
        self.forecast_page = self._forecast_page()
        self.center_stack.addWidget(self.gis_page)
        self.center_stack.addWidget(self.analysis_page)
        self.center_stack.addWidget(self.forecast_page)

        right_panel = QWidget()
        right_panel.setMinimumWidth(390)
        right_panel.setMaximumWidth(430)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)
        title = QLabel("Live Widgets")
        title.setFont(font(15, True))
        right_layout.addWidget(title)
        right_layout.addWidget(self._section("1) Atmospheric Breakdown", self.metrics_panel, expandable=True), 0)
        right_layout.addWidget(self._section("2) Multi-Variate Comparison", self.plot_panel, expandable=True), 1)
        right_layout.addWidget(self._section("3) Satellite / Raster Feed", self.satellite_panel, expandable=True), 0)

        info_tabs = QTabWidget()
        info_tabs.addTab(self.dataset_summary, "Summary")
        info_tabs.addTab(self.metadata_viewer, "Metadata")
        right_layout.addWidget(info_tabs, 1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.center_stack)
        splitter.addWidget(right_panel)
        splitter.setSizes([1040, 420])

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(splitter, 1)
        self.show_module("gis")

    def _gis_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        layout.addWidget(self._controls_bar())
        layout.addWidget(self._section("Geospatial Weather Visualization", self.map_panel), 1)
        layout.addWidget(self._time_bar(), 0)
        layout.addWidget(self._section("Dataset Slice Inspector", self.data_grid), 0)
        return page

    def _analysis_page(self):
        return self.analysis_panel

    def _forecast_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        layout.addWidget(self._forecast_controls_bar())
        layout.addWidget(self._section("Forecast Analysis Workspace", self.forecast_panel), 1)
        layout.addWidget(self._section("Model Comparison", self.comparison_panel), 0)
        return page

    def _controls_bar(self):
        frame = QFrame()
        frame.setObjectName("controlsBar")
        frame.setStyleSheet(
            f"""
            QFrame#controlsBar{{
                background:{C['panel']};
                border:1px solid {C['soft_border']};
                border-radius:6px;
            }}
            QLabel{{background:transparent;border:none;color:{C['txt']};}}
            """
        )
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        self.open_button = QPushButton("Open Dataset")
        self.open_button.clicked.connect(self.open_dataset_requested.emit)
        export_button = QPushButton("Export Metadata")
        export_button.clicked.connect(self.export_metadata_requested.emit)
        self.close_button = QPushButton("Close Dataset")
        self.close_button.clicked.connect(self.close_dataset_requested.emit)
        for button in (self.open_button, export_button, self.close_button):
            button.setStyleSheet(
                f"QPushButton{{color:#ffffff;background:{C['card']};border:1px solid {C['soft_border']};"
                "border-radius:5px;padding:6px 10px;font-weight:600;}}"
                f"QPushButton:hover{{border-color:{C['blue']};}}"
            )

        variable_label = QLabel("Variable")
        variable_label.setStyleSheet("color:#e9f4ff;font-weight:600;background:transparent;border:none;")
        time_label = QLabel("Time")
        time_label.setStyleSheet("color:#e9f4ff;font-weight:600;background:transparent;border:none;")

        layout.addWidget(variable_label)
        layout.addWidget(self.variable_combo)
        layout.addWidget(QLabel("Projection"))
        layout.addWidget(self.projection_combo)
        layout.addStretch()
        layout.addWidget(self.open_button)
        layout.addWidget(export_button)
        layout.addWidget(self.close_button)
        return frame

    def _forecast_controls_bar(self):
        frame = QFrame()
        frame.setObjectName("forecastControlsBar")
        frame.setStyleSheet(
            f"""
            QFrame#forecastControlsBar{{
                background:{C['panel']};
                border:1px solid {C['soft_border']};
                border-radius:6px;
            }}
            QLabel{{background:transparent;border:none;color:{C['txt']};}}
            """
        )
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        self.engineer_button = QPushButton("Engineer Features")
        self.engineer_button.setMinimumWidth(130)
        self.engineer_button.clicked.connect(self.engineer_features_requested.emit)
        self.engineer_button.setStyleSheet(
            f"QPushButton{{color:#ffffff;background:{C['card']};border:1px solid {C['soft_border']};"
            "border-radius:5px;padding:6px 12px;font-weight:600;}}"
            f"QPushButton:hover{{border-color:{C['cyan']};}}"
        )

        self.run_button = QPushButton("Run Forecast")
        self.run_button.setMinimumWidth(120)
        self.run_button.clicked.connect(self._emit_forecast_request)
        self.run_button.setStyleSheet(
            f"QPushButton{{color:#ffffff;background:{C['card']};border:1px solid {C['soft_border']};"
            "border-radius:5px;padding:6px 12px;font-weight:600;}}"
            f"QPushButton:hover{{border-color:{C['blue']};}}"
        )

        self.compare_button = QPushButton("Compare Models")
        self.compare_button.setMinimumWidth(140)
        self.compare_button.clicked.connect(self._emit_compare_request)
        self.compare_button.setStyleSheet(
            f"QPushButton{{color:#ffffff;background:{C['card']};border:1px solid {C['soft_border']};"
            "border-radius:5px;padding:6px 12px;font-weight:600;}}"
            f"QPushButton:hover{{border-color:{C['blue']};}}"
        )

        layout.addWidget(QLabel("Variable"))
        layout.addWidget(self.forecast_variable_combo)
        layout.addWidget(QLabel("Model"))
        layout.addWidget(self.forecast_combo)
        layout.addWidget(QLabel("Window"))
        layout.addWidget(self.forecast_window_spin)
        layout.addWidget(QLabel("Horizon"))
        layout.addWidget(self.forecast_horizon_spin)
        layout.addWidget(QLabel("Future Steps"))
        layout.addWidget(self.forecast_steps_spin)
        layout.addStretch()
        layout.addWidget(self.engineer_button)
        layout.addWidget(self.run_button)
        layout.addWidget(self.compare_button)
        return frame

    def _time_bar(self):
        frame = QFrame()
        frame.setObjectName("timeBar")
        frame.setStyleSheet(
            f"""
            QFrame#timeBar{{
                background:{C['panel']};
                border:1px solid {C['soft_border']};
                border-radius:6px;
            }}
            QLabel{{background:transparent;border:none;color:{C['txt']};}}
            """
        )
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)
        label = QLabel("Time Evolution")
        label.setFont(font(10, True))
        layout.addWidget(label)
        layout.addWidget(self.time_slider, 1)
        layout.addWidget(self.time_slice_spin)
        layout.addWidget(self.time_label)
        return frame

    def _section(self, title, widget, expandable=False):
        if expandable:
            return ExpandableSection(title, widget)
        frame = QFrame()
        frame.setObjectName("sectionFrame")
        frame.setStyleSheet(
            f"""
            QFrame#sectionFrame{{
                background:{C['panel']};
                border:1px solid {C['soft_border']};
                border-radius:6px;
            }}
            QLabel{{background:transparent;border:none;color:{C['txt']};}}
            """
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 6, 8, 8)
        title_label = QLabel(title)
        title_label.setFont(font(11, True))
        layout.addWidget(title_label)
        layout.addWidget(widget)
        return frame

    def choose_dataset_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Weather Dataset",
            "",
            "Weather Datasets (*.nc *.netcdf *.cdf *.grib *.grb *.grib2 *.grb2 *.grd);;"
            "NetCDF Files (*.nc *.netcdf *.cdf);;"
            "GRIB Files (*.grib *.grb *.grib2 *.grb2);;"
            "GRD Files (*.grd);;"
            "All Files (*)",
        )
        return file_path

    def choose_dataset_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Multiple Weather Datasets",
            "",
            "Weather Datasets (*.nc *.netcdf *.cdf *.grib *.grb *.grib2 *.grb2 *.grd);;"
            "NetCDF Files (*.nc *.netcdf *.cdf);;"
            "GRIB Files (*.grib *.grb *.grib2 *.grb2);;"
            "GRD Files (*.grd);;"
            "All Files (*)",
        )
        return file_paths

    def choose_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Dataset Folder (Multi-Year)")
        if not folder:
            return []
        return sorted(
            str(path)
            for path in Path(folder).iterdir()
            if path.suffix.lower() in SUPPORTED_DATASET_EXTENSIONS
        )

    def set_metadata(self, metadata):
        self.metadata_viewer.set_metadata(metadata)

    def set_dataset_summary(self, summary):
        self.dataset_summary.set_summary(summary)
        self.satellite_panel.set_dataset_context(summary)

    def set_variables(self, variables):
        self.variable_combo.blockSignals(True)
        self.variable_combo.clear()
        self.variable_combo.addItems(variables)
        self.variable_combo.blockSignals(False)
        self.forecast_variable_combo.blockSignals(True)
        self.forecast_variable_combo.clear()
        self.forecast_variable_combo.addItems(variables)
        self.forecast_variable_combo.blockSignals(False)
        if variables:
            self.variable_selected.emit(variables[0])

    def set_current_variable(self, variable_name):
        index = self.variable_combo.findText(variable_name)
        if index < 0:
            return
        self.variable_combo.blockSignals(True)
        self.variable_combo.setCurrentIndex(index)
        self.variable_combo.blockSignals(False)

    def set_weather_layers(self, layers):
        self.map_panel.set_layers(layers)

    def set_layer_checked(self, category, checked):
        self.map_panel.set_layer_checked(category, checked)

    def set_projection(self, projection_name):
        self.map_panel.set_projection(projection_name)

    def set_time_labels(self, labels):
        self._time_labels = labels
        self.configure_time_slice(len(labels))
        self._set_time_label(0)

    def configure_time_slice(self, size, value=0):
        if not getattr(self, "_time_labels", None) and size:
            self._time_labels = [str(index) for index in range(size)]
        enabled = size > 1
        safe_value = max(0, min(int(value), max(0, size - 1)))
        for control in (self.time_slider, self.time_slice_spin):
            control.blockSignals(True)
            control.setEnabled(enabled)
            control.setMinimum(0)
            control.setMaximum(max(0, size - 1))
            control.setValue(safe_value)
            control.blockSignals(False)
        self._set_time_label(safe_value)

    def current_time_slice(self):
        return self.time_slice_spin.value()

    def set_map_data(self, view_data):
        self.map_panel.render(view_data)

    def set_region_bounds(self, bounds):
        self.map_panel.set_region_bounds(bounds)

    def clear_region_bounds(self):
        self.map_panel.clear_region_bounds()

    def set_metrics(self, metrics):
        self.metrics_panel.set_metrics(metrics)

    def set_grid_rows(self, rows):
        self.data_grid.set_rows(rows)

    def set_slice_statistics(self, statistics):
        self.plot_panel.set_statistics(statistics)

    def set_variable_plot(self, data_array, time_index, comparison_series=None):
        self.plot_panel.plot_variable(data_array, time_index, comparison_series)

    def set_forecast(self, forecast):
        self.forecast_panel.set_forecast(forecast)
        self.show_module("forecast")

    def set_comparison(self, comparison_report):
        self.comparison_panel.set_comparison(comparison_report)
        self.show_module("forecast")

    def set_analysis_reports(self, analysis_report, preprocessing_report, dataset_info):
        self.analysis_panel.set_reports(analysis_report, preprocessing_report, dataset_info)

    def reset_dataset_view(self):
        self.metadata_viewer.clear()
        self.dataset_summary.clear()
        self.analysis_panel.clear()
        self.satellite_panel.clear()
        self.plot_panel.clear()
        self.forecast_panel.clear()
        self.comparison_panel.clear()
        self.map_panel.clear()
        self.map_panel.clear_region_bounds()
        self.map_panel.set_layers([])
        self.metrics_panel.clear()
        self.data_grid.clear()
        self.variable_combo.blockSignals(True)
        self.variable_combo.clear()
        self.variable_combo.blockSignals(False)
        self.forecast_variable_combo.blockSignals(True)
        self.forecast_variable_combo.clear()
        self.forecast_variable_combo.blockSignals(False)
        self.projection_combo.setCurrentText("Auto")
        self.set_time_labels([])
        self.time_label.setText("No dataset time coordinate")

    def set_status(self, message):
        self.statusBar().showMessage(message)

    def set_busy(self, busy, message=None):
        """Lock the controls that touch shared dataset/forecast state while a
        background task runs, and show an unmistakable 'still working'
        indicator instead of letting the window silently freeze."""
        for control in self._busy_controls:
            control.setEnabled(not busy)
        if busy:
            self.status_progress.show()
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            self.status_progress.hide()
            QApplication.restoreOverrideCursor()
        if message:
            self.set_status(message)

    def show_error(self, title, message):
        QMessageBox.critical(self, title, message)

    def show_info(self, title, message):
        QMessageBox.information(self, title, message)

    def _emit_variable_selection(self, variable_name):
        if variable_name:
            self.variable_selected.emit(variable_name)

    def _emit_projection_selection(self, projection_name):
        self.set_projection(projection_name)
        self.projection_selected.emit(projection_name)

    def _emit_forecast_request(self):
        self.run_forecast_requested.emit(
            {
                "variable": self.forecast_variable_combo.currentText(),
                "mode": self.forecast_combo.currentText(),
                "window_size": self.forecast_window_spin.value(),
                "horizon": self.forecast_horizon_spin.value(),
                "future_steps": self.forecast_steps_spin.value(),
            }
        )

    def _emit_compare_request(self):
        self.compare_models_requested.emit(
            {
                "variable": self.forecast_variable_combo.currentText(),
                "window_size": self.forecast_window_spin.value(),
                "horizon": self.forecast_horizon_spin.value(),
                "future_steps": self.forecast_steps_spin.value(),
            }
        )

    def show_module(self, name):
        if not hasattr(self, "center_stack"):
            return
        mapping = {
            "gis": (self.gis_page, "gis globe"),
            "analysis": (self.analysis_page, "time-series"),
            "forecast": (self.forecast_page, "forecast"),
        }
        page, key = mapping.get(name, mapping["gis"])
        self.center_stack.setCurrentWidget(page)
        self.sidebar.set_active(key)

    def _time_control_changed(self, value):
        sender = self.sender()
        other = self.time_slice_spin if sender is self.time_slider else self.time_slider
        other.blockSignals(True)
        other.setValue(value)
        other.blockSignals(False)
        self._set_time_label(value)
        self.time_slice_changed.emit(value)

    def _set_time_label(self, index):
        labels = getattr(self, "_time_labels", [])
        if labels:
            safe_index = max(0, min(index, len(labels) - 1))
            self.time_label.setText(labels[safe_index])
        else:
            self.time_label.setText("No dataset time coordinate")
