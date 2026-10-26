from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.visualization.heatmap_plotter import CorrelationHeatmapWidget
from src.visualization.widgets.dashboard_visuals import C, font, mono_font


class DataAnalysisPanel(QWidget):
    """Dedicated Data Analysis workspace presenting the results of the
    Preprocessing and Analysis pipelines.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._analysis_report = {}
        self._preprocessing_report = {}
        self._dataset_info = {}
        self._build_ui()

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 8, 10, 8)
        root_layout.setSpacing(8)

        # Status Header Banner
        self.header_card = QFrame()
        self.header_card.setStyleSheet(
            f"QFrame{{background:{C['panel']};border:1px solid {C['soft_border']};border-radius:6px;padding:6px;}}"
        )
        header_layout = QHBoxLayout(self.header_card)
        header_layout.setContentsMargins(10, 6, 10, 6)

        self.title_label = QLabel("Scientific Data Analysis Workspace")
        self.title_label.setFont(font(13, True))
        self.title_label.setStyleSheet(f"color:{C['txt']};")

        self.status_badge = QLabel("No Dataset Loaded")
        self.status_badge.setFont(font(10, True))
        self.status_badge.setStyleSheet(
            f"background:{C['card']};color:{C['dim']};border:1px solid {C['soft_border']};"
            "border-radius:4px;padding:4px 10px;"
        )

        self.error_banner = QLabel("")
        self.error_banner.setFont(font(10, True))
        self.error_banner.setStyleSheet(
            f"background:#4a1218;color:#ff8585;border:1px solid #7d2028;border-radius:4px;padding:4px 8px;"
        )
        self.error_banner.hide()

        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.error_banner)
        header_layout.addWidget(self.status_badge)

        root_layout.addWidget(self.header_card)

        # Main Analysis Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            f"""
            QTabWidget::pane{{border:1px solid {C['soft_border']};background:{C['panel']};border-radius:6px;}}
            QTabBar::tab{{background:{C['mid']};color:{C['txt']};padding:8px 16px;border:1px solid {C['soft_border']};border-bottom:none;border-top-left-radius:4px;border-top-right-radius:4px;}}
            QTabBar::tab:selected{{background:{C['panel']};color:{C['cyan']};font-weight:600;border-bottom:2px solid {C['cyan']};}}
            """
        )

        self.overview_tab = self._build_overview_tab()
        self.stats_tab = self._build_stats_tab()
        self.correlation_tab = self._build_correlation_tab()
        self.trend_tab = self._build_trend_tab()
        self.anomalies_tab = self._build_anomalies_tab()
        self.quality_tab = self._build_quality_tab()

        self.tabs.addTab(self.overview_tab, "Overview & Metadata")
        self.tabs.addTab(self.stats_tab, "Descriptive Statistics")
        self.tabs.addTab(self.correlation_tab, "Correlation Matrix")
        self.tabs.addTab(self.trend_tab, "Temporal Trends")
        self.tabs.addTab(self.anomalies_tab, "Anomalies & Extremes")
        self.tabs.addTab(self.quality_tab, "Preprocessing Quality")

        root_layout.addWidget(self.tabs, stretch=1)

    # -------------------------------------------------------------
    # 1. Overview Tab
    # -------------------------------------------------------------
    def _build_overview_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Top summary cards grid
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(10)

        self.card_vars = self._summary_card("Variables", "-", C["blue"])
        self.card_times = self._summary_card("Time Steps", "-", C["cyan"])
        self.card_quality = self._summary_card("Quality Score", "-", C["green"])
        self.card_missing = self._summary_card("Missing Values", "-", C["amber"])

        cards_layout.addWidget(self.card_vars)
        cards_layout.addWidget(self.card_times)
        cards_layout.addWidget(self.card_quality)
        cards_layout.addWidget(self.card_missing)

        layout.addLayout(cards_layout)

        # Details Table
        self.metadata_table = QTableWidget(0, 2)
        self.metadata_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.metadata_table.horizontalHeader().setStretchLastSection(True)
        self.metadata_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        layout.addWidget(self.metadata_table, stretch=1)

        return page

    # -------------------------------------------------------------
    # 2. Descriptive Statistics Tab
    # -------------------------------------------------------------
    def _build_stats_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        desc_label = QLabel("Comprehensive descriptive statistics for all numeric meteorological variables:")
        desc_label.setFont(font(10))
        desc_label.setStyleSheet(f"color:{C['dim']};")
        layout.addWidget(desc_label)

        self.stats_table = QTableWidget(0, 11)
        self.stats_table.setHorizontalHeaderLabels([
            "Variable",
            "Mean",
            "Median",
            "Std Dev",
            "Variance",
            "Min",
            "Max",
            "Q1 (25%)",
            "Q3 (75%)",
            "IQR",
            "Missing",
        ])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        layout.addWidget(self.stats_table, stretch=1)

        return page

    # -------------------------------------------------------------
    # 3. Correlation Analysis Tab
    # -------------------------------------------------------------
    def _build_correlation_tab(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Left: Heatmap
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        heatmap_title = QLabel("Pearson Correlation Heatmap")
        heatmap_title.setFont(font(11, True))
        self.heatmap_widget = CorrelationHeatmapWidget()
        left_layout.addWidget(heatmap_title)
        left_layout.addWidget(self.heatmap_widget, stretch=1)

        # Right: Strong Pairs & Table
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        pairs_title = QLabel("Strong Cross-Variable Correlations (|r| >= 0.70)")
        pairs_title.setFont(font(11, True))
        right_layout.addWidget(pairs_title)

        self.correlation_table = QTableWidget(0, 3)
        self.correlation_table.setHorizontalHeaderLabels(["Variable 1", "Variable 2", "Pearson r"])
        self.correlation_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(self.correlation_table, stretch=1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([650, 450])

        layout.addWidget(splitter)
        return page

    # -------------------------------------------------------------
    # 4. Temporal Trends Tab
    # -------------------------------------------------------------
    def _build_trend_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Controls bar
        ctrl_bar = QHBoxLayout()
        var_label = QLabel("Select Variable:")
        var_label.setFont(font(10, True))
        self.trend_var_combo = QComboBox()
        self.trend_var_combo.setMinimumWidth(220)
        self.trend_var_combo.currentTextChanged.connect(self._on_trend_variable_changed)

        self.trend_badge = QLabel("Trend: -")
        self.trend_badge.setFont(font(10, True))
        self.trend_badge.setStyleSheet(
            f"background:{C['card']};color:{C['txt']};border:1px solid {C['soft_border']};"
            "border-radius:4px;padding:4px 10px;"
        )

        self.slope_badge = QLabel("Slope: -")
        self.slope_badge.setFont(font(10, True))
        self.slope_badge.setStyleSheet(
            f"background:{C['card']};color:{C['cyan']};border:1px solid {C['soft_border']};"
            "border-radius:4px;padding:4px 10px;"
        )

        ctrl_bar.addWidget(var_label)
        ctrl_bar.addWidget(self.trend_var_combo)
        ctrl_bar.addWidget(self.trend_badge)
        ctrl_bar.addWidget(self.slope_badge)
        ctrl_bar.addStretch()

        layout.addLayout(ctrl_bar)

        # Matplotlib trend canvas
        self.trend_figure = Figure(facecolor="#121b27")
        self.trend_canvas = FigureCanvas(self.trend_figure)
        layout.addWidget(self.trend_canvas, stretch=1)

        return page

    # -------------------------------------------------------------
    # 5. Anomalies & Extremes Tab
    # -------------------------------------------------------------
    def _build_anomalies_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        splitter = QSplitter(Qt.Vertical)

        # Top: Anomalies (Z-Score)
        anom_widget = QWidget()
        anom_layout = QVBoxLayout(anom_widget)
        anom_layout.setContentsMargins(0, 0, 0, 0)
        anom_title = QLabel("Statistical Anomaly Detection (Z-Score Threshold: 2.0σ)")
        anom_title.setFont(font(11, True))
        anom_layout.addWidget(anom_title)

        self.anomalies_table = QTableWidget(0, 6)
        self.anomalies_table.setHorizontalHeaderLabels([
            "Variable", "Mean", "Std Dev", "Threshold (σ)", "Anomalous Points", "Anomaly %"
        ])
        self.anomalies_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.anomalies_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        anom_layout.addWidget(self.anomalies_table)

        # Bottom: Extreme Weather Events
        ext_widget = QWidget()
        ext_layout = QVBoxLayout(ext_widget)
        ext_layout.setContentsMargins(0, 0, 0, 0)
        ext_title = QLabel("Extreme Weather Analysis (Domain Threshold Event Counts)")
        ext_title.setFont(font(11, True))
        ext_layout.addWidget(ext_title)

        self.extremes_table = QTableWidget(0, 6)
        self.extremes_table.setHorizontalHeaderLabels([
            "Variable", "Min Observed", "Max Observed", "Mean", "High Extremes (> Threshold)", "Low Extremes (< Threshold)"
        ])
        self.extremes_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.extremes_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        ext_layout.addWidget(self.extremes_table)

        splitter.addWidget(anom_widget)
        splitter.addWidget(ext_widget)
        layout.addWidget(splitter)

        return page

    # -------------------------------------------------------------
    # 6. Preprocessing Quality Tab
    # -------------------------------------------------------------
    def _build_quality_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        # Quality cards
        q_cards = QHBoxLayout()
        self.qc_score = self._summary_card("Quality Score", "-", C["green"])
        self.qc_grade = self._summary_card("Dataset Grade", "-", C["cyan"])
        self.qc_ready = self._summary_card("Model Ready", "-", C["blue"])
        self.qc_dups = self._summary_card("Duplicates Dropped", "-", C["amber"])

        q_cards.addWidget(self.qc_score)
        q_cards.addWidget(self.qc_grade)
        q_cards.addWidget(self.qc_ready)
        q_cards.addWidget(self.qc_dups)
        content_layout.addLayout(q_cards)

        # Missing values stage report
        content_layout.addWidget(QLabel("Stage 1 & 4 — Missing Value Assessment & Temporal Gap Repair:"))
        self.missing_table = QTableWidget(0, 6)
        self.missing_table.setHorizontalHeaderLabels([
            "Variable", "Missing Before", "Missing After", "Repaired", "Permanent Mask", "Resolution Method"
        ])
        self.missing_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.missing_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.missing_table.setMinimumHeight(150)
        content_layout.addWidget(self.missing_table)

        # Statistical Outlier report
        content_layout.addWidget(QLabel("Stage 6 — Statistical Outlier Detection (Meteorological IQR):"))
        self.outlier_table = QTableWidget(0, 8)
        self.outlier_table.setHorizontalHeaderLabels([
            "Variable", "Method", "Action", "Multiplier", "Outlier Count", "Outlier %", "Severity", "Recommendation"
        ])
        self.outlier_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.outlier_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.outlier_table.setMinimumHeight(160)
        content_layout.addWidget(self.outlier_table)

        scroll.setWidget(content)
        layout.addWidget(scroll)

        return page

    def _summary_card(self, title, initial_value, color):
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame{{background:{C['card']};border:1px solid {C['soft_border']};border-radius:6px;padding:8px;}}"
        )
        l = QVBoxLayout(frame)
        l.setContentsMargins(6, 6, 6, 6)
        l.setSpacing(2)

        val_label = QLabel(initial_value)
        val_label.setFont(mono_font(16, True))
        val_label.setStyleSheet(f"color:{color};border:none;background:transparent;")
        val_label.setAlignment(Qt.AlignCenter)

        t_label = QLabel(title)
        t_label.setFont(font(9))
        t_label.setStyleSheet(f"color:{C['dim']};border:none;background:transparent;")
        t_label.setAlignment(Qt.AlignCenter)

        l.addWidget(val_label)
        l.addWidget(t_label)
        frame.val_label = val_label
        return frame

    # -------------------------------------------------------------
    # Data Population Methods
    # -------------------------------------------------------------
    def set_reports(self, analysis_report, preprocessing_report, dataset_info):
        self._analysis_report = analysis_report or {}
        self._preprocessing_report = preprocessing_report or {}
        self._dataset_info = dataset_info or {}

        if not self._analysis_report and not self._preprocessing_report and not self._dataset_info:
            self.clear()
            return

        name = self._dataset_info.get("dataset_name", "Active Dataset")
        self.status_badge.setText(f"Dataset: {name} (Loaded)")
        self.status_badge.setStyleSheet(
            f"background:#0d3326;color:{C['green']};border:1px solid #1c7352;border-radius:4px;padding:4px 10px;"
        )

        # Check for pipeline errors
        errors = self._preprocessing_report.get("pipeline_errors")
        if errors:
            err_text = f"Stage Errors: {', '.join(errors.keys())}"
            self.error_banner.setText(err_text)
            self.error_banner.show()
        else:
            self.error_banner.hide()

        self._populate_overview()
        self._populate_descriptive_stats()
        self._populate_correlation()
        self._populate_trends()
        self._populate_anomalies_and_extremes()
        self._populate_quality()

    def _populate_overview(self):
        variables = self._dataset_info.get("variables", {})
        n_vars = len(variables)
        self.card_vars.val_label.setText(str(n_vars))

        time_range = self._dataset_info.get("time_range", {})
        n_times = time_range.get("count", 0)
        self.card_times.val_label.setText(str(n_times) if n_times else "Static")

        val_analysis = self._preprocessing_report.get("validation", {}).get("analysis", {})
        quality_score = val_analysis.get("quality_score", "n/a")
        self.card_quality.val_label.setText(f"{quality_score}" if quality_score != "n/a" else "N/A")

        missing_data = self._preprocessing_report.get("missing_values", {}).get("analysis", {})
        tot_missing = sum(int(item.get("missing_after", 0)) for item in missing_data.values()) if missing_data else 0
        self.card_missing.val_label.setText(str(tot_missing))

        # Metadata table
        props = [
            ("Dataset Name", self._dataset_info.get("dataset_name", "")),
            ("Source File", self._dataset_info.get("file_path", "")),
            ("File Size", self._dataset_info.get("file_size", "")),
            ("Dimensions", str(self._dataset_info.get("dimensions", {}))),
            ("Time Period", f"{time_range.get('start', '')} to {time_range.get('end', '')}" if time_range else "N/A"),
            ("Coordinates", ", ".join(self._dataset_info.get("coordinate_names", []))),
            ("Quality Grade", val_analysis.get("dataset_quality", "N/A")),
            ("Model Readiness", "Approved" if val_analysis.get("ready_for_feature_engineering") else "Review Needed"),
        ]

        self.metadata_table.setRowCount(len(props))
        for row, (k, v) in enumerate(props):
            item_k = QTableWidgetItem(str(k))
            item_k.setFont(font(9, True))
            item_v = QTableWidgetItem(str(v))
            self.metadata_table.setItem(row, 0, item_k)
            self.metadata_table.setItem(row, 1, item_v)

    def _populate_descriptive_stats(self):
        stats_dict = self._analysis_report.get("statistics", {})
        self.stats_table.setRowCount(len(stats_dict))

        for row, (var_name, stat_entry) in enumerate(stats_dict.items()):
            summary = stat_entry.get("summary", {})
            vrange = stat_entry.get("range", {})

            vals = [
                var_name,
                self._format_num(summary.get("mean")),
                self._format_num(summary.get("median")),
                self._format_num(summary.get("std")),
                self._format_num(summary.get("variance")),
                self._format_num(vrange.get("min")),
                self._format_num(vrange.get("max")),
                self._format_num(summary.get("q1")),
                self._format_num(summary.get("q3")),
                self._format_num(summary.get("iqr")),
                str(summary.get("missing", 0)),
            ]

            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                if col == 0:
                    item.setFont(font(9, True))
                    item.setForeground(QColor(C["cyan"]))
                elif col == 10 and val != "0":
                    item.setForeground(QColor(C["amber"]))
                self.stats_table.setItem(row, col, item)

    def _populate_correlation(self):
        corr_info = self._analysis_report.get("correlation", {})
        matrix = corr_info.get("matrix")
        self.heatmap_widget.plot_matrix(matrix)

        strong_pos = corr_info.get("strong_positive", [])
        strong_neg = corr_info.get("strong_negative", [])
        all_strong = [(p["variable_1"], p["variable_2"], p["correlation"], "Strong Positive") for p in strong_pos]
        all_strong.extend([(p["variable_1"], p["variable_2"], p["correlation"], "Strong Negative") for p in strong_neg])

        self.correlation_table.setRowCount(len(all_strong))
        for row, (v1, v2, r, label) in enumerate(all_strong):
            item_v1 = QTableWidgetItem(v1)
            item_v2 = QTableWidgetItem(v2)
            item_r = QTableWidgetItem(f"{r:+.4f} ({label})")
            if r > 0:
                item_r.setForeground(QColor(C["cyan"]))
            else:
                item_r.setForeground(QColor(C["red"]))
            self.correlation_table.setItem(row, 0, item_v1)
            self.correlation_table.setItem(row, 1, item_v2)
            self.correlation_table.setItem(row, 2, item_r)

    def _populate_trends(self):
        trends = self._analysis_report.get("trend", {})
        self.trend_var_combo.blockSignals(True)
        self.trend_var_combo.clear()
        for var_name in trends.keys():
            self.trend_var_combo.addItem(var_name)
        self.trend_var_combo.blockSignals(False)

        if trends:
            first_var = list(trends.keys())[0]
            self._render_trend_plot(first_var)
        else:
            self.trend_figure.clear()
            self.trend_canvas.draw_idle()

    def _on_trend_variable_changed(self, var_name):
        if var_name:
            self._render_trend_plot(var_name)

    def _render_trend_plot(self, var_name):
        trends = self._analysis_report.get("trend", {})
        data = trends.get(var_name, {})
        series = data.get("series", [])
        m_avg = data.get("moving_average", [])
        slope = data.get("slope", 0.0)
        direction = data.get("trend", "Stable")

        # Update badges
        dir_color = C["green"] if direction == "Increasing" else (C["red"] if direction == "Decreasing" else C["dim"])
        self.trend_badge.setText(f"Trend: {direction}")
        self.trend_badge.setStyleSheet(
            f"background:{C['card']};color:{dir_color};border:1px solid {C['soft_border']};border-radius:4px;padding:4px 10px;"
        )
        self.slope_badge.setText(f"Slope: {slope:+.6f}/step")

        self.trend_figure.clear()
        if not series:
            ax = self.trend_figure.add_subplot(111)
            ax.set_facecolor("#121b27")
            ax.text(0.5, 0.5, "Insufficient time points for trend analysis", color="#92a5b5",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.trend_canvas.draw_idle()
            return

        ax = self.trend_figure.add_subplot(111)
        ax.set_facecolor("#121b27")
        x = np.arange(len(series))

        ax.plot(x, series, color=C["blue"], label="Domain Mean Series", lw=1.6, alpha=0.9)
        if m_avg:
            ax.plot(x, m_avg, color=C["cyan"], label="Moving Average", lw=2.0, ls="--")

        # Trendline
        if len(x) > 1:
            fit_line = slope * x + (series[0] if series else 0)
            ax.plot(x, fit_line, color=C["amber"], label=f"Linear Trend ({direction})", lw=1.5, ls=":")

        ax.set_title(f"Temporal Trend Analysis — {var_name}", color=C["txt"], fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Time Index", color=C["dim"], fontsize=9)
        ax.set_ylabel(f"{var_name} Value", color=C["dim"], fontsize=9)
        ax.tick_params(colors=C["dim"], labelsize=8)
        ax.grid(True, color="#273746", alpha=0.5, ls=":")
        ax.legend(facecolor="#182332", edgecolor="#2a3948", labelcolor=C["txt"], fontsize=8)

        for spine in ax.spines.values():
            spine.set_edgecolor("#2a3948")

        self.trend_figure.tight_layout()
        self.trend_canvas.draw_idle()

    def _populate_anomalies_and_extremes(self):
        # Anomalies
        anom_dict = self._analysis_report.get("anomalies", {})
        self.anomalies_table.setRowCount(len(anom_dict))
        for row, (var, info) in enumerate(anom_dict.items()):
            status = info.get("status", "Success")
            cnt = str(info.get("count", 0))
            pct = f"{info.get('percentage', 0.0):.2f}%"
            items = [
                var,
                self._format_num(info.get("mean")),
                self._format_num(info.get("std")),
                str(info.get("threshold", 2.0)),
                cnt,
                pct,
            ]
            for col, val in enumerate(items):
                item = QTableWidgetItem(val)
                if col == 0:
                    item.setFont(font(9, True))
                    item.setForeground(QColor(C["cyan"]))
                elif col in (4, 5) and cnt != "0":
                    item.setForeground(QColor(C["amber"]))
                self.anomalies_table.setItem(row, col, item)

        # Extremes
        ext_dict = self._analysis_report.get("extreme_weather", {})
        self.extremes_table.setRowCount(len(ext_dict))
        for row, (var, info) in enumerate(ext_dict.items()):
            events = info.get("extreme_events", [])
            high_str = "-"
            low_str = "-"
            for ev in events:
                if ev.get("type") == "High":
                    high_str = f"{ev.get('count', 0)} events (> {ev.get('threshold')})"
                elif ev.get("type") == "Low":
                    low_str = f"{ev.get('count', 0)} events (< {ev.get('threshold')})"

            items = [
                var,
                self._format_num(info.get("minimum")),
                self._format_num(info.get("maximum")),
                self._format_num(info.get("mean")),
                high_str,
                low_str,
            ]
            for col, val in enumerate(items):
                item = QTableWidgetItem(val)
                if col == 0:
                    item.setFont(font(9, True))
                    item.setForeground(QColor(C["cyan"]))
                elif col in (4, 5) and val != "-":
                    item.setForeground(QColor(C["red"]))
                self.extremes_table.setItem(row, col, item)

    def _populate_quality(self):
        val_analysis = self._preprocessing_report.get("validation", {}).get("analysis", {})
        dup_analysis = self._preprocessing_report.get("duplicates", {}).get("analysis", {})
        missing_dict = self._preprocessing_report.get("missing_values", {}).get("analysis", {})
        outlier_dict = self._preprocessing_report.get("outliers", {}).get("analysis", {})

        score = val_analysis.get("quality_score", "N/A")
        grade = val_analysis.get("dataset_quality", "N/A")
        ready = "YES" if val_analysis.get("ready_for_feature_engineering") else "NO"
        dups_removed = dup_analysis.get("duplicates_removed", 0)

        self.qc_score.val_label.setText(str(score))
        self.qc_grade.val_label.setText(str(grade))
        self.qc_ready.val_label.setText(ready)
        self.qc_dups.val_label.setText(str(dups_removed))

        # Missing values table
        self.missing_table.setRowCount(len(missing_dict))
        for row, (var, info) in enumerate(missing_dict.items()):
            items = [
                var,
                str(info.get("missing_before", 0)),
                str(info.get("missing_after", 0)),
                str(info.get("repaired_values", 0)),
                str(info.get("permanent_mask", 0)),
                str(info.get("method", "none")),
            ]
            for col, val in enumerate(items):
                item = QTableWidgetItem(val)
                if col == 0:
                    item.setFont(font(9, True))
                    item.setForeground(QColor(C["cyan"]))
                self.missing_table.setItem(row, col, item)

        # Outlier table
        self.outlier_table.setRowCount(len(outlier_dict))
        for row, (var, info) in enumerate(outlier_dict.items()):
            cnt = info.get("outlier_count", 0)
            pct = f"{info.get('outlier_percentage', 0.0):.2f}%"
            items = [
                var,
                str(info.get("method", "IQR")),
                str(info.get("action", "flag")),
                str(info.get("multiplier", 1.5)),
                str(cnt),
                pct,
                str(info.get("severity", "Low")),
                str(info.get("recommendation", "")),
            ]
            for col, val in enumerate(items):
                item = QTableWidgetItem(val)
                if col == 0:
                    item.setFont(font(9, True))
                    item.setForeground(QColor(C["cyan"]))
                elif col == 6:
                    sev = info.get("severity", "Low")
                    if sev == "High":
                        item.setForeground(QColor(C["red"]))
                    elif sev == "Moderate":
                        item.setForeground(QColor(C["amber"]))
                    else:
                        item.setForeground(QColor(C["green"]))
                self.outlier_table.setItem(row, col, item)

    def clear(self):
        self.status_badge.setText("No Dataset Loaded")
        self.status_badge.setStyleSheet(
            f"background:{C['card']};color:{C['dim']};border:1px solid {C['soft_border']};border-radius:4px;padding:4px 10px;"
        )
        self.error_banner.hide()
        for card in (self.card_vars, self.card_times, self.card_quality, self.card_missing,
                     self.qc_score, self.qc_grade, self.qc_ready, self.qc_dups):
            card.val_label.setText("-")

        for table in (self.metadata_table, self.stats_table, self.correlation_table,
                      self.anomalies_table, self.extremes_table, self.missing_table, self.outlier_table):
            table.setRowCount(0)

        self.heatmap_widget.clear()
        self.trend_var_combo.clear()
        self.trend_figure.clear()
        self.trend_canvas.draw_idle()

    def _format_num(self, val):
        if val is None or val == "":
            return "n/a"
        try:
            f = float(val)
            if np.isnan(f):
                return "NaN"
            return f"{f:.4f}"
        except (ValueError, TypeError):
            return str(val)
