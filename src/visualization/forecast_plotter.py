from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.visualization.widgets.dashboard_visuals import C, font


class ForecastPanel(QWidget):
    """Render Phase 1 univariate forecast outputs."""

    def __init__(self):
        super().__init__()
        self.figure = Figure(facecolor=C["panel"])
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(420)
        self.summary_label = QLabel("No forecast has been generated yet.")
        self.summary_label.setFont(font(11, True))
        self.summary_label.setStyleSheet(f"color:{C['txt']};padding:4px 0;")

        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.setFont(QFont("Segoe UI", 10))
        self.metrics_table.horizontalHeader().setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.verticalHeader().hide()
        self.metrics_table.setMinimumWidth(260)
        self.metrics_table.setMinimumHeight(155)
        self.metrics_table.setStyleSheet(
            f"QTableWidget{{color:{C['txt']};background:#111822;gridline-color:#273746;}}"
            f"QHeaderView::section{{color:{C['txt']};background:#1b2430;padding:5px;}}"
        )

        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(["Index / Timestamp", "Actual", "Predicted", "Future", "Error"])
        self.results_table.setFont(QFont("Segoe UI", 10))
        self.results_table.horizontalHeader().setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.verticalHeader().hide()
        self.results_table.setMinimumHeight(155)
        self.results_table.setStyleSheet(
            f"QTableWidget{{color:{C['txt']};background:#111822;gridline-color:#273746;}}"
            f"QHeaderView::section{{color:{C['txt']};background:#1b2430;padding:5px;}}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.canvas, 5)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)
        bottom_row.addWidget(self.metrics_table, 1)
        bottom_row.addWidget(self.results_table, 4)
        layout.addLayout(bottom_row, 2)
        self.clear()

    def set_forecast(self, forecast):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_facecolor("#10141a")
        axis.tick_params(colors="#dce3ea", labelsize=9)
        axis.title.set_color("#f0f4f8")
        axis.xaxis.label.set_color("#dce3ea")
        axis.yaxis.label.set_color("#dce3ea")

        x_values = list(range(len(forecast.series)))
        if x_values:
            train_end = forecast.test_index[0] if forecast.test_index else len(x_values)
            axis.axvspan(0, max(0, train_end), color="#49baff", alpha=0.08, label="history/train region")
            if forecast.test_index:
                axis.axvspan(forecast.test_index[0], forecast.test_index[-1] + 1, color="#ffab43", alpha=0.08, label="test region")
            if forecast.future_index:
                axis.axvspan(forecast.future_index[0], forecast.future_index[-1] + 1, color="#ff684f", alpha=0.08, label="future region")
                axis.axvline(forecast.future_index[0], color="#ff684f", linestyle=":", linewidth=1.4)

        axis.plot(x_values, forecast.series, color="#49baff", linewidth=2.0, label="historical regional series")
        if forecast.actual:
            axis.plot(forecast.test_index, forecast.actual, color="#67d18d", linewidth=2.2, marker="o", markersize=4, label="actual test values")
        if forecast.predicted:
            axis.plot(forecast.test_index, forecast.predicted, color="#ffab43", linewidth=2.2, marker="o", markersize=4, label="predicted test values")
        if forecast.future:
            axis.plot(forecast.future_index, forecast.future, color="#ff684f", linewidth=2.2, linestyle="--", marker="o", markersize=4, label="future forecast")

        units = f" ({forecast.units})" if forecast.units else ""
        region = "Selected Region" if forecast.region_bounds else "Full Dataset Domain"
        axis.set_title(f"Regional {forecast.variable} Forecast - {region}", fontsize=13, pad=12)
        axis.set_xlabel("time index")
        axis.set_ylabel(f"{forecast.variable}{units}")
        axis.grid(True, color="#2b3543", linewidth=0.55)
        legend = axis.legend(loc="best", fontsize=9, facecolor="#10141a", edgecolor="#2b3543")
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("#dce3ea")
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.13)
        self.canvas.draw_idle()
        self.summary_label.setText(
            f"{forecast.model_name} forecast for {forecast.variable}{units} | "
            f"{len(forecast.series)} historical steps | {len(forecast.future)} future steps"
        )

        rows = list(forecast.metrics.items())
        rows.extend(
            [
                ("model", forecast.model_name),
                ("future_steps", len(forecast.future)),
                ("region", forecast.region_bounds or "full domain"),
            ]
        )
        self.metrics_table.setRowCount(len(rows))
        for row, (metric, value) in enumerate(rows):
            metric_item = QTableWidgetItem(str(metric))
            value_item = QTableWidgetItem(str(value))
            metric_item.setForeground(QColor(C["txt"]))
            value_item.setForeground(QColor(C["txt"]))
            self.metrics_table.setItem(row, 0, metric_item)
            self.metrics_table.setItem(row, 1, value_item)
        self.metrics_table.resizeColumnsToContents()
        self._populate_results_table(forecast)

    def clear(self):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_facecolor("#10141a")
        axis.text(
            0.5,
            0.5,
            "Run a Phase 1 forecast to view actual vs predicted series",
            ha="center",
            va="center",
            color="#dce3ea",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        self.canvas.draw_idle()
        self.summary_label.setText("No forecast has been generated yet.")
        self.metrics_table.setRowCount(0)
        self.results_table.setRowCount(0)

    def _populate_results_table(self, forecast):
        rows = []
        label_lookup = {
            index: forecast.time_labels[index]
            for index in range(min(len(forecast.time_labels), len(forecast.series)))
        }

        for index, actual, predicted in zip(forecast.test_index, forecast.actual, forecast.predicted):
            label = label_lookup.get(index, str(index))
            rows.append([label, actual, predicted, "", actual - predicted])

        for index, value in zip(forecast.future_index, forecast.future):
            rows.append([f"future +{index - len(forecast.series) + 1}", "", "", value, ""])

        self.results_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                if isinstance(value, float):
                    value = f"{value:.4f}"
                item = QTableWidgetItem(str(value))
                item.setForeground(QColor(C["txt"]))
                self.results_table.setItem(row, column, item)


class ModelComparisonPanel(QWidget):
    """Render side-by-side evaluation metrics for multiple forecast models."""

    COLUMNS = ["model", "mse", "rmse", "mae", "mape", "r2"]

    def __init__(self):
        super().__init__()
        self.summary_label = QLabel("No model comparison has been run yet.")
        self.summary_label.setFont(font(11, True))
        self.summary_label.setStyleSheet(f"color:{C['txt']};padding:4px 0;")

        self.comparison_table = QTableWidget(0, len(self.COLUMNS))
        self.comparison_table.setHorizontalHeaderLabels(["Model", "MSE", "RMSE", "MAE", "MAPE", "R2"])
        self.comparison_table.setFont(QFont("Segoe UI", 10))
        self.comparison_table.horizontalHeader().setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.comparison_table.verticalHeader().hide()
        self.comparison_table.setMinimumHeight(160)
        self.comparison_table.setStyleSheet(
            f"QTableWidget{{color:{C['txt']};background:#111822;gridline-color:#273746;}}"
            f"QHeaderView::section{{color:{C['txt']};background:#1b2430;padding:5px;}}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.comparison_table)
        self.clear()

    def set_comparison(self, report):
        rows = report.to_rows()
        errors = report.to_dict().get("errors", {}).get("failed_models", {})

        self.comparison_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, key in enumerate(self.COLUMNS):
                item = QTableWidgetItem(str(row.get(key, "")))
                item.setForeground(QColor(C["txt"]))
                self.comparison_table.setItem(row_index, column_index, item)

        summary = f"{len(rows)} model(s) compared"
        if errors:
            summary += f" | {len(errors)} failed: {', '.join(errors.keys())}"
        self.summary_label.setText(summary)

    def clear(self):
        self.comparison_table.setRowCount(0)
        self.summary_label.setText("No model comparison has been run yet.")
