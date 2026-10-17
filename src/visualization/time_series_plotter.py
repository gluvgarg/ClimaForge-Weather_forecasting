import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class PlotPanel(QWidget):
    """Matplotlib-backed variable preview plus basic statistics table."""

    def __init__(self):
        super().__init__()
        self.compact_mode = False
        self._last_data_array = None
        self._last_time_index = 0
        self._last_comparison_series = None
        self._last_statistics = {}
        self.figure = Figure(facecolor="#171b22")
        self.canvas = FigureCanvas(self.figure)

        self.stats_table = QTableWidget(0, 2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.setFont(QFont("Segoe UI", 9))
        self.stats_table.horizontalHeader().setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.setMaximumHeight(170)

        layout = QVBoxLayout(self)
        preview_label = QLabel("Variable Preview")
        preview_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        stats_label = QLabel("Quick Statistics")
        stats_label.setFont(QFont("Segoe UI", 10, QFont.Bold))

        layout.addWidget(preview_label)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(stats_label)
        layout.addWidget(self.stats_table)
        self.preview_label = preview_label
        self.stats_label = stats_label

    def set_compact_mode(self, compact):
        self.compact_mode = bool(compact)
        self.preview_label.setVisible(not compact)
        self.stats_label.setVisible(not compact)
        self.stats_table.setVisible(not compact)
        self.setMinimumHeight(190 if compact else 320)

    def plot_variable(self, data_array, time_index=0, comparison_series=None):
        self._last_data_array = data_array
        self._last_time_index = time_index
        self._last_comparison_series = comparison_series
        self.figure.clear()
        if self.compact_mode and comparison_series and comparison_series.get("series"):
            axis = self.figure.add_subplot(111)
            self._plot_comparison(axis, comparison_series, time_index)
            self.figure.subplots_adjust(left=0.13, right=0.96, top=0.84, bottom=0.24)
            self.canvas.draw_idle()
            return
        if comparison_series and comparison_series.get("series"):
            grid = self.figure.add_gridspec(2, 1, height_ratios=[1.4, 1.0], hspace=0.42)
            axis = self.figure.add_subplot(grid[0, 0])
            comparison_axis = self.figure.add_subplot(grid[1, 0])
        else:
            axis = self.figure.add_subplot(111)
            comparison_axis = None
        axis.set_facecolor("#10141a")
        axis.tick_params(colors="#dce3ea")
        axis.title.set_color("#f0f4f8")
        axis.xaxis.label.set_color("#dce3ea")
        axis.yaxis.label.set_color("#dce3ea")

        title = self._title(data_array, time_index)

        try:
            plot_array = self._prepare_plot_array(data_array, time_index)
        except ValueError as error:
            self._show_message(axis, title, str(error))
            self.canvas.draw_idle()
            return

        values = np.asarray(plot_array.values)

        if values.ndim == 0:
            axis.text(0.5, 0.5, str(values.item()), ha="center", va="center")
            axis.set_axis_off()
        elif values.ndim == 1:
            axis.plot(values, color="#5b8def", linewidth=1.8)
            axis.set_title(title)
            axis.set_xlabel(self._dimension_label(plot_array, 0))
            axis.set_ylabel(self._value_label(data_array))
            axis.grid(True, color="#2b3543", linewidth=0.6)
        elif values.ndim == 2:
            try:
                image_values = np.asarray(values, dtype=float)
            except (TypeError, ValueError):
                self._show_message(axis, title, "Selected 2D variable is not numeric.")
                self.canvas.draw_idle()
                return
            image = axis.imshow(image_values, cmap="viridis", aspect="auto")
            axis.set_title(title)
            axis.set_xlabel(self._dimension_label(plot_array, 1))
            axis.set_ylabel(self._dimension_label(plot_array, 0))
            colorbar = self.figure.colorbar(image, ax=axis, shrink=0.86)
            colorbar.set_label(self._value_label(data_array), color="#dce3ea")
            colorbar.ax.yaxis.set_tick_params(color="#dce3ea")
            for label in colorbar.ax.get_yticklabels():
                label.set_color("#dce3ea")
        else:
            self._show_message(axis, title, f"Unsupported plotted shape: {values.shape}")

        if comparison_axis is not None:
            self._plot_comparison(comparison_axis, comparison_series, time_index)

        self.figure.subplots_adjust(left=0.12, right=0.92, top=0.88, bottom=0.12, hspace=0.55)
        self.canvas.draw_idle()

    def set_statistics(self, statistics):
        self._last_statistics = dict(statistics)
        self.stats_table.setRowCount(len(statistics))
        for row, (metric, value) in enumerate(statistics.items()):
            self.stats_table.setItem(row, 0, QTableWidgetItem(str(metric)))
            self.stats_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.stats_table.resizeColumnsToContents()

    def clear(self):
        self.figure.clear()
        self.canvas.draw_idle()
        self.stats_table.setRowCount(0)
        self._last_data_array = None
        self._last_comparison_series = None
        self._last_statistics = {}

    def expanded_copy(self):
        panel = PlotPanel()
        panel.set_compact_mode(False)
        if self._last_data_array is not None:
            panel.plot_variable(self._last_data_array, self._last_time_index, self._last_comparison_series)
            panel.set_statistics(self._last_statistics)
        return panel

    def _prepare_plot_array(self, data_array, time_index):
        squeezed = data_array.squeeze(drop=True)
        if squeezed.ndim <= 2:
            return squeezed

        if squeezed.ndim == 3:
            time_dim = self._time_dimension(squeezed)
            if time_dim is None:
                time_dim = squeezed.dims[0]
            safe_index = max(0, min(int(time_index), squeezed.sizes[time_dim] - 1))
            sliced = squeezed.isel({time_dim: safe_index}).squeeze(drop=True)
            if sliced.ndim > 2:
                raise ValueError(f"Cannot reduce variable to 2D. Shape: {data_array.shape}")
            return sliced

        raise ValueError(f"Variables with more than 3 dimensions are not plotted yet. Shape: {data_array.shape}")

    def _time_dimension(self, data_array):
        for dim_name in data_array.dims:
            if dim_name.lower() in {"time", "valid_time", "forecast_time"}:
                return dim_name
        return None

    def _title(self, data_array, time_index):
        long_name = data_array.attrs.get("long_name", data_array.name or "Selected variable")
        units = data_array.attrs.get("units", "")
        suffix = f" [{units}]" if units else ""
        if data_array.ndim == 3:
            return f"{long_name}{suffix} - time slice {time_index}"
        return f"{long_name}{suffix}"

    def _dimension_label(self, data_array, axis_index):
        if axis_index >= len(data_array.dims):
            return ""
        dim_name = data_array.dims[axis_index]
        units = data_array.coords[dim_name].attrs.get("units", "") if dim_name in data_array.coords else ""
        return f"{dim_name} ({units})" if units else dim_name

    def _value_label(self, data_array):
        name = data_array.attrs.get("long_name", data_array.name or "value")
        units = data_array.attrs.get("units", "")
        return f"{name} ({units})" if units else name

    def _show_message(self, axis, title, message):
        axis.set_title(title)
        axis.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            color="#dce3ea",
            wrap=True,
            transform=axis.transAxes,
        )
        axis.set_axis_off()

    def _plot_comparison(self, axis, comparison_series, time_index):
        axis.set_facecolor("#10141a")
        axis.tick_params(colors="#dce3ea", labelsize=8)
        axis.title.set_color("#f0f4f8")
        axis.xaxis.label.set_color("#dce3ea")
        axis.yaxis.label.set_color("#dce3ea")
        axis.set_title("Multi-variable spatial mean across time", fontsize=9)
        colors = ["#49baff", "#ff684f", "#67d18d", "#ffab43", "#b48cff"]
        max_len = 0

        for index, series in enumerate(comparison_series.get("series", [])):
            values = np.asarray(series.get("values", []), dtype=float)
            if values.size == 0:
                continue
            x_values = np.arange(values.size)
            max_len = max(max_len, values.size)
            label = series.get("name", "")
            units = series.get("units", "")
            if units:
                label = f"{label} [{units}]"
            axis.plot(x_values, values, linewidth=1.4, color=colors[index % len(colors)], label=label)

        if max_len:
            safe_index = max(0, min(int(time_index), max_len - 1))
            axis.axvline(safe_index, color="#ffffff", linewidth=0.9, alpha=0.8)
            axis.set_xlim(0, max_len - 1 if max_len > 1 else 1)

        axis.set_xlabel("time index")
        axis.set_ylabel("spatial mean")
        axis.grid(True, color="#2b3543", linewidth=0.5, alpha=0.85)
        legend = axis.legend(loc="best", fontsize=7, facecolor="#10141a", edgecolor="#2b3543")
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("#dce3ea")
