from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QVBoxLayout, QWidget


class CorrelationHeatmapWidget(QWidget):
    """Matplotlib-backed correlation heatmap canvas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(facecolor="#121b27")
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def plot_matrix(self, corr_df):
        self.figure.clear()
        if corr_df is None or (isinstance(corr_df, pd.DataFrame) and corr_df.empty):
            ax = self.figure.add_subplot(111)
            ax.set_facecolor("#121b27")
            ax.text(
                0.5,
                0.5,
                "No correlation data available",
                color="#92a5b5",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        if not isinstance(corr_df, pd.DataFrame):
            corr_df = pd.DataFrame(corr_df)

        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#121b27")
        matrix_vals = corr_df.values.astype(float)
        cols = [str(c) for c in corr_df.columns]

        cax = ax.imshow(matrix_vals, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        cbar = self.figure.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="#e9f4ff", labelcolor="#e9f4ff")
        cbar.outline.set_edgecolor("#2a3948")

        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=35, ha="right", color="#e9f4ff", fontsize=9)
        ax.set_yticklabels(cols, color="#e9f4ff", fontsize=9)
        ax.tick_params(axis="both", colors="#e9f4ff")

        # Annotate correlation values in cells if matrix is reasonably sized
        if len(cols) <= 15:
            for i in range(len(cols)):
                for j in range(len(cols)):
                    val = matrix_vals[i, j]
                    if not np.isnan(val):
                        text_color = "#ffffff" if abs(val) > 0.55 else "#121b27"
                        ax.text(
                            j,
                            i,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=8,
                            weight="bold",
                        )

        for spine in ax.spines.values():
            spine.set_edgecolor("#2a3948")

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def clear(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#121b27")
        ax.set_axis_off()
        self.canvas.draw_idle()
