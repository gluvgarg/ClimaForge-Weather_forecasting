from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import QFrame, QGridLayout, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from src.visualization.widgets.dashboard_visuals import C


class MetricsPanel(QWidget):
    """Dataset-derived live metric summary for the selected time slice."""

    def __init__(self):
        super().__init__()
        self.cards = {}
        self.table = QTableWidget(0, 5)
        self.setMinimumHeight(215)
        self.table.setMaximumHeight(86)
        self.table.setHorizontalHeaderLabels(["Metric", "Mean", "Units", "Min", "Max"])
        self.table.setFont(QFont("Segoe UI", 9))
        self.table.horizontalHeader().setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            """
            QTableWidget {
                color: #d8e7f2;
                alternate-background-color: #142031;
                background: #101722;
                gridline-color: #2c3c4d;
            }
            QTableWidget::item {
                color: #d8e7f2;
                padding: 3px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.card_grid = QGridLayout()
        self.card_grid.setContentsMargins(0, 0, 0, 0)
        self.card_grid.setSpacing(8)
        layout.addLayout(self.card_grid)
        layout.addWidget(self.table)
        self._build_cards()

    def set_metrics(self, metrics):
        by_label = {metric.get("label", ""): metric for metric in metrics}
        for label, card in self.cards.items():
            metric = by_label.get(label, {})
            value = metric.get("value", "N/A")
            units = metric.get("units", "")
            card["value"].setText(f"{value} {units}".strip())
            card["range"].setText(
                f"{metric.get('min', 'N/A')} to {metric.get('max', 'N/A')}"
                if metric
                else "No dataset variable"
            )

        self.table.setRowCount(len(metrics))
        for row, metric in enumerate(metrics):
            values = [
                metric.get("label", ""),
                metric.get("value", ""),
                metric.get("units", ""),
                metric.get("min", ""),
                metric.get("max", ""),
            ]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(str(value)))
                self.table.item(row, col).setForeground(QColor("#d8e7f2"))
        self.table.resizeColumnsToContents()

    def clear(self):
        self.table.setRowCount(0)
        for card in self.cards.values():
            card["value"].setText("N/A")
            card["range"].setText("No dataset variable")

    def _build_cards(self):
        for index, label in enumerate(["Humidity", "Pressure", "Wind Speed", "Precipitation"]):
            card = QFrame()
            card.setMinimumHeight(76)
            card.setStyleSheet(
                f"""
                QFrame{{
                    background:#111a25;
                    border:1px solid {C['soft_border']};
                    border-radius:6px;
                }}
                QLabel{{background:transparent;border:none;}}
                """
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 7, 8, 7)
            title = QLabel(label)
            title.setFont(QFont("Segoe UI", 8, QFont.Bold))
            title.setStyleSheet(f"color:{C['dim']};")
            value = QLabel("N/A")
            value.setAlignment(Qt.AlignCenter)
            value.setFont(QFont("Segoe UI", 13, QFont.Bold))
            value.setStyleSheet("color:#f0f6ff;")
            value.setMinimumHeight(32)
            range_label = QLabel("No dataset variable")
            range_label.setAlignment(Qt.AlignCenter)
            range_label.setFont(QFont("Segoe UI", 8))
            range_label.setStyleSheet(f"color:{C['muted']};")
            card_layout.addWidget(title)
            card_layout.addWidget(value)
            card_layout.addWidget(range_label)
            self.card_grid.addWidget(card, index // 2, index % 2)
            self.cards[label] = {"value": value, "range": range_label}
