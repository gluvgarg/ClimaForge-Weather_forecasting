from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class DataSliceGrid(QWidget):
    """Tabular inspection of the selected variable/time geospatial slice."""

    def __init__(self):
        super().__init__()
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Latitude", "Longitude", "Value", "Units", "Anomaly"])
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
        layout.addWidget(self.table)

    def set_rows(self, rows):
        self.table.setRowCount(len(rows))
        for row_index, row_values in enumerate(rows):
            for col_index, value in enumerate(row_values):
                self.table.setItem(row_index, col_index, QTableWidgetItem(str(value)))
                self.table.item(row_index, col_index).setForeground(QColor("#d8e7f2"))
        self.table.resizeColumnsToContents()

    def clear(self):
        self.table.setRowCount(0)
