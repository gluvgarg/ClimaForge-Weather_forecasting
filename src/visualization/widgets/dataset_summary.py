from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget


class DatasetSummaryPanel(QWidget):
    """Readable scientific summary of the loaded dataset."""

    def __init__(self):
        super().__init__()
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Field", "Value"])
        self.tree.setFont(QFont("Segoe UI", 9))
        self.tree.header().setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.tree.setColumnWidth(0, 180)
        self.tree.setStyleSheet(
            """
            QTreeWidget {
                color: #d8e7f2;
                background: #101722;
                alternate-background-color: #142031;
            }
            QTreeWidget::item {
                color: #d8e7f2;
                min-height: 22px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree)

    def set_summary(self, summary):
        self.tree.clear()
        if not summary:
            self.tree.addTopLevelItem(self._item("Dataset", "No dataset loaded"))
            return

        file_item = self._section("File")
        file_item.addChild(self._item("Dataset Name", summary.get("dataset_name", "")))
        file_item.addChild(self._item("Path", summary.get("file_path", "")))
        file_item.addChild(self._item("Size", summary.get("file_size", "")))
        file_item.addChild(self._item("Variable Count", str(summary.get("variable_count", 0))))

        validation = summary.get("validation", {})
        validation_item = self._section("Validation")
        validation_item.addChild(self._item("Valid", validation.get("valid", "")))
        validation_item.addChild(self._item("Latitude Coord", validation.get("lat_name", "")))
        validation_item.addChild(self._item("Longitude Coord", validation.get("lon_name", "")))
        validation_item.addChild(self._item("Time Coord", validation.get("time_name", "")))
        for warning in validation.get("warnings", []):
            validation_item.addChild(self._item("Warning", warning))
        for error in validation.get("errors", []):
            validation_item.addChild(self._item("Error", error))

        time_range = summary.get("time_range", {})
        time_item = self._section("Time Coordinate")
        if time_range:
            time_item.addChild(self._item("Name", time_range.get("name", "")))
            time_item.addChild(self._item("Start", time_range.get("start", "")))
            time_item.addChild(self._item("End", time_range.get("end", "")))
            time_item.addChild(self._item("Count", time_range.get("count", "")))
        else:
            time_item.addChild(self._item("Status", "No explicit time coordinate"))

        dimensions_item = self._section("Dimensions")
        for name, size in summary.get("dimensions", {}).items():
            dimensions_item.addChild(self._item(name, str(size)))

        coordinates_item = self._section("Coordinate Ranges")
        for name, coord_range in summary.get("coordinates", {}).items():
            coordinates_item.addChild(
                self._item(name, f"{coord_range.get('min', '')} to {coord_range.get('max', '')}")
            )

        variables_item = self._section("Variables")
        for name, profile in summary.get("variables", {}).items():
            variable_item = self._item(name, profile.get("long_name", name))
            variable_item.addChild(self._item("Units", profile.get("units") or "unitless"))
            variable_item.addChild(self._item("Shape", " x ".join(str(item) for item in profile.get("shape", []))))
            variable_item.addChild(self._item("Dimensions", ", ".join(profile.get("dims", []))))
            variable_item.addChild(self._item("Data Type", profile.get("dtype", "")))
            variable_item.addChild(self._item("Valid Values", profile.get("valid_count", "")))
            variable_item.addChild(self._item("Missing Values", profile.get("missing_values", "")))
            variables_item.addChild(variable_item)

        attrs_item = self._section("Dataset Attributes")
        attrs = summary.get("attributes", {})
        if attrs:
            for name, value in attrs.items():
                attrs_item.addChild(self._item(name, value))
        else:
            attrs_item.addChild(self._item("Attributes", "None"))

        for index in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(index).setExpanded(True)
        self.tree.resizeColumnToContents(0)

    def clear(self):
        self.tree.clear()

    def _section(self, title):
        item = self._item(title, "")
        self.tree.addTopLevelItem(item)
        return item

    def _item(self, field, value):
        item = QTreeWidgetItem([str(field), str(value)])
        item.setForeground(0, QColor("#d8e7f2"))
        item.setForeground(1, QColor("#d8e7f2"))
        return item
