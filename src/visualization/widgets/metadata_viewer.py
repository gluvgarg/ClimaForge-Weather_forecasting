from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem


class MetadataViewer(QTreeWidget):
    """Tree view for dimensions, coordinates, variables, and attributes."""

    def __init__(self):
        super().__init__()
        self.setHeaderLabels(["Property", "Value"])
        self.setColumnWidth(0, 190)
        self.setFont(QFont("Segoe UI", 9))
        self.header().setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.setStyleSheet(
            """
            QTreeWidget {
                font-family: "Segoe UI";
                font-size: 12px;
            }
            QTreeWidget::item {
                min-height: 22px;
            }
            """
        )

    def set_metadata(self, metadata):
        self.clear()
        for section, content in metadata.items():
            section_item = QTreeWidgetItem([section.title(), ""])
            self.addTopLevelItem(section_item)
            self._add_value(section_item, content)
            section_item.setExpanded(True)
        self.resizeColumnToContents(0)

    def _add_value(self, parent, value):
        if isinstance(value, dict):
            for key, child_value in value.items():
                item = QTreeWidgetItem([str(key), self._preview(child_value)])
                parent.addChild(item)
                if isinstance(child_value, (dict, list, tuple)):
                    self._add_value(item, child_value)
        elif isinstance(value, (list, tuple)):
            for index, child_value in enumerate(value):
                item = QTreeWidgetItem([str(index), self._preview(child_value)])
                parent.addChild(item)
                if isinstance(child_value, (dict, list, tuple)):
                    self._add_value(item, child_value)

    def _preview(self, value):
        if isinstance(value, dict):
            return f"{len(value)} entries"
        if isinstance(value, (list, tuple)):
            return ", ".join(str(item) for item in value[:6])
        return str(value)
