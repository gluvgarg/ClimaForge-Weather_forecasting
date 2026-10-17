import sys

# torch must be imported before PyQt5 - on Windows, importing PyQt5 first
# poisons the DLL search path and torch's native libraries fail to load.
import torch

from PyQt5.QtWidgets import QApplication

from src.visualization.main_controller import MainController
from src.visualization.dashboard import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Weather Data Analysis System")

    window = MainWindow()
    controller = MainController(window)
    window.controller = controller
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
