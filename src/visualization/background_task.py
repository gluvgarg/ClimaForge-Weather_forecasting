from PyQt5.QtCore import QThread, pyqtSignal


class BackgroundTask(QThread):
    """Runs one callable off the GUI thread so long operations (dataset
    load/merge, model training, comparison) no longer freeze the window.

    The wrapped function must accept a `progress_callback` keyword argument;
    every backend method this is used with already does (see DatasetManager,
    ForecastManager, TabularForecastManager, ModelComparisonManager).
    """

    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, progress_callback=self.progress.emit, **self.kwargs)
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.finished_ok.emit(result)
