from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.regression_metrics import ForecastEvaluator
from src.forecasting.feature_scaling import TabularFeatureScaler
from src.forecasting.forecast_view_model import ForecastViewModel
from src.forecasting.sequence_windowing import SequenceWindowGenerator
from src.forecasting.tabular_forecast_models import SEQUENCE_FORECAST_MODES, build_tabular_forecaster
from src.utils.constants import PROCESSED_DATA_DIR

TARGET = "rainfall_target_24h"


class TabularForecastManager:
    """Trains/predicts ML and DL forecasters on the feature-engineered
    train/test splits produced by FeaturePipeline (data/processed/*.csv).

    Training/test data is capped (MAX_TRAINING_ROWS etc.) before it reaches
    any model. A multi-year merged dataset can have millions of rows -
    unbounded Random Forest/XGBoost/etc. on that scale has been measured to
    need tens of GB of RAM, which crashes the app. Capping keeps every model
    usable regardless of how many years are merged; the actual sample size
    used is reported back in the metrics so it's never silently different
    from what the dataset actually contains.
    """

    REQUIRED_FILES = ("X_train.csv", "y_train.csv", "X_test.csv", "y_test.csv")

    MAX_TRAINING_ROWS = 250_000
    MAX_TEST_ROWS = 100_000
    MAX_TRAINING_SEQUENCES = 50_000
    MAX_TEST_SEQUENCES = 20_000

    def __init__(self):
        self.evaluator = ForecastEvaluator()
        self.sequence_generator = SequenceWindowGenerator()

    def run_forecast(
        self, mode, processed_dir=None, scale_features=None, sequence_length=6, progress_callback=None
    ):
        progress = progress_callback or (lambda message: None)
        processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DATA_DIR
        self._check_processed_files_exist(processed_dir)

        progress("Loading feature-engineered data...")
        X_train = pd.read_csv(processed_dir / "X_train.csv")
        y_train = pd.read_csv(processed_dir / "y_train.csv")
        X_test = pd.read_csv(processed_dir / "X_test.csv")
        y_test = pd.read_csv(processed_dir / "y_test.csv")

        if scale_features is None:
            scale_features = mode in SEQUENCE_FORECAST_MODES

        if scale_features:
            scaler = TabularFeatureScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        progress(f"Training {mode}...")
        if mode in SEQUENCE_FORECAST_MODES:
            forecaster, series, actual, predicted, test_index, sample_info = self._run_sequence_mode(
                mode, X_train, y_train, X_test, y_test, sequence_length
            )
        else:
            forecaster, series, actual, predicted, test_index, sample_info = self._run_row_mode(
                mode, X_train, y_train, X_test, y_test
            )

        progress("Evaluating...")
        metrics = self.evaluator.evaluate(actual, predicted)
        metrics["training_rows"] = sample_info

        return ForecastViewModel(
            variable=TARGET,
            units="mm",
            model_name=forecaster.name,
            time_labels=[str(index) for index in range(len(series))],
            series=[float(value) for value in series],
            test_index=test_index,
            actual=[float(value) for value in actual],
            predicted=[float(value) for value in predicted],
            future_index=[],
            future=[],
            metrics=metrics,
            region_bounds={},
        )

    def _run_row_mode(self, mode, X_train, y_train, X_test, y_test):
        numeric_train = X_train.select_dtypes(include="number")
        numeric_test = X_test.select_dtypes(include="number")
        y_train_values = y_train[TARGET].to_numpy(dtype=float)
        y_test_values = y_test[TARGET].to_numpy(dtype=float)

        total_train = len(numeric_train)

        train_idx = self._capped_indices(total_train, self.MAX_TRAINING_ROWS)
        if train_idx is not None:
            numeric_train = numeric_train.iloc[train_idx]
            y_train_values = y_train_values[train_idx]

        test_idx = self._capped_indices(len(numeric_test), self.MAX_TEST_ROWS)
        if test_idx is not None:
            numeric_test = numeric_test.iloc[test_idx]
            y_test_values = y_test_values[test_idx]

        forecaster = build_tabular_forecaster(mode)
        forecaster.fit(numeric_train, y_train_values)
        predicted = forecaster.predict(numeric_test)

        series = np.concatenate([y_train_values, y_test_values])
        test_index = list(range(len(y_train_values), len(y_train_values) + len(y_test_values)))
        sample_info = f"{len(numeric_train)} of {total_train} training rows"
        return forecaster, series, y_test_values, predicted, test_index, sample_info

    def _run_sequence_mode(self, mode, X_train, y_train, X_test, y_test, sequence_length):
        train_frame = X_train.assign(**{TARGET: y_train[TARGET].to_numpy(dtype=float)})
        test_frame = X_test.assign(**{TARGET: y_test[TARGET].to_numpy(dtype=float)})

        # max_sequences stops SequenceWindowGenerator as soon as it has enough
        # windows, instead of building every possible sequence for the whole
        # dataset and discarding most of it - for a multi-year merged grid
        # that difference is 10+ minutes and several GB of RAM vs. seconds.
        X_train_seq, y_train_seq, feature_columns = self.sequence_generator.make_sequences(
            train_frame, TARGET, sequence_length=sequence_length, max_sequences=self.MAX_TRAINING_SEQUENCES
        )
        X_test_seq, y_test_seq, _ = self.sequence_generator.make_sequences(
            test_frame,
            TARGET,
            sequence_length=sequence_length,
            feature_columns=feature_columns,
            max_sequences=self.MAX_TEST_SEQUENCES,
        )
        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            raise ValueError(
                "Not enough chronological rows per grid cell to build sequences "
                f"of length {sequence_length}."
            )

        forecaster = build_tabular_forecaster(mode, n_features=X_train_seq.shape[-1])
        forecaster.fit(X_train_seq, y_train_seq)
        predicted = forecaster.predict(X_test_seq)

        series = np.concatenate([y_train_seq, y_test_seq])
        test_index = list(range(len(y_train_seq), len(y_train_seq) + len(y_test_seq)))
        capped_note = " (capped for performance)" if len(X_train_seq) >= self.MAX_TRAINING_SEQUENCES else ""
        sample_info = f"{len(X_train_seq)} training sequences{capped_note}"
        return forecaster, series, y_test_seq, predicted, test_index, sample_info

    def _capped_indices(self, n, cap, seed=42):
        if n <= cap:
            return None
        return np.sort(np.random.RandomState(seed).choice(n, size=cap, replace=False))

    def _check_processed_files_exist(self, processed_dir):
        missing = [name for name in self.REQUIRED_FILES if not (processed_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                "Missing feature-engineered files in "
                f"{processed_dir}: {', '.join(missing)}. Open a dataset first so "
                "Feature Engineering can run, or select a different dataset."
            )
