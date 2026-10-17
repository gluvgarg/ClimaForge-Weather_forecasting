import json
from pathlib import Path

import pandas as pd
import xarray as xr

from src.features.feature_summary import FeatureSummary
from src.preprocessing.scaler import DataScaler


class TabularFeatureScaler:
    """Apply DataScaler's statistical scaling to tabular ML/DL feature frames.

    DataScaler is fit once on X_train only (to avoid train/val/test leakage);
    the resulting per-column statistics are then re-applied to X_val/X_test.
    Coordinate-like columns (TIME/LATITUDE/LONGITUDE aliases) are left
    untouched because DataScaler already excludes them by name.
    """

    def __init__(self, method="standard"):
        self.method = method
        self.state = {}

    def fit_transform(self, X_train):
        numeric_columns = [
            column for column in X_train.columns if pd.api.types.is_numeric_dtype(X_train[column])
        ]
        train_dataset = xr.Dataset(
            {column: ("row", X_train[column].to_numpy(dtype=float)) for column in numeric_columns}
        )
        report = FeatureSummary()
        scaled_dataset = DataScaler(self.method).process(train_dataset, report)
        analysis = report.to_dict().get("scaling", {}).get("analysis", {})

        self.state = {
            "method": self.method,
            "columns": {column: analysis[column] for column in numeric_columns if column in analysis},
        }

        transformed = X_train.copy()
        for column in numeric_columns:
            transformed[column] = scaled_dataset[column].values
        return transformed

    def transform(self, X):
        if not self.state:
            raise RuntimeError("TabularFeatureScaler must be fit before calling transform().")

        transformed = X.copy()
        for column, info in self.state["columns"].items():
            if column not in transformed.columns or not info.get("scaled"):
                continue
            values = transformed[column].to_numpy(dtype=float)
            if self.state["method"] == "standard":
                transformed[column] = (values - info["mean"]) / info["std"]
            else:
                transformed[column] = (values - info["minimum"]) / (info["maximum"] - info["minimum"])
        return transformed

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.state, file, indent=2)
        return path

    def load(self, path):
        with Path(path).open("r", encoding="utf-8") as file:
            self.state = json.load(file)
        return self
