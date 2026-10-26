from pathlib import Path

import pandas as pd

from src.features.feature_summary import FeatureSummary
from src.features.seasonal_features import TemporalFeatureGenerator
from src.features.meteorological_features import MeteorologicalFeatureGenerator
from src.features.lag_features import LagFeatureGenerator
from src.features.rolling_features import RollingFeatureGenerator
from src.features.target_generator import TargetGenerator


class FeaturePipeline:
    """
    Feature Engineering Pipeline

    Input
    -----
    Clean xarray Dataset

    Output
    ------
    data/processed/

        features.nc
        features.csv

        X_train.csv
        X_val.csv
        X_test.csv

        y_train.csv
        y_val.csv
        y_test.csv

        feature_report.json
    """

    TARGET = "rainfall_target_24h"

    def __init__(self):

        self.stages = [

            ("Temporal Features", TemporalFeatureGenerator()),

            ("Meteorological Features", MeteorologicalFeatureGenerator()),

            ("Lag Features", LagFeatureGenerator()),

            ("Rolling Features", RollingFeatureGenerator()),

            ("Target Generation", TargetGenerator()),

        ]

    def run(self, dataset, output_dir=None):

        report = FeatureSummary()

        featured = dataset.copy(deep=True)

        # ----------------------------------------
        # Feature Generation
        # ----------------------------------------

        for stage_name, stage in self.stages:

            print(f"Running {stage_name}")

            featured = stage.process(
                featured,
                report,
            )

        # ----------------------------------------
        # Export Directory
        # ----------------------------------------

        output_dir = Path(output_dir) if output_dir is not None else Path("data/processed")
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------
        # Save Engineered Dataset
        # ----------------------------------------

        print("Saving features.nc")
        featured.to_netcdf(
            output_dir / "features.nc"
        )
        print("Saved features.nc")

        # ----------------------------------------
        # Convert to DataFrame ONLY ONCE
        # ----------------------------------------

        df = (
            featured
            .to_dataframe()
            .reset_index()
        )

        print(df.columns.tolist())
        print("Target exists:", self.TARGET in df.columns)

        # remove rows where target unavailable

        if self.TARGET in df.columns:

            df = df.dropna(
                subset=[self.TARGET]
            )

        df.to_csv(
            output_dir / "features.csv",
            index=False,
        )

        # ----------------------------------------
        # Train / Validation / Test
        # ----------------------------------------

        if self.TARGET in df.columns:
            total = len(df)
            time_cols = [c for c in df.columns if c.lower() in {"time", "valid_time", "forecast_time"}]
            time_col = time_cols[0] if time_cols else None

            if time_col is not None and df[time_col].nunique() >= 3:
                unique_times = pd.Series(df[time_col].unique()).sort_values().values
                n_times = len(unique_times)
                t_train_end = max(1, min(int(n_times * 0.70), n_times - 2))
                t_val_end = max(t_train_end + 1, min(int(n_times * 0.85), n_times - 1))

                train_times = set(unique_times[:t_train_end])
                val_times = set(unique_times[t_train_end:t_val_end])
                test_times = set(unique_times[t_val_end:])

                train_df = df[df[time_col].isin(train_times)]
                val_df = df[df[time_col].isin(val_times)]
                test_df = df[df[time_col].isin(test_times)]

                X_train = train_df.drop(columns=[self.TARGET])
                y_train = train_df[[self.TARGET]]
                X_val = val_df.drop(columns=[self.TARGET])
                y_val = val_df[[self.TARGET]]
                X_test = test_df.drop(columns=[self.TARGET])
                y_test = test_df[[self.TARGET]]

                split_strategy = "chronological_time"
            else:
                total = len(df)
                train_end = int(total * 0.70)
                val_end = int(total * 0.85)

                X = df.drop(columns=[self.TARGET])
                y = df[[self.TARGET]]

                X_train = X.iloc[:train_end]
                X_val = X.iloc[train_end:val_end]
                X_test = X.iloc[val_end:]

                y_train = y.iloc[:train_end]
                y_val = y.iloc[train_end:val_end]
                y_test = y.iloc[val_end:]

                split_strategy = "sequential_rows"

            X_train.to_csv(
                output_dir / "X_train.csv",
                index=False,
            )

            X_val.to_csv(
                output_dir / "X_val.csv",
                index=False,
            )

            X_test.to_csv(
                output_dir / "X_test.csv",
                index=False,
            )

            y_train.to_csv(
                output_dir / "y_train.csv",
                index=False,
            )

            y_val.to_csv(
                output_dir / "y_val.csv",
                index=False,
            )

            y_test.to_csv(
                output_dir / "y_test.csv",
                index=False,
            )

            report.add(
                "dataset_split",
                "split_strategy",
                split_strategy,
            )

            report.add(
                "dataset_split",
                "total_rows",
                total,
            )

            report.add(
                "dataset_split",
                "train_rows",
                len(X_train),
            )

            report.add(
                "dataset_split",
                "validation_rows",
                len(X_val),
            )

            report.add(
                "dataset_split",
                "test_rows",
                len(X_test),
            )

            report.add(
                "dataset_split",
                "feature_columns",
                list(X_train.columns),
            )

            report.add(
                "dataset_split",
                "target_column",
                self.TARGET,
            )

        report.save_json(
            output_dir / "feature_report.json"
        )

        return featured, report