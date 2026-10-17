from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loaders.netcdf_loader import NetCDFLoader
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.features.feature_pipeline import FeaturePipeline
from src.forecasting.forecast_manager import ForecastManager
from src.forecasting.forecast_models import SUPPORTED_MODES
from src.forecasting.machine_learning.random_forest import RandomForestForecaster
from src.forecasting.model_comparison_manager import ModelComparisonManager
from src.forecasting.tabular_forecast_manager import TabularForecastManager
from src.forecasting.tabular_forecast_models import TABULAR_FORECAST_MODES
from src.reporting.export_manager import ExportManager
from src.reporting.report_generator import ReportManager
from src.utils.constants import MODEL_TRAINED_DIR, PROCESSED_DATA_DIR


def main():

    dataset_path = Path("data/raw/RF25_ind2025_rfp25.nc")   # <-- change this

    print("=" * 60)
    print("Loading and preprocessing dataset...")
    print("=" * 60)

    loader = NetCDFLoader()
    dataset = loader.load(dataset_path)

    preprocessing = PreprocessingPipeline()
    clean_dataset, _ = preprocessing.run(dataset)

    variable_name = list(clean_dataset.data_vars)[0]
    data_array = clean_dataset[variable_name]

    print("\n" + "=" * 60)
    print("Statistical forecasters (univariate)...")
    print("=" * 60)

    forecast_manager = ForecastManager()
    for mode in SUPPORTED_MODES:
        forecast = forecast_manager.run_forecast(data_array, variable_name, mode=mode)
        print(mode, "->", forecast.metrics)

    print("\n" + "=" * 60)
    print("Running feature pipeline to generate data/processed/ splits...")
    print("=" * 60)

    feature_pipeline = FeaturePipeline()
    feature_pipeline.run(clean_dataset)

    print("\n" + "=" * 60)
    print("Tabular ML/DL forecasters...")
    print("=" * 60)

    tabular_manager = TabularForecastManager()
    for mode in TABULAR_FORECAST_MODES:
        try:
            forecast = tabular_manager.run_forecast(mode)
            print(mode, "->", forecast.metrics)
        except Exception as error:
            print(mode, "-> FAILED:", error)

    print("\n" + "=" * 60)
    print("Model comparison...")
    print("=" * 60)

    comparison_manager = ModelComparisonManager()
    report = comparison_manager.compare(
        SUPPORTED_MODES + TABULAR_FORECAST_MODES,
        data_array=data_array,
        variable_name=variable_name,
    )
    print(report.to_dict())

    export_manager = ExportManager()
    report_manager = ReportManager()
    print("Comparison CSV:", export_manager.export_comparison_csv(report.to_rows()))
    print("Comparison JSON:", export_manager.export_json(report.to_dict(), stem="model_comparison"))
    print("Comparison HTML:", report_manager.generate_model_comparison_report(report.to_rows()))

    print("\n" + "=" * 60)
    print("save() / load() round trip...")
    print("=" * 60)

    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv").select_dtypes(include="number")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv")["rainfall_target_24h"]
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv").select_dtypes(include="number")

    model = RandomForestForecaster()
    model.fit(X_train, y_train)
    save_path = model.save(MODEL_TRAINED_DIR / "random_forest_smoke_test.joblib")
    reloaded = RandomForestForecaster.load(save_path)

    original_predictions = model.predict(X_test)
    reloaded_predictions = reloaded.predict(X_test)
    print("Predictions match after reload:", np.allclose(original_predictions, reloaded_predictions))


if __name__ == "__main__":
    main()
