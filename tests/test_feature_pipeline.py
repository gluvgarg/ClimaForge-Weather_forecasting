from pathlib import Path

from src.data.loaders.netcdf_loader import NetCDFLoader
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.features.feature_pipeline import FeaturePipeline


def main():

    dataset_path = Path("data/raw/sample_weather.nc")   # <-- change this

    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    loader = NetCDFLoader()
    dataset = loader.load(dataset_path)

    print(dataset)

    print("\n" + "=" * 60)
    print("Running preprocessing...")
    print("=" * 60)

    preprocessing = PreprocessingPipeline()

    clean_dataset, preprocessing_report = preprocessing.run(dataset)

    print("Preprocessing completed.")

    print("\n" + "=" * 60)
    print("Running feature pipeline...")
    print("=" * 60)

    feature_pipeline = FeaturePipeline()

    featured_dataset, feature_report = feature_pipeline.run(clean_dataset)

    print("\nFeature pipeline completed successfully.")

    print("\nEngineered variables:")
    print(list(featured_dataset.data_vars))

    print("\nFeature report:")
    print(feature_report.to_dict())

    print("\nGenerated files should exist in:")
    print("data/processed/")


if __name__ == "__main__":
    main()