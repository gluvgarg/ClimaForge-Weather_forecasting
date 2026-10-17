from pathlib import Path

from src.data.loaders.netcdf_loader import NetCDFLoader
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline


def main():

    # --------------------------------------------------
    # Select dataset
    # --------------------------------------------------

    dataset_path = Path(
        "data/raw/RF25_ind2025_rfp25.nc"
    )

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    loader = NetCDFLoader()

    dataset = loader.load(dataset_path)

    # --------------------------------------------------
    # Run preprocessing pipeline
    # --------------------------------------------------

    pipeline = PreprocessingPipeline()

    cleaned_dataset, report = pipeline.run(dataset)

    # --------------------------------------------------
    # Print report
    # --------------------------------------------------

    report.pretty_print()

    # --------------------------------------------------
    # Save report
    # --------------------------------------------------

    report_path = Path(
        "data/interim/preprocessing_report.json"
    )

    report.save_json(report_path)

    print("\n")
    print("=" * 60)
    print("REPORT SAVED")
    print("=" * 60)
    print(report_path.resolve())

    # --------------------------------------------------
    # Show report dictionary
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("REPORT DICTIONARY")
    print("=" * 60)

    print(report.to_dict())

    dataset.close()
    cleaned_dataset.close()


if __name__ == "__main__":
    main()