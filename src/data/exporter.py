from pathlib import Path


class PreprocessingExporter:

    def __init__(self, output_directory="data/interim"):

        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def export(
        self,
        dataset,
        report,
        dataset_name,
    ):

        dataset_name = Path(dataset_name).stem

        csv_path = (
            self.output_directory
            / f"{dataset_name}_clean.csv"
        )

        preprocessing_path = (
            self.output_directory
            / f"{dataset_name}_preprocessing_report.json"
        )

        # ---------------------------------------
        # Save Clean Dataset (CSV)
        # ---------------------------------------

        dataset.to_dataframe().reset_index().to_csv(
            csv_path,
            index=False,
        )

        # ---------------------------------------
        # Save Preprocessing Report
        # ---------------------------------------

        report.save_json(preprocessing_path)

        return {

            "clean_dataset_csv": csv_path,

            "preprocessing_report": preprocessing_path,

        }