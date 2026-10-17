from pathlib import Path
import json
import csv


class AnalysisExporter:
    """
    Exports analysis results.

    Outputs
    -------
    reports/
        analysis_summary.json
        descriptive_statistics.csv
    """

    def __init__(self, output_directory="reports"):

        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def export_summary(self, report):

        output_path = (
            self.output_directory
            / "analysis_summary.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                report,
                f,
                indent=4,
                default=str,
            )

    def export_statistics_csv(self, report):

        output_path = (
            self.output_directory
            / "descriptive_statistics.csv"
        )

        with open(
            output_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:

            writer = csv.writer(csvfile)

            writer.writerow([
                "Variable",
                "Mean",
                "Median",
                "Std",
                "Variance",
                "Minimum",
                "Maximum",
                "Q1",
                "Q3",
                "IQR",
                "Missing Values",
            ])

            for variable, values in report["statistics"].items():

                summary = values["summary"]
                variable_range = values["range"]

                writer.writerow([
                    variable,
                    summary.get("mean"),
                    summary.get("median"),
                    summary.get("std"),
                    summary.get("variance"),
                    variable_range.get("min"),
                    variable_range.get("max"),
                    summary.get("q1"),
                    summary.get("q3"),
                    summary.get("iqr"),
                    summary.get("missing"),
                ])