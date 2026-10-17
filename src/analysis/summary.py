class AnalysisSummary:
    """
    Generates a high-level summary
    from the analysis report.
    """

    def generate(self, report):

        metadata = report["metadata"]
        statistics = report["statistics"]

        total_variables = len(statistics)

        variable_names = list(statistics.keys())

        total_missing = 0

        for variable in statistics.values():

            total_missing += (
                variable["summary"].get("missing", 0)
            )

        return {

            "dataset_name":
                metadata.get("dataset_name"),

            "dimensions":
                metadata.get("dimensions"),

            "coordinates":
                metadata.get("coordinates"),

            "variables":
                variable_names,

            "number_of_variables":
                total_variables,

            "total_missing_values":
                total_missing,

            "correlation_analysis":
                "correlation" in report,

            "trend_analysis":
                "trend" in report,

            "seasonality_analysis":
                "seasonality" in report,

            "extreme_weather_analysis":
                "extreme_weather" in report,

            "anomaly_detection":
                "anomalies" in report,
        }