import numpy as np


class QualityValidator:
    """
    Final quality assessment after preprocessing.

    This stage verifies that the dataset is suitable
    for downstream Feature Engineering.
    """

    MAX_ALLOWED_MISSING = 40.0

    def process(self, dataset, report):

        analysis = {}

        total_variables = len(dataset.data_vars)

        numeric_variables = 0
        variables_with_units = 0

        variable_statistics = {}

        missing_percentages = []

        # ---------------------------------------------
        # Variable Level Validation
        # ---------------------------------------------

        for name, variable in dataset.data_vars.items():

            is_numeric = np.issubdtype(variable.dtype, np.number)

            if is_numeric:
                numeric_variables += 1

            has_units = (
                "units" in variable.attrs
                and str(variable.attrs["units"]).strip() != ""
            )

            if has_units:
                variables_with_units += 1

            total = variable.size

            missing = int(variable.isnull().sum())

            percentage = (
                (missing / total) * 100
                if total > 0 else 0
            )

            missing_percentages.append(percentage)

            variable_statistics[name] = {

                "numeric": bool(is_numeric),

                "has_units": bool(has_units),

                "missing_percentage": round(
                    percentage,
                    2,
                ),
            }

        # ---------------------------------------------
        # Coordinate Validation
        # ---------------------------------------------

        coordinate_report = {

            "time": False,

            "latitude": False,

            "longitude": False,

        }

        for coord in dataset.coords:

            lower = coord.lower()

            if lower in [
                "time",
                "valid_time",
                "forecast_time",
            ]:
                coordinate_report["time"] = True

            elif lower in [
                "latitude",
                "lat",
            ]:
                coordinate_report["latitude"] = True

            elif lower in [
                "longitude",
                "lon",
            ]:
                coordinate_report["longitude"] = True

        # ---------------------------------------------
        # Missing Value Statistics
        # ---------------------------------------------

        average_missing = (
            np.mean(missing_percentages)
            if missing_percentages
            else 0.0
        )

        # ---------------------------------------------
        # Quality Score
        # ---------------------------------------------

        score = 100

        score -= average_missing * 0.20

        if numeric_variables != total_variables:
            score -= 10

        if variables_with_units != total_variables:
            score -= 5

        if not coordinate_report["time"]:
            score -= 10

        if not coordinate_report["latitude"]:
            score -= 10

        if not coordinate_report["longitude"]:
            score -= 10

        score = max(0, round(score, 2))

        # ---------------------------------------------
        # Ready?
        # ---------------------------------------------

        ready = (

            score >= 70

            and average_missing <= self.MAX_ALLOWED_MISSING

            and coordinate_report["time"]

            and coordinate_report["latitude"]

            and coordinate_report["longitude"]

        )

        # ---------------------------------------------
        # Dataset Grade
        # ---------------------------------------------

        if score >= 90:
            grade = "Excellent"

        elif score >= 75:
            grade = "Good"

        elif score >= 60:
            grade = "Fair"

        else:
            grade = "Poor"

        # ---------------------------------------------
        # Report
        # ---------------------------------------------

        analysis["quality_score"] = float(score)

        analysis["dataset_quality"] = grade

        analysis["ready_for_feature_engineering"] = bool(ready)

        analysis["total_variables"] = total_variables

        analysis["numeric_variables"] = numeric_variables

        analysis["variables_with_units"] = variables_with_units

        analysis["average_missing_percentage"] = round(
            average_missing,
            2,
        )

        analysis["coordinates"] = coordinate_report

        analysis["variable_statistics"] = variable_statistics

        report.add(
            "validation",
            "analysis",
            analysis,
        )

        return dataset