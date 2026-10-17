import xarray as xr


class DuplicateHandler:
    """
    Detect and remove duplicate weather observations.

    Duplicate observations are identified using the spatial-temporal
    coordinates (time, latitude, longitude).

    The dataset is temporarily converted to a DataFrame for duplicate
    detection and then converted back to an xarray.Dataset.
    """

    TIME_COLUMNS = {
        "time",
        "valid_time",
        "forecast_time",
        "TIME",
    }

    LAT_COLUMNS = {
        "latitude",
        "lat",
        "LATITUDE",
    }

    LON_COLUMNS = {
        "longitude",
        "lon",
        "LONGITUDE",
    }

    # -----------------------------------------------------

    def process(self, dataset: xr.Dataset, report):

        dataframe = dataset.to_dataframe().reset_index()

        rows_before = len(dataframe)

        coordinate_columns = []

        for column in dataframe.columns:

            lower = column.lower()

            if lower in {
                "time",
                "valid_time",
                "forecast_time",
            }:
                coordinate_columns.append(column)

            elif lower in {
                "latitude",
                "lat",
            }:
                coordinate_columns.append(column)

            elif lower in {
                "longitude",
                "lon",
            }:
                coordinate_columns.append(column)

        if len(coordinate_columns) >= 3:

            dataframe = dataframe.drop_duplicates(
                subset=coordinate_columns,
                keep="first",
            )

        rows_after = len(dataframe)

        duplicates_removed = rows_before - rows_after

        percentage = 0.0

        if rows_before > 0:
            percentage = (
                duplicates_removed / rows_before
            ) * 100

        cleaned_dataset = (
            dataframe
            .set_index(coordinate_columns)
            .to_xarray()
        )

        report.add(
            "duplicates",
            "analysis",
            {
                "rows_before": rows_before,
                "rows_after": rows_after,
                "duplicates_removed": duplicates_removed,
                "duplicate_percentage": round(
                    percentage,
                    4,
                ),
                "coordinate_columns": coordinate_columns,
            },
        )

        return cleaned_dataset