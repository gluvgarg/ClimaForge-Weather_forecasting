import numpy as np
import pandas as pd


class SequenceWindowGenerator:
    """Build multivariate sequence windows from a tabular feature dataframe.

    Rows are grouped by grid cell (lat/lon) and ordered chronologically so each
    sequence spans a single physical location, then windowed into
    (samples, sequence_length, features) tensors for sequence models. The
    target used for each window is the value aligned to the window's last row -
    correct because TargetGenerator already forward-shifts the rainfall target.
    """

    TIME_NAMES = {"time", "valid_time", "forecast_time"}
    LAT_NAMES = {"lat", "latitude"}
    LON_NAMES = {"lon", "longitude"}

    def make_sequences(self, dataframe, target, sequence_length=6, feature_columns=None, max_sequences=None, seed=42):
        time_column = self._find_column(dataframe, self.TIME_NAMES)
        lat_column = self._find_column(dataframe, self.LAT_NAMES)
        lon_column = self._find_column(dataframe, self.LON_NAMES)

        if feature_columns is None:
            exclude = {target, time_column, lat_column, lon_column}
            feature_columns = [
                column
                for column in dataframe.columns
                if column not in exclude and pd.api.types.is_numeric_dtype(dataframe[column])
            ]

        group_keys = [column for column in (lat_column, lon_column) if column]
        sort_keys = group_keys + ([time_column] if time_column else [])
        ordered = dataframe.sort_values(sort_keys) if sort_keys else dataframe

        if group_keys:
            grouped = ordered.groupby(group_keys)
            keys = list(grouped.groups.keys())
            if max_sequences is not None:
                # Visit grid cells in random order so stopping early doesn't
                # bias the sample toward one spatial corner of the domain.
                # Cells are only materialized on demand (get_group) instead
                # of building every group up front, so an early stop actually
                # skips the work for cells never visited - for a multi-year
                # merged grid, building every possible sequence before
                # capping has been measured to take 10+ minutes and several
                # GB of RAM for no benefit once the result is subsampled anyway.
                np.random.RandomState(seed).shuffle(keys)
            groups = ((key, grouped.get_group(key)) for key in keys)
        else:
            groups = [(None, ordered)]

        sequences, targets = [], []
        for _, group in groups:
            feature_values = group[feature_columns].to_numpy(dtype=float)
            target_values = group[target].to_numpy(dtype=float)
            for end in range(sequence_length, len(group) + 1):
                start = end - sequence_length
                window_target = target_values[end - 1]
                if np.isnan(window_target) or np.isnan(feature_values[start:end]).any():
                    continue
                sequences.append(feature_values[start:end])
                targets.append(window_target)
                if max_sequences is not None and len(sequences) >= max_sequences:
                    return (
                        np.asarray(sequences, dtype=float),
                        np.asarray(targets, dtype=float),
                        feature_columns,
                    )

        if not sequences:
            return (
                np.empty((0, sequence_length, len(feature_columns))),
                np.empty((0,)),
                feature_columns,
            )
        return np.asarray(sequences, dtype=float), np.asarray(targets, dtype=float), feature_columns

    def _find_column(self, dataframe, names):
        lowered = {name.lower() for name in names}
        for column in dataframe.columns:
            if column.lower() in lowered:
                return column
        return None
