import numpy as np
import pandas as pd


class CorrelationAnalysis:
    """
    Performs correlation analysis between all numerical variables
    in the weather dataset.
    """

    def analyze(self, dataset):
        """
        Returns
        -------
        dict
            {
                "matrix": DataFrame,
                "strong_positive": list,
                "strong_negative": list
            }
        """

        dataframe = self._dataset_to_dataframe(dataset)

        if dataframe.empty:
            return {
                "matrix": pd.DataFrame(),
                "strong_positive": [],
                "strong_negative": []
            }

        correlation = dataframe.corr(method="pearson")

        positive = []
        negative = []

        columns = correlation.columns

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):

                value = correlation.iloc[i, j]

                if np.isnan(value):
                    continue

                pair = {
                    "variable_1": columns[i],
                    "variable_2": columns[j],
                    "correlation": round(float(value), 4)
                }

                if value >= 0.70:
                    positive.append(pair)

                elif value <= -0.70:
                    negative.append(pair)

        return {
            "method": "Pearson Correlation",
            "variables": list(correlation.columns),
            "matrix": correlation,
            "strong_positive": positive,
            "strong_negative": negative,
            "total_positive": len(positive),
            "total_negative": len(negative)
        }

    def export_csv(self, correlation_matrix, output_path):
        """
        Save correlation matrix as CSV.
        """

        correlation_matrix.to_csv(output_path)

    def _dataset_to_dataframe(self, dataset):
        """
        Convert xarray dataset into a flattened pandas DataFrame.
        """

        data = {}

        for variable in dataset.data_vars:

            values = np.asarray(dataset[variable].values)

            data[variable] = values.flatten()

        dataframe = pd.DataFrame(data)

        dataframe = dataframe.select_dtypes(include=np.number)

        return dataframe.dropna(how="all")