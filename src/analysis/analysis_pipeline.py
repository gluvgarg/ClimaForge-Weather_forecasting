from src.analysis.metadata import MetadataExtractor
from src.analysis.descriptive_statistics import AnalysisEngine
from src.analysis.summary import AnalysisSummary
from src.analysis.analysis_exporter import AnalysisExporter
from src.analysis.metadata import MetadataExtractor
from src.analysis.descriptive_statistics import AnalysisEngine
from src.analysis.correlation_analysis import CorrelationAnalysis
from src.analysis.trend_analysis import TrendAnalysis
from src.analysis.seasonality_analysis import SeasonalityAnalysis
from src.analysis.extreme_weather_analysis import ExtremeWeatherAnalysis
from src.analysis.anomaly_detection import AnomalyDetection
from src.analysis.summary import AnalysisSummary
from src.analysis.analysis_exporter import AnalysisExporter

class AnalysisPipeline:
    """
    Complete Analysis Pipeline.

    Responsibilities
    ----------------
    1. Extract metadata.
    2. Compute statistics.
    3. Generate summary.
    4. Export results.
    """

    def __init__(self):

        self.metadata = MetadataExtractor()
        self.statistics = AnalysisEngine()

        self.correlation = CorrelationAnalysis()
        self.trend = TrendAnalysis()
        self.seasonality = SeasonalityAnalysis()
        self.extreme = ExtremeWeatherAnalysis()
        self.anomaly = AnomalyDetection()

        self.summary = AnalysisSummary()
        self.exporter = AnalysisExporter()

    def run(self, dataset):

        report = {}

        # ---------------------------
        # Metadata
        # ---------------------------

        report["metadata"] = self.metadata.extract(dataset)

        # ---------------------------
        # Descriptive Statistics
        # ---------------------------

        statistics = {}

        for variable_name, variable in dataset.data_vars.items():

            statistics[variable_name] = {

                "summary":
                    self.statistics.summarize(variable),

                "range":
                    self.statistics.variable_range(variable),

                "distribution":
                    self.statistics.distribution(variable),

                "rolling_average":
                    self.statistics.rolling_average(variable),

                "temporal_mean":
                    self.statistics.temporal_spatial_mean(variable)
            }

        report["statistics"] = statistics

        # ---------------------------
        # Correlation
        # ---------------------------

        report["correlation"] = self.correlation.analyze(dataset)

        # ---------------------------
        # Trend
        # ---------------------------

        report["trend"] = self.trend.analyze(dataset)

        # ---------------------------
        # Seasonality
        # ---------------------------

        report["seasonality"] = self.seasonality.analyze(dataset)

        # ---------------------------
        # Extreme Weather
        # ---------------------------

        report["extreme_weather"] = self.extreme.analyze(dataset)

        # ---------------------------
        # Anomaly Detection
        # ---------------------------

        report["anomalies"] = self.anomaly.analyze(dataset)

        # ---------------------------
        # Overall Summary
        # ---------------------------

        report["summary"] = self.summary.generate(report)

        # ---------------------------
        # Export
        # ---------------------------

        self.exporter.export_summary(report)
        self.exporter.export_statistics_csv(report)

        return report