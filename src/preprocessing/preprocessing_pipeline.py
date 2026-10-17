from src.core.logger import get_logger

from src.preprocessing.summary import PreprocessingSummary
from src.preprocessing.missing_value_handler import MissingValueHandler
from src.preprocessing.timestamp_validator import TimestampValidator
from src.preprocessing.duplicate_handler import DuplicateHandler
from src.preprocessing.standardization import VariableStandardizer
from src.preprocessing.temporal_alignment import SpatialTemporalAligner
from src.preprocessing.quality_validator import QualityValidator
from src.preprocessing.outlier_detector import OutlierHandler


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline.
    Every stage receives:
        Dataset
        ↓
        process(dataset, report)
        ↓
        Dataset
    The report is updated throughout the pipeline.
    """

    def __init__(self):

        self.logger = get_logger(__name__)
        self.stages = [
            ("Missing Values", MissingValueHandler()),
            ("Timestamp Validation", TimestampValidator()),
            ("Duplicate Removal", DuplicateHandler()),
            ("Variable Standardization", VariableStandardizer()),
            ("Spatial/Temporal Alignment", SpatialTemporalAligner()),
            ("Quality Validation", QualityValidator()),
            ("Outlier Detection", OutlierHandler()),
        ]

    def run(self, dataset):
        report = PreprocessingSummary()
        cleaned = dataset.copy(deep=True)
        self.logger.info("Starting preprocessing pipeline...")

        for stage_name, stage in self.stages:
            try:
                self.logger.info("Running stage: %s", stage_name)
                cleaned = stage.process(cleaned, report)

            except Exception as e:
                self.logger.exception(
                    "Stage '%s' failed.", stage_name
                )
                report.add(
                    "pipeline_errors",
                    stage_name,
                    str(e)
                )

        self.logger.info("Preprocessing pipeline completed.")

        return cleaned, report