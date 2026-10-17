import json
from datetime import datetime

import pandas as pd

from src.core.logger import get_logger
from src.utils.constants import EXPORT_DIR, PLOT_EXPORT_DIR, PROCESSED_DATA_DIR


class ExportManager:
    """Export scientific datasets, metadata, telemetry, and derived artifacts."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def export_metadata(self, metadata, source_path=None):
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = EXPORT_DIR / f"metadata_{timestamp}.json"

        payload = {
            "source": str(source_path) if source_path else "",
            "metadata": metadata,
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, default=str)

        self.logger.info("Exported metadata: %s", output_path)
        return output_path

    def export_json(self, payload, stem="export"):
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = EXPORT_DIR / f"{stem}_{self._stamp()}.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, default=str)
        self.logger.info("Exported JSON: %s", output_path)
        return output_path

    def export_telemetry_csv(self, rows, stem="telemetry"):
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = EXPORT_DIR / f"{stem}_{self._stamp()}.csv"
        columns = ["timestamp", "latitude", "longitude", "value", "units", "anomaly"]
        pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
        self.logger.info("Exported telemetry CSV: %s", output_path)
        return output_path

    def export_comparison_csv(self, rows, stem="model_comparison"):
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = EXPORT_DIR / f"{stem}_{self._stamp()}.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False)
        self.logger.info("Exported model comparison CSV: %s", output_path)
        return output_path

    def export_dataset_netcdf(self, dataset, stem="processed_dataset"):
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PROCESSED_DATA_DIR / f"{stem}_{self._stamp()}.nc"
        dataset.to_netcdf(output_path)
        self.logger.info("Exported dataset NetCDF: %s", output_path)
        return output_path

    def export_plot_image(self, figure, stem="plot"):
        PLOT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PLOT_EXPORT_DIR / f"{stem}_{self._stamp()}.png"
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        self.logger.info("Exported plot image: %s", output_path)
        return output_path

    def _stamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")
