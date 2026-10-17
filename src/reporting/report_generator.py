from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from src.core.logger import get_logger
from src.utils.constants import REPORT_DIR


class ReportManager:
    """Generate HTML and PDF scientific reports from the active xarray dataset."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def generate_html_report(self, metadata, summary, variable_name, statistics, telemetry_rows):
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPORT_DIR / f"weather_report_{self._stamp()}.html"
        rows_html = "\n".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
            for row in telemetry_rows[:50]
        )
        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Weather Data Analysis Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 28px; color: #17212b; }}
    h1, h2 {{ color: #0b4778; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #c8d3df; padding: 6px 8px; font-size: 12px; }}
    th {{ background: #e7f1fb; }}
    code {{ background: #eef3f7; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Weather Data Analysis Report</h1>
  <p><b>Generated:</b> {datetime.now()}</p>
  <p><b>Dataset:</b> {summary.get("dataset_name", "")}</p>
  <p><b>Source:</b> {summary.get("file_path", "")}</p>
  <p><b>Active variable:</b> <code>{variable_name}</code></p>
  <h2>Dimensions</h2>
  <pre>{summary.get("dimensions", {})}</pre>
  <h2>Coordinate Ranges</h2>
  <pre>{summary.get("coordinates", {})}</pre>
  <h2>Statistics</h2>
  <pre>{statistics}</pre>
  <h2>Validation</h2>
  <pre>{summary.get("validation", {})}</pre>
  <h2>Telemetry Sample</h2>
  <table>
    <tr><th>Timestamp</th><th>Latitude</th><th>Longitude</th><th>Value</th><th>Units</th><th>Anomaly</th></tr>
    {rows_html}
  </table>
  <h2>Metadata Keys</h2>
  <pre>{list(metadata.keys())}</pre>
</body>
</html>"""
        output_path.write_text(html, encoding="utf-8")
        self.logger.info("Generated HTML report: %s", output_path)
        return output_path

    def generate_pdf_report(self, dataset, summary, variable_name, statistics):
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPORT_DIR / f"weather_report_{self._stamp()}.pdf"
        with PdfPages(output_path) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.08, 0.94, "Weather Data Analysis Report", fontsize=18, weight="bold")
            fig.text(0.08, 0.89, f"Dataset: {summary.get('dataset_name', '')}", fontsize=10)
            fig.text(0.08, 0.86, f"Variable: {variable_name}", fontsize=10)
            fig.text(0.08, 0.82, f"Dimensions: {summary.get('dimensions', {})}", fontsize=9)
            fig.text(0.08, 0.78, f"Time: {summary.get('time_range', {})}", fontsize=9)
            fig.text(0.08, 0.72, f"Statistics: {statistics}", fontsize=9, wrap=True)
            pdf.savefig(fig)
            plt.close(fig)

            if variable_name and variable_name in dataset:
                values = np.asarray(dataset[variable_name].values, dtype=float)
                fig, axis = plt.subplots(figsize=(8.27, 5.5))
                axis.hist(values[~np.isnan(values)].ravel(), bins=30, color="#2b7bbb", alpha=0.85)
                axis.set_title(f"Distribution: {variable_name}")
                axis.set_xlabel(dataset[variable_name].attrs.get("units", "value"))
                axis.set_ylabel("count")
                pdf.savefig(fig)
                plt.close(fig)

        self.logger.info("Generated PDF report: %s", output_path)
        return output_path

    def generate_model_comparison_report(self, comparison_rows, metadata=None):
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPORT_DIR / f"model_comparison_{self._stamp()}.html"
        metadata = metadata or {}
        header_html = "".join(
            f"<th>{key}</th>" for key in (comparison_rows[0].keys() if comparison_rows else [])
        )
        rows_html = "\n".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row.values()) + "</tr>"
            for row in comparison_rows
        )
        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Model Comparison Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 28px; color: #17212b; }}
    h1, h2 {{ color: #0b4778; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #c8d3df; padding: 6px 8px; font-size: 12px; }}
    th {{ background: #e7f1fb; }}
    pre {{ background: #eef3f7; padding: 8px; }}
  </style>
</head>
<body>
  <h1>Model Comparison Report</h1>
  <p><b>Generated:</b> {datetime.now()}</p>
  <h2>Metrics</h2>
  <table>
    <tr>{header_html}</tr>
    {rows_html}
  </table>
  <h2>Metadata</h2>
  <pre>{metadata}</pre>
</body>
</html>"""
        output_path.write_text(html, encoding="utf-8")
        self.logger.info("Generated model comparison report: %s", output_path)
        return output_path

    def _stamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")
