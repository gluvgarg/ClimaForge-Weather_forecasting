import json
from pathlib import Path


class ModelComparisonReport:
    """Collects per-model evaluation metrics for side-by-side comparison."""

    def __init__(self):
        self.report = {}

    def add(self, section, key, value):
        self.report.setdefault(section, {})
        self.report[section][key] = value

    def add_result(self, model_name, metrics):
        self.add("models", model_name, metrics)

    def to_dict(self):
        return self.report

    def to_rows(self):
        rows = []
        for model_name, metrics in self.report.get("models", {}).items():
            row = {"model": model_name}
            row.update(metrics)
            rows.append(row)
        return rows

    def save_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self.report,
                f,
                indent=4,
                default=str,
            )
