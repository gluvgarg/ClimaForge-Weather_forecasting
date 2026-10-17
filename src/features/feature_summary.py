import json
from pathlib import Path


class FeatureSummary:

    def __init__(self):
        self.report = {}

    def add(self, section, key, value):
        self.report.setdefault(section, {})
        self.report[section][key] = value

    def to_dict(self):
        return self.report

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