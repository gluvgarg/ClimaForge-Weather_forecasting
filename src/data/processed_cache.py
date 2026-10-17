import hashlib
import inspect
import json
import re
from pathlib import Path

from src.features.feature_pipeline import FeaturePipeline
from src.features.seasonal_features import TemporalFeatureGenerator
from src.features.meteorological_features import MeteorologicalFeatureGenerator
from src.features.lag_features import LagFeatureGenerator
from src.features.rolling_features import RollingFeatureGenerator
from src.features.target_generator import TargetGenerator
from src.utils.constants import PROCESSED_DATA_DIR

REQUIRED_FILES = ("X_train.csv", "X_val.csv", "X_test.csv", "y_train.csv", "y_val.csv", "y_test.csv")

_PIPELINE_CLASSES = (
    FeaturePipeline,
    TemporalFeatureGenerator,
    MeteorologicalFeatureGenerator,
    LagFeatureGenerator,
    RollingFeatureGenerator,
    TargetGenerator,
)

_YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


class ProcessedDatasetCache:
    """Decides whether FeaturePipeline's output for a given dataset can be
    reused or must be regenerated, and where it should live on disk.

    Colocated with DatasetManager (its only caller), matching the existing
    PreprocessingExporter precedent - FeaturePipeline itself stays completely
    cache-unaware.
    """

    MANIFEST_NAME = "cache_manifest.json"

    def dataset_key(self, source_paths):
        source_paths = [Path(p) for p in source_paths]
        if len(source_paths) == 1:
            return source_paths[0].stem

        years = []
        for path in source_paths:
            match = _YEAR_PATTERN.search(path.stem)
            years.append(match.group(0) if match else None)

        path_hash = self._hash_text(
            "|".join(sorted(str(path.resolve()) for path in source_paths))
        )[:6]

        prefix = f"merged_{min(years)}_{max(years)}" if all(years) else "merged"
        return f"{prefix}_{path_hash}"

    def resolve_directory(self, dataset_key):
        return PROCESSED_DATA_DIR / dataset_key

    def signature(self, dataset_key, source_paths):
        parts = [dataset_key, self._pipeline_source_hash()]
        for path in sorted((Path(p) for p in source_paths), key=lambda p: str(p.resolve())):
            stat = path.stat()
            parts.append(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")
        return self._hash_text("|".join(parts))

    def is_valid(self, directory, signature):
        directory = Path(directory)
        manifest_path = directory / self.MANIFEST_NAME
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if manifest.get("signature") != signature:
            return False
        return all((directory / name).exists() for name in REQUIRED_FILES)

    def write_manifest(self, directory, signature, source_paths):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "signature": signature,
            "source_paths": [str(Path(path).resolve()) for path in source_paths],
        }
        (directory / self.MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _pipeline_source_hash(self):
        # Editing a stage's actual logic (e.g. LagFeatureGenerator.LAGS) changes
        # this hash automatically, so there is no manually-maintained version
        # constant to forget to bump. Degrades gracefully (skips) if source is
        # unavailable, e.g. a future frozen/PyInstaller build with no .py files
        # on disk - a known, accepted limitation, not solved here.
        chunks = []
        for pipeline_class in _PIPELINE_CLASSES:
            try:
                chunks.append(Path(inspect.getsourcefile(pipeline_class)).read_bytes())
            except (TypeError, OSError):
                continue
        return self._hash_bytes(b"".join(chunks))

    def _hash_text(self, text):
        return self._hash_bytes(text.encode("utf-8"))

    def _hash_bytes(self, data):
        return hashlib.sha256(data).hexdigest()
