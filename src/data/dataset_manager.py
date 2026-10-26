from pathlib import Path 
import numpy as np

from src.core.logger import get_logger
from src.data.loaders.grd_loader import GRDLoader
from src.data.loaders.grib_loader import GRIBLoader
from src.data.loaders.netcdf_loader import NetCDFLoader
from src.data.validation import DatasetValidator
from src.data.exporter import PreprocessingExporter
from src.data.dataset_merger import DatasetMerger
from src.data.processed_cache import ProcessedDatasetCache
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.analysis.analysis_pipeline import AnalysisPipeline
from src.features.feature_pipeline import FeaturePipeline
from src.data.weather_variables import WeatherVariableNormalizer
from src.utils.constants import (
    SUPPORTED_DATASET_EXTENSIONS,
    SUPPORTED_GRD_EXTENSIONS,
    SUPPORTED_GRIB_EXTENSIONS,
    SUPPORTED_NETCDF_EXTENSIONS,
)


class DatasetManager:
    """Owns the currently loaded dataset and hides format-specific loading."""

    def __init__(self):
        self.dataset = None
        self.file_path = None
        self.validation_report = {}
        self.preprocessing_report = {}
        self.analysis_report = {}
        self.feature_report = {}
        self.processed_directory = None
        self.processed_dataset_key = None
        self.source_paths = []
        self.loaders = {
            "netcdf": NetCDFLoader(),
            "grib": GRIBLoader(),
            "grd": GRDLoader(),
        }
        self.validator = DatasetValidator()
        self.normalizer = WeatherVariableNormalizer()
        self.exporter = PreprocessingExporter()
        self.preprocessing_pipeline = PreprocessingPipeline()
        self.analysis_pipeline = AnalysisPipeline()
        self.feature_pipeline = FeaturePipeline()
        self.dataset_merger = DatasetMerger()
        self.processed_cache = ProcessedDatasetCache()
        self.logger = get_logger(__name__)

    def load_dataset(self, file_path, progress_callback=None):
        progress = progress_callback or (lambda message: None)
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix not in SUPPORTED_DATASET_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{suffix}'. Supported formats: NetCDF, GRIB, GRD."
            )

        progress(f"Loading {path.name}...")
        loader = self._loader_for_suffix(suffix)
        dataset = loader.load(path)

        dataset = self._process_loaded_dataset(
            dataset,
            dataset_key=path.stem,
            source_paths=[path],
            progress_callback=progress,
        )

        self.file_path = path

        self.logger.info(
            "Loaded %s dataset: %s",
            self.dataset.attrs.get("source_format", suffix),
            path,
        )

        return dataset

    def load_datasets(self, file_paths, progress_callback=None):
        """Load, merge, and process multiple yearly datasets as one dataset.

        Preprocessing, Analysis, and Feature Engineering all then run exactly
        as they do for a single-file dataset - the merge is the only thing
        that differs about this workflow.
        """
        progress = progress_callback or (lambda message: None)
        paths = [Path(file_path) for file_path in file_paths]
        if not paths:
            raise ValueError("No dataset files provided.")

        loaded = []
        for path in paths:
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_DATASET_EXTENSIONS:
                raise ValueError(
                    f"Unsupported format '{suffix}'. Supported formats: NetCDF, GRIB, GRD."
                )
            progress(f"Loading {path.name}...")
            loader = self._loader_for_suffix(suffix)
            loaded.append(loader.load(path))

        progress(f"Merging {len(paths)} datasets...")
        dataset = self.dataset_merger.merge(loaded, paths)
        dataset_key = self.processed_cache.dataset_key(paths)

        dataset = self._process_loaded_dataset(
            dataset,
            dataset_key=dataset_key,
            source_paths=paths,
            progress_callback=progress,
        )

        self.file_path = paths[0]

        self.logger.info(
            "Loaded merged dataset from %d files: %s",
            len(paths),
            dataset_key,
        )

        return dataset

    def _process_loaded_dataset(self, dataset, dataset_key, source_paths, progress_callback=None):
        progress = progress_callback or (lambda message: None)

        self.validation_report = self.validator.validate(dataset)
        if not self.validation_report["valid"]:
            raise ValueError(
                "Dataset validation failed:\n"
                + "\n".join(self.validation_report["errors"])
            )
        # -------------------------------
        # Preprocessing
        # -------------------------------

        progress("Preprocessing...")
        dataset, preprocessing_report = (
            self.preprocessing_pipeline.run(dataset)
        )

        self.preprocessing_report = (
            preprocessing_report.to_dict()
        )

        # -------------------------------
        # Analysis
        # -------------------------------

        progress("Analyzing...")
        analysis_report = (
            self.analysis_pipeline.run(dataset)
        )

        self.analysis_report = analysis_report
        # -------------------------------
        # Export Preprocessing Outputs
        # -------------------------------

        self.exporter.export(
            dataset=dataset,
            report=preprocessing_report,
            dataset_name=dataset_key,
            export_csv=False,
        )

        # -------------------------------
        # Feature Engineering Setup (On-demand)
        # -------------------------------

        processed_directory = self.processed_cache.resolve_directory(dataset_key)
        self.processed_directory = processed_directory
        self.processed_dataset_key = dataset_key
        self.source_paths = source_paths
        self.dataset = dataset

        signature = self.processed_cache.signature(dataset_key, source_paths)
        if self.processed_cache.is_valid(processed_directory, signature):
            progress("Valid cached features available for forecasting.")
            self.logger.info("Cached features available: %s", processed_directory)
        else:
            self.logger.info("Features will be generated on-demand when required by forecasting.")

        return dataset

    def ensure_features(self, progress_callback=None):
        """Ensure feature-engineered tabular files exist in processed_directory,
        either by reusing valid cached files or running FeaturePipeline on demand.
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded.")
        progress = progress_callback or (lambda message: None)
        processed_directory = self.processed_cache.resolve_directory(self.processed_dataset_key)
        signature = self.processed_cache.signature(self.processed_dataset_key, self.source_paths)

        if self.processed_cache.is_valid(processed_directory, signature):
            progress("Reusing cached feature-engineered outputs...")
            self.logger.info("Reusing cached feature-engineered outputs: %s", processed_directory)
            self.processed_directory = processed_directory
            return processed_directory

        progress("Engineering features...")
        self.logger.info("Running feature engineering: %s", processed_directory)
        _, feature_report = self.feature_pipeline.run(
            self.dataset,
            output_dir=processed_directory,
        )
        self.feature_report = feature_report.to_dict()
        self.processed_cache.write_manifest(processed_directory, signature, self.source_paths)
        self.processed_directory = processed_directory
        return processed_directory

    def close_dataset(self):
        """Release the active xarray dataset and reset file state."""
        if self.dataset is not None:
            self.dataset.close()
        self.dataset = None
        self.file_path = None
        self.validation_report = {}
        self.preprocessing_report = {}
        self.analysis_report = {}
        self.feature_report = {}
        self.processed_directory = None
        self.processed_dataset_key = None
        self.source_paths = []

    def get_dataset(self):
        return self.dataset

    def has_dataset(self):
        return self.dataset is not None

    def get_variable_names(self):
        if self.dataset is None:
            return []
        names = list(self.dataset.data_vars)
        wind = self.normalizer.resolve(self.dataset)["wind"]
        if wind.get("vector_available") and not wind.get("scalar") and "wind_speed" not in names:
            names.insert(0, "wind_speed")
        return names

    def has_variable(self, variable_name):
        if self.dataset is None:
            return False
        if variable_name in self.dataset.data_vars:
            return True
        return variable_name == "wind_speed" and self._derived_wind_speed() is not None

    def get_variable(self, variable_name):
        if not self.has_variable(variable_name):
            raise KeyError(f"Variable not found in dataset: {variable_name}")
        if variable_name not in self.dataset.data_vars and variable_name == "wind_speed":
            return self._derived_wind_speed()
        return self.dataset[variable_name]

    def get_dimensions(self):
        if self.dataset is None:
            return {}
        return {name: int(size) for name, size in self.dataset.sizes.items()}

    def get_coordinates(self):
        if self.dataset is None:
            return []
        return list(self.dataset.coords)

    def get_coordinate_ranges(self):
        if self.dataset is None:
            return {}

        ranges = {}
        for name, coord in self.dataset.coords.items():
            values = coord.values
            if values.size == 0:
                ranges[name] = {"min": "", "max": ""}
                continue
            try:
                ranges[name] = {
                    "min": str(values.min()),
                    "max": str(values.max()),
                }
            except TypeError:
                ranges[name] = {
                    "min": str(values[0]),
                    "max": str(values[-1]),
                }
        return ranges

    def get_variable_profile(self, variable_name):
        variable = self.get_variable(variable_name)
        values = self._numeric_values(variable)
        missing = ""
        valid = ""
        if values is not None:
            missing = int(np.count_nonzero(np.isnan(values)))
            valid = int(np.count_nonzero(~np.isnan(values)))
        return {
            "name": variable_name,
            "dims": list(variable.dims),
            "shape": [int(size) for size in variable.shape],
            "dtype": str(variable.dtype),
            "units": variable.attrs.get("units", ""),
            "long_name": variable.attrs.get("long_name", variable_name),
            "ndim": int(variable.ndim),
            "attrs": dict(variable.attrs),
            "valid_count": valid,
            "missing_values": missing,
        }

    def get_dataset_info(self):
        if self.dataset is None:
            return {}

        file_size = self.file_path.stat().st_size if self.file_path and self.file_path.exists() else 0
        variables = {
            name: self.get_variable_profile(name)
            for name in self.get_variable_names()
        }
        time_range = self._time_range()

        return {
            "dataset_name": self.dataset.attrs.get("title", self.file_path.name if self.file_path else "Loaded dataset"),
            "file_path": str(self.file_path) if self.file_path else "",
            "file_size_bytes": int(file_size),
            "file_size": self._format_file_size(file_size),
            "dimensions": self.get_dimensions(),
            "coordinates": self.get_coordinate_ranges(),
            "coordinate_names": self.get_coordinates(),
            "time_range": time_range,
            "variable_count": len(variables),
            "variables": variables,
            "attributes": dict(self.dataset.attrs),
            "validation": self.validation_report,
            "preprocessing": self.preprocessing_report,
            "analysis": self.analysis_report,
            "pipeline_errors": self.preprocessing_report.get("pipeline_errors", {}),
        }

    def get_validation_report(self):
        return dict(self.validation_report)

    def get_time_dimension_size(self, variable_name):
        if not self.has_variable(variable_name):
            return 0

        variable = self.get_variable(variable_name)
        for dim_name in variable.dims:
            if dim_name.lower() in {"time", "valid_time", "forecast_time"}:
                return int(variable.sizes[dim_name])
        return int(variable.shape[0]) if variable.ndim == 3 else 0

    def _format_file_size(self, size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes / (1024 * 1024):.1f} MB"

    def _time_range(self):
        if self.dataset is None:
            return {}
        for coord_name, coord in self.dataset.coords.items():
            if coord_name.lower() in {"time", "valid_time", "forecast_time"} and coord.size:
                values = coord.values
                return {
                    "name": coord_name,
                    "start": str(values[0]),
                    "end": str(values[-1]),
                    "count": int(coord.size),
                }
        return {}

    def _numeric_values(self, data_array):
        try:
            return np.asarray(data_array.values, dtype=float)
        except (TypeError, ValueError):
            return None

    def _derived_wind_speed(self):
        if self.dataset is None:
            return None
        wind = self.normalizer.resolve(self.dataset)["wind"]
        if not wind.get("vector_available"):
            return None
        u = self.dataset[wind["u"]]
        v = self.dataset[wind["v"]]
        speed = np.sqrt(u**2 + v**2)
        speed.name = "wind_speed"
        speed.attrs["long_name"] = "Derived wind speed"
        speed.attrs["units"] = u.attrs.get("units", "m/s")
        speed.attrs["derived_from"] = f"{wind['u']}, {wind['v']}"
        return speed

    def _loader_for_suffix(self, suffix):
        if suffix in SUPPORTED_NETCDF_EXTENSIONS:
            return self.loaders["netcdf"]
        if suffix in SUPPORTED_GRIB_EXTENSIONS:
            return self.loaders["grib"]
        if suffix in SUPPORTED_GRD_EXTENSIONS:
            return self.loaders["grd"]
        raise ValueError(f"Unsupported dataset suffix: {suffix}")
    
    def get_preprocessing_report(self):
        return dict(self.preprocessing_report)
    
    def get_analysis_report(self):
        return dict(self.analysis_report)
     
