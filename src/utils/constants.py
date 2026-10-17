from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPORT_DIR = BASE_DIR / "reports" / "statistics"
REPORT_DIR = BASE_DIR / "reports" / "forecast_reports"
PLOT_EXPORT_DIR = BASE_DIR / "reports" / "plots"
CARTOPY_CACHE_DIR = DATA_DIR / "external" / "cartopy_cache"
LOG_DIR = BASE_DIR / "experiments" / "experiment_logs"

MODEL_DIR = BASE_DIR / "models"
MODEL_CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
MODEL_SCALER_DIR = MODEL_DIR / "scalers"
MODEL_TRAINED_DIR = MODEL_DIR / "trained"

SUPPORTED_NETCDF_EXTENSIONS = {".nc", ".netcdf", ".cdf"}
SUPPORTED_GRIB_EXTENSIONS = {".grib", ".grb", ".grib2", ".grb2"}
SUPPORTED_GRD_EXTENSIONS = {".grd"}
SUPPORTED_DATASET_EXTENSIONS = (
    SUPPORTED_NETCDF_EXTENSIONS
    | SUPPORTED_GRIB_EXTENSIONS
    | SUPPORTED_GRD_EXTENSIONS
)

DEFAULT_PROJECTION = "Auto"
SUPPORTED_PROJECTIONS = ["Auto", "PlateCarree", "Mercator", "Orthographic"]
DEFAULT_RENDER_MAX_CELLS = 250000
