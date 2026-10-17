from datetime import datetime
from pathlib import Path

from src.core.logger import get_logger
from src.data.loaders.grd_loader import GRDLoader
from src.data.loaders.grib_loader import GRIBLoader
from src.data.loaders.netcdf_loader import NetCDFLoader
from src.utils.constants import PROCESSED_DATA_DIR


class DatasetConverter:
    """Reusable conversion workflows that produce standardized NetCDF outputs."""

    def __init__(self):
        self.loaders = {
            ".nc": NetCDFLoader(),
            ".netcdf": NetCDFLoader(),
            ".cdf": NetCDFLoader(),
            ".grib": GRIBLoader(),
            ".grb": GRIBLoader(),
            ".grib2": GRIBLoader(),
            ".grb2": GRIBLoader(),
            ".grd": GRDLoader(),
        }
        self.logger = get_logger(__name__)

    def load_as_xarray(self, file_path):
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix not in self.loaders:
            raise ValueError(f"Unsupported conversion source: {suffix}")
        return self.loaders[suffix].load(path)

    def convert_to_netcdf(self, file_path, output_path=None):
        dataset = self.load_as_xarray(file_path)
        source = Path(file_path)
        if output_path is None:
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = PROCESSED_DATA_DIR / f"{source.stem}_standardized_{stamp}.nc"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset.attrs["standardized_from"] = str(source)
        dataset.to_netcdf(output_path)
        self.logger.info("Converted %s to NetCDF: %s", source, output_path)
        return output_path

    def save_processed_dataset(self, dataset, name="processed_dataset"):
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PROCESSED_DATA_DIR / f"{name}_{stamp}.nc"
        dataset.to_netcdf(output_path)
        self.logger.info("Saved processed dataset: %s", output_path)
        return output_path
