from pathlib import Path

import xarray as xr


class GRIBLoader:
    """Load GRIB/GRIB2 weather products through cfgrib into xarray."""

    def load(self, file_path):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        try:
            dataset = xr.open_dataset(path, engine="cfgrib")
        except ValueError as error:
            raise RuntimeError(
                "GRIB loading requires cfgrib and ECMWF ecCodes. "
                "Install cfgrib/eccodes or convert this file to NetCDF first."
            ) from error

        dataset.attrs.setdefault("source_format", "GRIB")
        dataset.attrs.setdefault("source_path", str(path))
        return dataset
