from pathlib import Path

import xarray as xr


class NetCDFLoader:
    """Load NetCDF datasets into xarray's unified Dataset structure."""

    def load(self, file_path):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        dataset = xr.open_dataset(path, decode_times=True)
        dataset.attrs.setdefault("source_format", "NetCDF")
        dataset.attrs.setdefault("source_path", str(path))
        return dataset
