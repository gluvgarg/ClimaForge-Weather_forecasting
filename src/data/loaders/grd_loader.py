from pathlib import Path

import numpy as np
import xarray as xr


class GRDLoader:
    """Load simple ASCII GRD grids into a standardized xarray Dataset."""

    def load(self, file_path):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        with path.open("r", encoding="utf-8", errors="ignore") as file:
            header = {}
            values = []
            for raw_line in file:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                key = parts[0].lower()
                if key in {"ncols", "nrows", "xllcorner", "yllcorner", "xllcenter", "yllcenter", "cellsize", "nodata_value"}:
                    header[key] = float(parts[1])
                else:
                    values.extend(float(item) for item in parts)

        required = {"ncols", "nrows", "cellsize"}
        if not required.issubset(header):
            raise ValueError("GRD parser expects ESRI ASCII grid keys: ncols, nrows, cellsize.")

        ncols = int(header["ncols"])
        nrows = int(header["nrows"])
        data = np.asarray(values, dtype=float).reshape(nrows, ncols)
        nodata = header.get("nodata_value")
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)

        x0 = header.get("xllcorner", header.get("xllcenter", 0.0))
        y0 = header.get("yllcorner", header.get("yllcenter", 0.0))
        cell = header["cellsize"]
        lon = x0 + np.arange(ncols) * cell
        lat = y0 + np.arange(nrows) * cell

        dataset = xr.Dataset(
            data_vars={
                "grd_value": (
                    ("lat", "lon"),
                    np.flipud(data),
                    {"long_name": "GRD grid value", "units": ""},
                )
            },
            coords={
                "lat": ("lat", lat[::-1], {"units": "degrees_north", "long_name": "latitude"}),
                "lon": ("lon", lon, {"units": "degrees_east", "long_name": "longitude"}),
            },
            attrs={"source_format": "GRD", "source_path": str(path)},
        )
        return dataset
