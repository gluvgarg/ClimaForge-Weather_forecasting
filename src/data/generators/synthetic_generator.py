from pathlib import Path

import numpy as np
import xarray as xr


def build_sample_dataset():
    rng = np.random.default_rng(17)
    time = np.array(
        ["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04"],
        dtype="datetime64[D]",
    )
    lat = np.linspace(8.0, 37.0, 24)
    lon = np.linspace(68.0, 97.0, 32)

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    seasonal = np.arange(len(time))[:, None, None] * 0.8

    # Each variable follows the same scientific data layout:
    # time x latitude x longitude. This mirrors common NetCDF weather products.
    temperature = 296 + seasonal + 3 * np.sin(lat_grid / 8)[None, :, :] + rng.normal(0, 0.7, (4, 24, 32))
    pressure = 1008 - 0.08 * lat_grid[None, :, :] + rng.normal(0, 0.8, (4, 24, 32))
    humidity = 62 + 15 * np.cos(lon_grid / 9)[None, :, :] + rng.normal(0, 3.0, (4, 24, 32))
    u_wind = 2.5 + 1.1 * np.cos(lat_grid / 7)[None, :, :] + rng.normal(0, 0.25, (4, 24, 32))
    v_wind = 1.6 + 0.9 * np.sin(lon_grid / 8)[None, :, :] + rng.normal(0, 0.25, (4, 24, 32))
    wind_speed = np.sqrt(u_wind**2 + v_wind**2)
    precipitation = np.maximum(0, rng.gamma(1.6, 2.0, (4, 24, 32)) - 1.2)

    return xr.Dataset(
        data_vars={
            "temperature": (
                ("time", "lat", "lon"),
                temperature,
                {"units": "K", "long_name": "Air temperature"},
            ),
            "pressure": (
                ("time", "lat", "lon"),
                pressure,
                {"units": "hPa", "long_name": "Surface pressure"},
            ),
            "humidity": (
                ("time", "lat", "lon"),
                humidity,
                {"units": "%", "long_name": "Relative humidity"},
            ),
            "wind_speed": (
                ("time", "lat", "lon"),
                wind_speed,
                {"units": "m/s", "long_name": "Wind speed"},
            ),
            "u_wind": (
                ("time", "lat", "lon"),
                u_wind,
                {"units": "m/s", "long_name": "U wind component", "standard_name": "eastward_wind"},
            ),
            "v_wind": (
                ("time", "lat", "lon"),
                v_wind,
                {"units": "m/s", "long_name": "V wind component", "standard_name": "northward_wind"},
            ),
            "precipitation": (
                ("time", "lat", "lon"),
                precipitation,
                {"units": "mm", "long_name": "Precipitation"},
            ),
        },
        coords={
            "time": ("time", time, {"long_name": "valid time"}),
            "lat": ("lat", lat, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", lon, {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs={
            "title": "Sample multi-variable weather dataset",
            "source": "Generated locally for GUI and model testing",
        },
    )


def main():
    output_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "sample_weather.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_sample_dataset().to_netcdf(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
