import numpy as np
import xarray as xr


class MeteorologicalFeatureGenerator:
    """
    Generate additional meteorological features from the
    standardized weather dataset.

    Expected standardized variables:
        temperature
        pressure
        humidity
        wind_speed
        u_wind
        v_wind
        precipitation
    """

    def process(self, dataset: xr.Dataset, report):

        featured = dataset.copy(deep=True)

        generated = []

        # --------------------------------------------------
        # Wind Direction
        # --------------------------------------------------

        if (
            "u_wind" in featured.data_vars and
            "v_wind" in featured.data_vars and
            "wind_direction" not in featured.data_vars
        ):

            featured["wind_direction"] = (
                np.degrees(
                    np.arctan2(
                        featured["v_wind"],
                        featured["u_wind"],
                    )
                ) + 360
            ) % 360

            featured["wind_direction"].attrs.update({
                "long_name": "Wind Direction",
                "units": "degrees",
            })

            generated.append("wind_direction")

        # --------------------------------------------------
        # Temperature (Kelvin → Celsius)
        # --------------------------------------------------

        if (
            "temperature" in featured.data_vars and
            "temperature_celsius" not in featured.data_vars
        ):

            units = featured["temperature"].attrs.get("units", "").lower()

            # Convert only if temperature is in Kelvin
            if units in {"k", "kelvin"}:

                featured["temperature_celsius"] = (
                    featured["temperature"] - 273.15
                )

                featured["temperature_celsius"].attrs.update({
                    "long_name": "Temperature (Celsius)",
                    "units": "°C",
                })

                generated.append("temperature_celsius")

        # --------------------------------------------------
        # Pressure (Pa → hPa)
        # --------------------------------------------------

        if (
            "pressure" in featured.data_vars and
            "pressure_hpa" not in featured.data_vars
        ):

            units = featured["pressure"].attrs.get("units", "").lower()

            # Convert only if pressure is still in Pascal
            if units in {"pa", "pascal", "pascals"}:

                featured["pressure_hpa"] = (
                    featured["pressure"] / 100.0
                )

                featured["pressure_hpa"].attrs.update({
                    "long_name": "Surface Pressure",
                    "units": "hPa",
                })

                generated.append("pressure_hpa")

        # --------------------------------------------------
        # Report
        # --------------------------------------------------

        report.add(
            "meteorological_features",
            "generated_features",
            generated,
        )

        return featured