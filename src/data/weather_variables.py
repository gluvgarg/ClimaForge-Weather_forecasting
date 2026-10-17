class WeatherVariableNormalizer:
    """Map heterogeneous dataset names into standard atmospheric concepts."""

    CONCEPTS = {
        "wind": {
            "label": "Wind Speed",
            "button": "WND",
            "scalar_aliases": {"wind_speed", "windspeed", "wind", "ws", "wind_velocity"},
            "u_aliases": {"u10", "u_wind", "u_component_of_wind", "eastward_wind", "uwnd"},
            "v_aliases": {"v10", "v_wind", "v_component_of_wind", "northward_wind", "vwnd"},
        },
        "temperature": {
            "label": "Temperature",
            "button": "TMP",
            "scalar_aliases": {"temperature", "temp", "t2m", "air_temperature", "air_temp"},
        },
        "pressure": {
            "label": "Pressure",
            "button": "PRS",
            "scalar_aliases": {"pressure", "surface_pressure", "msl", "sp", "barometric_pressure"},
        },
        "humidity": {
            "label": "Humidity",
            "button": "HUM",
            "scalar_aliases": {"humidity", "relative_humidity", "rh", "q", "specific_humidity"},
        },
        "precipitation": {
            "label": "Precipitation",
            "button": "PCP",
            "scalar_aliases": {"precipitation", "precip", "rain", "rainfall", "tp", "total_precipitation"},
        },
    }

    PRIMARY_ORDER = ["wind", "temperature", "pressure", "humidity", "precipitation"]

    def resolve(self, dataset):
        resolved = {}
        for concept, config in self.CONCEPTS.items():
            scalar = self._find_by_aliases(dataset, config.get("scalar_aliases", set()))
            item = {
                "concept": concept,
                "label": config["label"],
                "button": config["button"],
                "scalar": scalar,
                "available": scalar is not None,
            }
            if concept == "wind":
                item["u"] = self._find_by_aliases(dataset, config["u_aliases"])
                item["v"] = self._find_by_aliases(dataset, config["v_aliases"])
                item["vector_available"] = item["u"] is not None and item["v"] is not None
                item["available"] = item["available"] or item["vector_available"]
            resolved[concept] = item
        return resolved

    def concept_for_variable(self, variable_name, data_array):
        candidates = self._candidates(variable_name, data_array)
        for concept, config in self.CONCEPTS.items():
            aliases = set(config.get("scalar_aliases", set()))
            aliases |= set(config.get("u_aliases", set()))
            aliases |= set(config.get("v_aliases", set()))
            if candidates & aliases:
                return concept
            for alias in aliases:
                if len(alias) >= 4 and any(alias in candidate for candidate in candidates):
                    return concept
        return "generic"

    def _find_by_aliases(self, dataset, aliases):
        for variable_name, variable in dataset.data_vars.items():
            if self._candidates(variable_name, variable) & aliases:
                return variable_name
        return None

    def _candidates(self, variable_name, variable):
        return {
            variable_name.lower(),
            variable.attrs.get("standard_name", "").lower(),
            variable.attrs.get("long_name", "").lower().replace(" ", "_"),
        }
