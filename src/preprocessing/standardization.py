from src.data.weather_variables import WeatherVariableNormalizer


class VariableStandardizer:
    """
    Standardize weather variables.

    Responsibilities
    ----------------
    1. Identify weather variables.
    2. Validate units.
    3. Build standardized mapping.
    4. Report missing weather concepts.

    NOTE:
    Dataset variables are NOT renamed.
    We only create a standardized mapping.
    """

    EXPECTED_UNITS = {
        "temperature": ["K", "C", "°C"],
        "pressure": ["Pa", "hPa", "mb"],
        "humidity": ["%", "kg/kg"],
        "precipitation": ["mm", "kg m-2"],
        "wind": ["m/s", "km/h", "knots", "m s**-1","m s^-1"],
    }

    def __init__(self):

        self.normalizer = WeatherVariableNormalizer() 

    def process(self, dataset, report):

        resolved = self.normalizer.resolve(dataset)

        report_data = {}

        for concept, info in resolved.items():

            report_data[concept] = {

                "available": info["available"],

                "dataset_variable": None,

                "units": None,

                "expected_units": self.EXPECTED_UNITS.get(concept, []),

                "unit_valid": False,

                "remarks": "",
            }

            if not info["available"]:

                report_data[concept]["remarks"] = "Variable not found"

                continue 

            if concept == "wind" and info["vector_available"]:

                u_var = dataset[info["u"]]
                v_var = dataset[info["v"]]

                units = u_var.attrs.get("units", "")

                report_data[concept].update({

                    "dataset_variable": [

                        info["u"],

                        info["v"]

                    ],

                    "units": units,

                    "unit_valid": self._unit_matches(
                        units,
                        self.EXPECTED_UNITS["wind"],
                    ),

                    "remarks": "Derived from vector components",

                })

            else:

                variable_name = info["scalar"]

                variable = dataset[variable_name]

                units = variable.attrs.get("units", "")

                report_data[concept].update({

                    "dataset_variable": variable_name,

                    "units": units,

                    "unit_valid": self._unit_matches(
                        units,
                        self.EXPECTED_UNITS.get(
                        concept,
                        [],
                        ),
                    ),

                    "remarks": "Detected",

                })

        report.add(
            "standardization",
            "analysis",
            report_data,
        )

        return dataset
    
    def _unit_matches(self, unit, expected):

        if not unit:
            return False

        normalized = (
            unit.lower()
            .replace(" ", "")
            .replace("**", "^")
            .replace("/s", "s^-1")
        )

        for e in expected:

            candidate = (
                e.lower()
                .replace(" ", "")
                .replace("**", "^")
                .replace("/s", "s^-1")
            )

            if normalized == candidate:
                return True

        return False