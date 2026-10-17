from dataclasses import dataclass


@dataclass
class ForecastViewModel:
    variable: str
    units: str
    model_name: str
    time_labels: list
    series: list
    test_index: list
    actual: list
    predicted: list
    future_index: list
    future: list
    metrics: dict
    region_bounds: dict
