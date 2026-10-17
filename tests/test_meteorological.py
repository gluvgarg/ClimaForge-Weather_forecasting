import xarray as xr

from src.preprocessing.summary import PreprocessingSummary
from src.features.meteorological_features import MeteorologicalFeatureGenerator

ds = xr.open_dataset("data/processed/processed_dataset_20260607_213959.nc")

report = PreprocessingSummary()

generator = MeteorologicalFeatureGenerator()

result = generator.process(ds, report)

print(result)

print("\nReport:")
print(report.to_dict())