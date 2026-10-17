import xarray as xr

from src.preprocessing.summary import PreprocessingSummary
from src.features.seasonal_features import TemporalFeatureGenerator

ds = xr.open_dataset("data/processed/processed_dataset_20260607_213959.nc")

report = PreprocessingSummary()

generator = TemporalFeatureGenerator()

result = generator.process(ds, report)

print(result)

print("\nReport:")
print(report.to_dict())
