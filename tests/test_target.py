import xarray as xr

from src.preprocessing.summary import PreprocessingSummary
from src.features.target_generator import TargetGenerator

ds = xr.open_dataset("data/raw/RF25_ind2025_rfp25.nc")

report = PreprocessingSummary()

generator = TargetGenerator()

result = generator.process(ds, report)

print(result)

print("\nReport:")
print(report.to_dict())