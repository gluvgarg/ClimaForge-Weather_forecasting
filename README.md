# Weather Forecasting System

Desktop weather analytics and forecasting system built with Python, PyQt5, xarray, Matplotlib, and Cartopy.

## Current Structure

```text
config/
data/
  raw/
  interim/
  processed/
  synthetic/
  external/
experiments/
models/
reports/
src/
  core/
  controller/
  data/
  analysis/
  forecasting/
  evaluation/
  visualization/
  reporting/
  utils/
tests/
main.py
requirements.txt
setup.py
```

## Run

```powershell
.\venv\Scripts\python.exe main.py
```

## What Is Implemented

- PyQt5 desktop dashboard
- xarray-based dataset workflow
- NetCDF, GRIB, GRD, and CSV loader modules
- dataset validation and metadata extraction
- Cartopy GIS weather visualization
- weather layer controls and region selection
- telemetry grid and live metrics
- basic statistical analysis
- Phase 1 univariate forecasting with moving average and linear regression
- report and export managers

## Notes

The directory layout now follows the requested `src/` project structure. Only implemented project work was moved into the new layout; advanced model families listed in the reference structure are represented as directories but not filled with fake placeholder model code.
