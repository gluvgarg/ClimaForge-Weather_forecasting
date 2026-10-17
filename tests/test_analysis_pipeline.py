import json

from pathlib import Path
import sys

from src.data.loaders.netcdf_loader import NetCDFLoader
from src.analysis.analysis_pipeline import AnalysisPipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))


loader = NetCDFLoader()

dataset = loader.load(
    ROOT / "data/raw/sample_weather.nc"
)

pipeline = AnalysisPipeline()

report = pipeline.run(dataset)

print(json.dumps(report["summary"], indent=4))

# from src.analysis.correlation_analysis import CorrelationAnalysis

# correlation = CorrelationAnalysis()

# result = correlation.compute(dataset)

# print(result["matrix"])

# print("\nStrong Positive Correlations:")
# print(result["strong_positive"])

# print("\nStrong Negative Correlations:")
# print(result["strong_negative"])

# from src.analysis.trend_analysis import TrendAnalysis

# trend = TrendAnalysis()

# result = trend.analyze(dataset)

# import pprint
# pprint.pp(result)


# from src.analysis.seasonality_analysis import SeasonalityAnalysis

# season = SeasonalityAnalysis()

# result = season.analyze(dataset)

# import pprint
# pprint.pp(result)


# from src.analysis.extreme_weather_analysis import ExtremeWeatherAnalysis

# extreme = ExtremeWeatherAnalysis()

# result = extreme.analyze(dataset)

# import pprint

# pprint.pp(result)

# from src.analysis.anomaly_detection import AnomalyDetection

# detector = AnomalyDetection()

# result = detector.analyze(dataset)

# import pprint

# pprint.pp(result)