from setuptools import find_packages, setup


setup(
    name="weather-forecasting-system",
    version="2.1.0",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "xarray",
        "netCDF4",
        "cfgrib",
        "pandas",
        "numpy",
        "matplotlib",
        "cartopy",
        "geopandas",
        "PyQt5",
        "seaborn",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "torch",
    ],
)
