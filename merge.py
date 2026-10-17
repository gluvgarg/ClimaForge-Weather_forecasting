import xarray as xr

input_file = "era5_merged_clean.nc"
output_file = "era5_final.nc"

ds = xr.open_dataset(input_file)

# Keep only the meteorological variables we need
keep = ["u10", "v10", "t2m", "sp", "tp"]
ds = ds[keep]

# Add correct units
ds["u10"].attrs["units"] = "m s-1"
ds["v10"].attrs["units"] = "m s-1"
ds["t2m"].attrs["units"] = "K"
ds["sp"].attrs["units"] = "Pa"
ds["tp"].attrs["units"] = "m"

# Add readable names
ds["u10"].attrs["long_name"] = "10 metre U wind component"
ds["v10"].attrs["long_name"] = "10 metre V wind component"
ds["t2m"].attrs["long_name"] = "2 metre temperature"
ds["sp"].attrs["long_name"] = "Surface pressure"
ds["tp"].attrs["long_name"] = "Total precipitation"

# Save
ds.to_netcdf(output_file)

print(ds)
print(f"\nSaved as: {output_file}")