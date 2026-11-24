import os
import xarray as xr
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_DIR = os.path.join(
    BASE_DIR, "data", "processed", "forecast_s2020"
)

raw_ds = xr.open_dataset(
    os.path.join(
        FORECAST_DIR,
        "tas_dcppB_forecast_s2020_regridded_to_era5.nc"
    )
)

cnn_ds = xr.open_dataset(
    os.path.join(
        FORECAST_DIR,
        "cnn_corrected",
        "tas_cnn_corrected_dcppB_s2020.nc"
    )
)

raw = raw_ds["tas"].isel(sample=0)
cnn = cnn_ds["tas_cnn_corrected"].isel(sample=0)

results = []

for ly in raw.lead_year.values:
    correction = (
        cnn.sel(lead_year=ly) - raw.sel(lead_year=ly)
    )

    stacked = correction.stack(point=("lat", "lon"))
    stacked = stacked.dropna("point")

    # Lima koreksi paling negatif
    lowest = stacked.sortby(stacked).isel(point=slice(0, 5))

    for rank in range(5):
        value = float(lowest.isel(point=rank).values)
        lat = float(lowest.lat.isel(point=rank).values)
        lon = float(lowest.lon.isel(point=rank).values)

        results.append({
            "Lead Year": int(ly),
            "Rank": rank + 1,
            "Latitude": lat,
            "Longitude": lon,
            "Correction": value
        })

df = pd.DataFrame(results)

print(df.to_string(index=False))

output_path = os.path.join(
    BASE_DIR, "outputs", "tas_extreme_locations_by_leadyear.csv"
)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print("\nCSV tersimpan:", output_path)

raw_ds.close()
cnn_ds.close()