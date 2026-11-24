import os
import xarray as xr
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_DIR = os.path.join(
    BASE_DIR, "data", "processed", "forecast_s2020"
)

raw_path = os.path.join(
    FORECAST_DIR,
    "tas_dcppB_forecast_s2020_regridded_to_era5.nc"
)

cnn_path = os.path.join(
    FORECAST_DIR, "cnn_corrected",
    "tas_cnn_corrected_dcppB_s2020.nc"
)

raw_ds = xr.open_dataset(raw_path)
cnn_ds = xr.open_dataset(cnn_path)

raw = raw_ds["tas"].isel(sample=0)
cnn = cnn_ds["tas_cnn_corrected"].isel(sample=0)

regions = {
    "Papua Tengah": (-5, -3, 136, 139),
    "Papua Barat": (-2, 0, 132.5, 135),
    "Papua Tenggara": (-6.5, -5, 142.5, 144.5),
}

results = []

for name, (lat_min, lat_max, lon_min, lon_max) in regions.items():

    for ly in raw.lead_year.values:

        r = raw.sel(
            lead_year=ly,
            lat=slice(lat_max, lat_min),
            lon=slice(lon_min, lon_max)
        )

        c = cnn.sel(
            lead_year=ly,
            lat=slice(lat_max, lat_min),
            lon=slice(lon_min, lon_max)
        )

        d = c - r

        results.append({
            "Region": name,
            "Lead Year": int(ly),
            "RAW Mean": float(r.mean()),
            "CNN Mean": float(c.mean()),
            "Correction Mean": float(d.mean()),
            "Correction Min": float(d.min()),
            "Correction Max": float(d.max()),
        })

df = pd.DataFrame(results)

print(df.to_string(index=False))

output_path = os.path.join(
    BASE_DIR, "outputs",
    "tas_papua_correction_by_leadyear.csv"
)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print("\nCSV tersimpan:", output_path)

raw_ds.close()
cnn_ds.close()
