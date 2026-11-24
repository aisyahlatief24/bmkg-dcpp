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

results = []

# Wilayah yang sama: Papua Tengah
for member_idx in range(5):

    raw = raw_ds["tas"].isel(
        sample=member_idx,
        lead_year=1
    )

    cnn = cnn_ds["tas_cnn_corrected"].isel(
        sample=member_idx,
        lead_year=1
    )

    lat_mask = (raw.lat >= -6) & (raw.lat <= -2)
    lon_mask = (raw.lon >= 135) & (raw.lon <= 140)

    correction = (cnn - raw).where(
        lat_mask & lon_mask,
        drop=True
    )

    values = correction.values
    values = values[np.isfinite(values)]

    results.append({
        "Member": str(raw_ds.member.values[member_idx]),
        "Grid Count": len(values),
        "Mean Correction": np.mean(values),
        "Minimum Correction": np.min(values),
        "Maximum Correction": np.max(values),
        "Grid < -5C": np.sum(values < -5),
        "Grid < -8C": np.sum(values < -8),
    })

df = pd.DataFrame(results)

print(df.to_string(index=False))

output = os.path.join(
    BASE_DIR,
    "outputs",
    "tas_extreme_member_comparison.csv"
)

os.makedirs(os.path.dirname(output), exist_ok=True)
df.to_csv(output, index=False)

print("\nCSV tersimpan:", output)

raw_ds.close()
cnn_ds.close()