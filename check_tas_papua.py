
import os
import xarray as xr
import numpy as np

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

raw = raw_ds["tas"].isel(sample=0, lead_year=0)
cnn = cnn_ds["tas_cnn_corrected"].isel(
    sample=0, lead_year=0
)

correction = cnn - raw

# Area sekitar titik-titik koreksi ekstrem
regions = {
    "Papua Tengah": {
        "lat": (-5, -3),
        "lon": (136, 139)
    },
    "Papua Barat": {
        "lat": (-2, 0),
        "lon": (132.5, 135)
    },
    "Papua Tenggara": {
        "lat": (-6.5, -5),
        "lon": (142.5, 144.5)
    }
}

for name, bounds in regions.items():
    subset_raw = raw.sel(
        lat=slice(bounds["lat"][1], bounds["lat"][0]),
        lon=slice(*bounds["lon"])
    )
    subset_cnn = cnn.sel(
        lat=slice(bounds["lat"][1], bounds["lat"][0]),
        lon=slice(*bounds["lon"])
    )
    subset_diff = correction.sel(
        lat=slice(bounds["lat"][1], bounds["lat"][0]),
        lon=slice(*bounds["lon"])
    )

    print(f"\n=== {name} ===")
    print("Jumlah grid:", subset_raw.size)
    print("RAW mean:", float(subset_raw.mean()))
    print("CNN mean:", float(subset_cnn.mean()))
    print("Correction mean:", float(subset_diff.mean()))
    print("Correction min:", float(subset_diff.min()))
    print("Correction max:", float(subset_diff.max()))
    print("Correction std:", float(subset_diff.std()))

raw_ds.close()
cnn_ds.close()
