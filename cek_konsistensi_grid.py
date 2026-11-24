
import os
import xarray as xr
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
FORECAST = os.path.join(PROCESSED, "forecast_s2020")

pairs = [
    (
        "TAS",
        os.path.join(PROCESSED, "tas_model_regridded.nc"),
        os.path.join(
            FORECAST,
            "tas_dcppB_forecast_s2020_regridded_to_era5.nc",
        ),
    ),
    (
        "PR",
        os.path.join(PROCESSED, "pr_model_regridded.nc"),
        os.path.join(
            FORECAST,
            "pr_dcppB_forecast_s2020_regridded_to_chirps025.nc",
        ),
    ),
]

for name, train_path, forecast_path in pairs:
    print(f"\n=== {name} ===")

    if not os.path.exists(train_path):
        print("File training tidak ditemukan:", train_path)
        continue

    with xr.open_dataarray(train_path) as train, \
         xr.open_dataarray(forecast_path) as forecast:

        print("Dimensi training:", dict(train.sizes))
        print("Dimensi forecast:", dict(forecast.sizes))

        for coord in ["lat", "lon"]:
            a = train[coord].values
            b = forecast[coord].values

            same = (
                a.shape == b.shape
                and np.allclose(a, b, rtol=0, atol=1e-6)
            )

            print(f"Koordinat {coord} sama:", same)

            if not same:
                print("  Training awal:", a[:5])
                print("  Forecast awal:", b[:5])

print("\nPengecekan selesai.")
