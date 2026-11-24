import os
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
FORECAST = os.path.join(PROCESSED, "forecast_s2020")

# Data historis
model = xr.open_dataarray(
    os.path.join(PROCESSED, "tas_model_regridded.nc")
)

# Data forecast
forecast = xr.open_dataset(
    os.path.join(
        FORECAST,
        "tas_dcppB_forecast_s2020_regridded_to_era5.nc"
    )
)["tas"]

locations = [
    (-4.00, 137.25),
    (-4.00, 137.00),
    (-4.00, 138.25),
    (-4.00, 138.00),
    (-4.00, 137.50),
]

results = []

for lat, lon in locations:
    iy = np.abs(model.lat.values - lat).argmin()
    ix = np.abs(model.lon.values - lon).argmin()

    # Nilai historis
    historical = model.isel(lat=iy, lon=ix).values
    historical = historical[np.isfinite(historical)]

    # Nilai forecast: member r1, seluruh lead year
    future = forecast.isel(
        sample=0,
        lat=iy,
        lon=ix
    ).values
    future = future[np.isfinite(future)]

    results.append({
        "Latitude": float(model.lat.values[iy]),
        "Longitude": float(model.lon.values[ix]),
        "Historical Mean": float(np.mean(historical)),
        "Historical Min": float(np.min(historical)),
        "Historical Max": float(np.max(historical)),
        "Forecast Mean": float(np.mean(future)),
        "Forecast Min": float(np.min(future)),
        "Forecast Max": float(np.max(future)),
    })

print(pd.DataFrame(results).to_string(index=False))

model.close()