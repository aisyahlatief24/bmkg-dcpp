import os
import numpy as np
import xarray as xr
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

forecast_path = os.path.join(
    BASE_DIR, "data", "processed", "forecast_s2020",
    "tas_dcppB_forecast_s2020_regridded_to_era5.nc"
)

model_path = os.path.join(BASE_DIR, "models", "cnn_tas.h5")
stats_path = os.path.join(
    BASE_DIR, "models", "cnn_tas_norm_stats.npz"
)

ds = xr.open_dataset(forecast_path)
da = ds["tas"]

model = load_model(model_path, compile=False)
stats = np.load(stats_path)

ch_mean = stats["ch_mean"]
ch_std = stats["ch_std"]
y_mean = float(stats["y_mean"])
y_std = float(stats["y_std"])

# Member r1, Lead Year 1
raw = da.isel(sample=0, lead_year=0).values
lat = da.lat.values
lon = da.lon.values

n_lat, n_lon = raw.shape
lon_grid, lat_grid = np.meshgrid(lon, lat)

pad_lat = (-n_lat) % 4
pad_lon = (-n_lon) % 4

def predict_with_init_year(year):
    X = np.stack([
        raw,
        lat_grid,
        lon_grid,
        np.full(raw.shape, 1),
        np.full(raw.shape, year)
    ], axis=-1).astype(np.float32)

    X = np.nan_to_num((X - ch_mean) / ch_std)
    X = X[np.newaxis, ...]

    if pad_lat or pad_lon:
        X = np.pad(
            X,
            ((0, 0), (0, pad_lat), (0, pad_lon), (0, 0))
        )

    pred = model.predict(X, verbose=0)[0, :n_lat, :n_lon, 0]
    return pred * y_std + y_mean

pred_2020 = predict_with_init_year(2020)
pred_2004 = predict_with_init_year(2004)

diff = pred_2020 - pred_2004

print("=== SENSITIVITAS INIT_YEAR ===")
print("Mean pred 2020:", np.mean(pred_2020))
print("Mean pred 2004:", np.mean(pred_2004))
print("Mean absolute difference:", np.mean(np.abs(diff)))
print("Maximum absolute difference:", np.max(np.abs(diff)))
print("Correlation:", np.corrcoef(
    pred_2020.ravel(), pred_2004.ravel()
)[0, 1])

ds.close()
