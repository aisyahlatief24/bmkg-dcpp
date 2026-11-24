
import os
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FORECAST_DIR = os.path.join(PROCESSED_DIR, "forecast_s2020")

# Lokasi 41 grid yang telah teridentifikasi
extreme_file = os.path.join(
    BASE_DIR, "outputs", "tas_historical_extreme_relation.csv"
)
extreme_df = pd.read_csv(extreme_file)

# Forecast CNN
cnn_da = xr.open_dataset(
    os.path.join(
        FORECAST_DIR, "cnn_corrected",
        "tas_cnn_corrected_dcppB_s2020.nc"
    )
)["tas_cnn_corrected"]

# Forecast mentah
raw_da = xr.open_dataarray(
    os.path.join(
        FORECAST_DIR,
        "tas_dcppB_forecast_s2020_regridded_to_era5.nc"
    )
)

# Prediksi historis CNN
hist_pred = np.load(
    os.path.join(
        PROCESSED_DIR, "cnn_tas_predictions_fullgrid.npz"
    )
)

y_test = hist_pred["y_test"]
y_pred = hist_pred["y_pred"]

hist_model = xr.open_dataarray(
    os.path.join(PROCESSED_DIR, "tas_model_regridded.nc")
)

hist_obs = xr.open_dataarray(
    os.path.join(PROCESSED_DIR, "tas_obs_matched.nc")
)

rows = []

for _, point in extreme_df.iterrows():
    lat = point["latitude"]
    lon = point["longitude"]

    # Cari indeks grid berdasarkan koordinat
    i = int(np.argmin(np.abs(hist_model.lat.values - lat)))
    j = int(np.argmin(np.abs(hist_model.lon.values - lon)))

    # Statistik testing historis
    historical_era5 = np.nanmean(y_test[:, i, j])
    historical_cnn = np.nanmean(y_pred[:, i, j])
    historical_error = historical_cnn - historical_era5

    # Forecast r1, LY2
    raw = raw_da.sel(
        member="r1i1p1f1", lead_year=2
    ).values[i, j]

    corrected = cnn_da.sel(
        member="r1i1p1f1", lead_year=2
    ).values[i, j]

    rows.append({
        "latitude": lat,
        "longitude": lon,
        "historical_era5_test_mean": historical_era5,
        "historical_cnn_test_mean": historical_cnn,
        "historical_cnn_error": historical_error,
        "raw_forecast": raw,
        "cnn_forecast": corrected,
        "cnn_forecast_minus_era5_mean": corrected - historical_era5,
        "raw_minus_historical_era5": raw - historical_era5
    })

result = pd.DataFrame(rows)

result = result.sort_values(
    "cnn_forecast_minus_era5_mean"
)

output = os.path.join(
    BASE_DIR, "outputs",
    "tas_cnn_forecast_vs_historical.csv"
)

result.to_csv(output, index=False)

print("\n=== 15 GRID: FORECAST CNN VS ERA5 HISTORIS ===")
print(result.head(15).round(3).to_string(index=False))

print("\n=== RINGKASAN ===")
print(result[[
    "historical_cnn_error",
    "cnn_forecast_minus_era5_mean",
    "raw_minus_historical_era5"
]].describe().round(3))

print("\nCSV tersimpan:", output)
