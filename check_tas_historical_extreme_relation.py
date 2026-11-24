
import os
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FORECAST_DIR = os.path.join(PROCESSED_DIR, "forecast_s2020")

# ==================================================
# 1. LOAD DATA
# ==================================================

# Forecast mentah
raw_da = xr.open_dataarray(
    os.path.join(
        FORECAST_DIR,
        "tas_dcppB_forecast_s2020_regridded_to_era5.nc"
    )
)

# Forecast hasil koreksi CNN
cnn_ds = xr.open_dataset(
    os.path.join(
        FORECAST_DIR,
        "cnn_corrected",
        "tas_cnn_corrected_dcppB_s2020.nc"
    )
)

cnn_da = cnn_ds["tas_cnn_corrected"]

# Data historis model dan referensi
hist_model = xr.open_dataarray(
    os.path.join(PROCESSED_DIR, "tas_model_regridded.nc")
)

hist_obs = xr.open_dataarray(
    os.path.join(PROCESSED_DIR, "tas_obs_matched.nc")
)

# Prediksi CNN pada data testing historis
hist_pred = np.load(
    os.path.join(
        PROCESSED_DIR,
        "cnn_tas_predictions_fullgrid.npz"
    )
)

y_test = hist_pred["y_test"]
y_pred = hist_pred["y_pred"]

# ==================================================
# 2. PILIH MEMBER R1 DAN LEAD YEAR 2
# ==================================================

raw = raw_da.sel(member="r1i1p1f1").sel(lead_year=2)
corrected = cnn_da.sel(member="r1i1p1f1").sel(lead_year=2)

# Samakan koordinat
corrected = corrected.sel(
    lat=raw.lat,
    lon=raw.lon
)

correction = corrected.values - raw.values

lat = raw.lat.values
lon = raw.lon.values

# ==================================================
# 3. PILIH GRID EKSTREM DI PAPUA
# ==================================================

lat_mask = (lat >= -6) & (lat <= -2)
lon_mask = (lon >= 135) & (lon <= 140)

region_mask = np.outer(lat_mask, lon_mask)

extreme_mask = region_mask & (correction < -5)

indices = np.argwhere(extreme_mask)

print("Jumlah grid ekstrem:", len(indices))

# ==================================================
# 4. HITUNG STATISTIK HISTORIS
# ==================================================

hist_model_values = hist_model.values
hist_obs_values = hist_obs.values

historical_error = y_pred - y_test

rows = []

for i, j in indices:
    # Koordinat lokasi
    latitude = lat[i]
    longitude = lon[j]

    # Forecast
    raw_value = raw.values[i, j]
    cnn_value = corrected.values[i, j]
    correction_value = correction[i, j]

    # Rata-rata historis seluruh pasangan data
    model_series = hist_model_values[:, i, j]
    obs_series = hist_obs_values[:, i, j]

    model_mean = np.nanmean(model_series)
    obs_mean = np.nanmean(obs_series)

    # Rata-rata error CNN pada 300 data testing
    error_series = historical_error[:, i, j]

    cnn_error_mean = np.nanmean(error_series)
    cnn_error_min = np.nanmin(error_series)
    cnn_error_max = np.nanmax(error_series)

    rows.append({
        "latitude": latitude,
        "longitude": longitude,
        "raw_forecast": raw_value,
        "cnn_forecast": cnn_value,
        "correction": correction_value,
        "historical_model_mean": model_mean,
        "historical_era5_mean": obs_mean,
        "historical_cnn_error_mean": cnn_error_mean,
        "historical_cnn_error_min": cnn_error_min,
        "historical_cnn_error_max": cnn_error_max
    })

# ==================================================
# 5. SIMPAN DAN TAMPILKAN HASIL
# ==================================================

df = pd.DataFrame(rows)

df = df.sort_values("correction")

output_path = os.path.join(
    BASE_DIR,
    "outputs",
    "tas_historical_extreme_relation.csv"
)

df.to_csv(output_path, index=False)

print("\n=== 15 GRID DENGAN KOREKSI PALING NEGATIF ===")
print(df.head(15).round(4).to_string(index=False))

print("\n=== RINGKASAN STATISTIK ===")
print(df.describe().round(4).to_string())

print("\nCSV tersimpan:", output_path)