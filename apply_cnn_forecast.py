import os
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_DIR = os.path.join(
    BASE_DIR, "data", "processed", "forecast_s2020"
)
OUTPUT_DIR = os.path.join(FORECAST_DIR, "cnn_corrected")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 1. LOAD MODEL DAN STATISTIK NORMALISASI
# --------------------------------------------------

MODEL_PATH = os.path.join(MODELS_DIR, "cnn_tas.h5")
STATS_PATH = os.path.join(MODELS_DIR, "cnn_tas_norm_stats.npz")

FORECAST_PATH = os.path.join(
    FORECAST_DIR,
    "tas_dcppB_forecast_s2020_regridded_to_era5.nc"
)

cnn_model = load_model(MODEL_PATH, compile=False)
stats = np.load(STATS_PATH)

ch_mean = stats["ch_mean"]
ch_std = stats["ch_std"]
y_mean = float(stats["y_mean"])
y_std = float(stats["y_std"])

print("Model CNN berhasil dimuat.")
print("ch_mean:", ch_mean)
print("ch_std :", ch_std)
print("y_mean :", y_mean)
print("y_std  :", y_std)

# --------------------------------------------------
# 2. LOAD FORECAST TAS
# --------------------------------------------------

ds = xr.open_dataset(FORECAST_PATH)
da = ds["tas"]

raw = da.values

members = da.member.values
lead_years = da.lead_year.values
lat = da.lat.values
lon = da.lon.values

n_sample, n_lead, n_lat, n_lon = raw.shape

print("\n=== INFORMASI FORECAST ===")
print("Shape:", raw.shape)
print("Members:", members)
print("Lead years:", lead_years)
print("Grid:", n_lat, "x", n_lon)

# Inisialisasi forecast s2020
init_year = int(
    str(ds["init_date"].values[0]).replace("s", "")
)

# --------------------------------------------------
# 3. SIAPKAN CHANNEL SPASIAL
# --------------------------------------------------

lon_grid, lat_grid = np.meshgrid(lon, lat)

lat_ch = np.broadcast_to(
    lat_grid, (n_lat, n_lon)
)
lon_ch = np.broadcast_to(
    lon_grid, (n_lat, n_lon)
)

# Padding sama seperti saat training
pad_lat = (-n_lat) % 4
pad_lon = (-n_lon) % 4

print("\nPadding latitude:", pad_lat)
print("Padding longitude:", pad_lon)

# --------------------------------------------------
# 4. PREDIKSI SETIAP MEMBER DAN LEAD YEAR
# --------------------------------------------------

corrected = np.full(
    raw.shape, np.nan, dtype=np.float32
)
for s in range(n_sample):
    member = str(members[s])

    for l in range(n_lead):

        ly = int(lead_years[l])
        model_value = raw[s, l]

        # CNN hanya menggunakan 1 channel: TAS
        X = model_value[..., np.newaxis].astype(np.float32)

        # Normalisasi menggunakan statistik training
        X_norm = (X - ch_mean) / ch_std

        # Tangani NaN dan Inf
        X_norm = np.nan_to_num(
            X_norm,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # Tambahkan batch dimension
        X_input = X_norm[np.newaxis, ...]

        # Padding agar sesuai input U-Net (100, 224, 1)
        if pad_lat or pad_lon:
            X_input = np.pad(
                X_input,
                (
                    (0, 0),
                    (0, pad_lat),
                    (0, pad_lon),
                    (0, 0)
                ),
                mode="constant"
            )

        # Pastikan bentuk input benar
        assert X_input.shape == (1, 100, 224, 1), X_input.shape

        # Prediksi CNN
        pred_scaled = cnn_model.predict(
            X_input,
            verbose=0
        )

        # Hilangkan padding dan output channel
        pred_scaled = pred_scaled[
            0, :n_lat, :n_lon, 0
        ]

        # Kembalikan ke satuan Celsius
        pred = pred_scaled * y_std + y_mean

        corrected[s, l] = pred.astype(np.float32)

        print(
            f"Member {member}, LY{ly}: "
            f"min={np.nanmin(pred):.2f} °C, "
            f"max={np.nanmax(pred):.2f} °C"
        )
# --------------------------------------------------
# 5. SIMPAN HASIL FORECAST
# --------------------------------------------------

corrected_da = xr.DataArray(
    corrected,
    dims=("sample", "lead_year", "lat", "lon"),
    coords={
        "sample": da.sample.values,
        "lead_year": lead_years,
        "lat": lat,
        "lon": lon,
        "member": ("sample", members),
        "init_date": ("sample", ds.init_date.values),
    },
    name="tas_cnn_corrected",
    attrs={
        "long_name": "TAS bias-corrected using CNN U-Net",
        "units": "degC",
        "method": "CNN U-Net",
        "forecast_initialization": "s2020",
        "normalization": "Training statistics",
        "padding": "Bottom and right, to multiple of 4",
    },
)

output_path = os.path.join(
    OUTPUT_DIR,
    "tas_cnn_corrected_dcppB_s2020.nc",
)

corrected_da.to_dataset().to_netcdf(output_path)

print("\n" + "=" * 60)
print("SELESAI: CNN TAS FORECAST")
print("=" * 60)

print("Output:", output_path)
print("Shape:", corrected.shape)
print("Minimum:", float(np.nanmin(corrected)))
print("Maximum:", float(np.nanmax(corrected)))
print("NaN count:", int(np.isnan(corrected).sum()))

ds.close()
