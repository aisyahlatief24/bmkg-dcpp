# %% [markdown]#
# Mencakup:
# 1. Statistik deskriptif (mean, min, max, std)
# 2. Peta klimatologi (rata-rata spasial)
# 3. Time series tren tahunan
# 4. Pola musiman (rata-rata bulanan)
# 5. Cek missing value
# 6. Distribusi data (histogram)
#
# Dijalankan untuk KEDUA variabel (tas & pr), dari data OBSERVASI
# (ERA5/CHIRPS) karena itu yang punya rentang waktu kalender penuh
# dan paling representatif untuk gambaran umum kondisi Indonesia.

# %%
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "eksplorasi")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Load Data Observasi (Rentang Waktu Kalender Penuh)

# %%
era5_ds = xr.open_dataset(os.path.join(RAW_DIR, "era5", "era5_tas_indonesia_1982_2020.nc"))
tas_obs = era5_ds["t2m"] - 273.15  # Kelvin -> Celsius
tas_obs = tas_obs.rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})
tas_obs.attrs["units"] = "degC"

chirps_ds = xr.open_dataset(os.path.join(RAW_DIR, "chirps", "chirps_monthly_indonesia_1982_2020.nc"))
pr_obs = chirps_ds["precip"]
pr_obs = pr_obs.rename({"latitude": "lat", "longitude": "lon"})
# Konversi mm/bulan -> mm/hari
days_in_month = pr_obs["time"].dt.days_in_month
pr_obs = pr_obs / days_in_month
pr_obs.attrs["units"] = "mm/day"

print(f"TAS (ERA5): {dict(tas_obs.sizes)}, rentang {str(tas_obs.time.values[0])[:10]} - {str(tas_obs.time.values[-1])[:10]}")
print(f"PR (CHIRPS): {dict(pr_obs.sizes)}, rentang {str(pr_obs.time.values[0])[:10]} - {str(pr_obs.time.values[-1])[:10]}")

# %% [markdown]
# ## 2. Statistik Deskriptif

# %%
def print_descriptive_stats(da, name, unit):
    vals = da.values
    vals_valid = vals[~np.isnan(vals)]
    print(f"\n--- Statistik Deskriptif: {name} ---")
    print(f"  Mean   : {vals_valid.mean():.3f} {unit}")
    print(f"  Median : {np.median(vals_valid):.3f} {unit}")
    print(f"  Std    : {vals_valid.std():.3f} {unit}")
    print(f"  Min    : {vals_valid.min():.3f} {unit}")
    print(f"  Max    : {vals_valid.max():.3f} {unit}")
    print(f"  Jumlah nilai nol (khusus curah hujan): {(vals_valid == 0).sum():,} dari {len(vals_valid):,} ({100*(vals_valid==0).mean():.1f}%)")
    return {
        "Variabel": name, "Mean": vals_valid.mean(), "Median": np.median(vals_valid),
        "Std": vals_valid.std(), "Min": vals_valid.min(), "Max": vals_valid.max()
    }

stats_tas = print_descriptive_stats(tas_obs, "TAS (Suhu)", "°C")
stats_pr = print_descriptive_stats(pr_obs, "PR (Curah Hujan)", "mm/hari")

stats_df = pd.DataFrame([stats_tas, stats_pr])
stats_df.to_csv(os.path.join(OUTPUT_DIR, "statistik_deskriptif.csv"), index=False)
print("\nDisimpan: outputs/eksplorasi/statistik_deskriptif.csv")

# %% [markdown]
# ## 3. Cek Missing Value

# %%
def check_anomali(da, name, valid_range):
    n_total = da.size
    n_nan = int(np.isnan(da.values).sum())
    n_out_of_range = int(((da.values < valid_range[0]) | (da.values > valid_range[1])).sum())
    print(f"\n--- Cek Anomali: {name} ---")
    print(f"  Total titik data     : {n_total:,}")
    print(f"  Missing value (NaN)  : {n_nan:,} ({100*n_nan/n_total:.2f}%)")
    print(f"  Di luar rentang wajar {valid_range}: {n_out_of_range:,}")
    if n_nan == 0 and n_out_of_range == 0:
        print("  Status: tidak ada anomali.")
    else:
        print("  Status: PERLU DICEK LEBIH LANJUT.")

check_anomali(tas_obs, "TAS", valid_range=(-10, 45))   # rentang wajar suhu Indonesia
check_anomali(pr_obs, "PR", valid_range=(0, 500))      # rentang wajar curah hujan harian

# %% [markdown]
# ## 4. Peta Klimatologi (Rata-rata Spasial Sepanjang Waktu)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

tas_clim = tas_obs.mean(dim="time")
im0 = tas_clim.plot(ax=axes[0], cmap="coolwarm", add_colorbar=True)
axes[0].set_title("Klimatologi Suhu Rata-rata 1982-2020 (°C)")

pr_clim = pr_obs.mean(dim="time")
im1 = pr_clim.plot(ax=axes[1], cmap="Blues", add_colorbar=True)
axes[1].set_title("Klimatologi Curah Hujan Rata-rata 1982-2020 (mm/hari)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_peta_klimatologi.png"), dpi=100)
plt.close()
print("\nGambar tersimpan: outputs/eksplorasi/01_peta_klimatologi.png")

# %% [markdown]
# ## 5. Time Series Tren Tahunan (Rata-rata Spasial per Tahun)

# %%
tas_annual = tas_obs.mean(dim=["lat", "lon"]).groupby("time.year").mean()
pr_annual = pr_obs.mean(dim=["lat", "lon"]).groupby("time.year").mean()

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(tas_annual.year, tas_annual.values, marker='o', color='firebrick')
axes[0].set_title("Tren Suhu Rata-rata Tahunan Indonesia (ERA5)")
axes[0].set_ylabel("Suhu (°C)")
axes[0].grid(alpha=0.3)

axes[1].plot(pr_annual.year, pr_annual.values, marker='o', color='steelblue')
axes[1].set_title("Tren Curah Hujan Rata-rata Tahunan Indonesia (CHIRPS)")
axes[1].set_ylabel("Curah Hujan (mm/hari)")
axes[1].set_xlabel("Tahun")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_tren_tahunan.png"), dpi=100)
plt.close()
print("Gambar tersimpan: outputs/eksplorasi/02_tren_tahunan.png")

# Hitung tren linear sederhana
from numpy.polynomial import polynomial as P
tas_trend_per_year = np.polyfit(tas_annual.year, tas_annual.values, 1)[0]
pr_trend_per_year = np.polyfit(pr_annual.year, pr_annual.values, 1)[0]
print(f"\nTren suhu: {tas_trend_per_year:+.4f} °C/tahun ({tas_trend_per_year*39:+.2f} °C selama 39 tahun)")
print(f"Tren curah hujan: {pr_trend_per_year:+.4f} mm/hari/tahun")

# %% [markdown]
# ## 6. Pola Musiman (Rata-rata per Bulan, Semua Tahun Digabung)

# %%
tas_monthly = tas_obs.mean(dim=["lat", "lon"]).groupby("time.month").mean()
pr_monthly = pr_obs.mean(dim=["lat", "lon"]).groupby("time.month").mean()

month_names = ['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des']

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(month_names, tas_monthly.values, marker='o', color='firebrick')
axes[0].axhline(tas_monthly.mean().item(), linestyle='--', color='gray', alpha=0.6, label='Rata-rata')
axes[0].set_title("Pola Musiman Suhu (Klimatologi Bulanan)")
axes[0].set_ylabel("Suhu (°C)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].bar(month_names, pr_monthly.values, color='steelblue')
axes[1].set_title("Pola Musiman Curah Hujan (Klimatologi Bulanan)")
axes[1].set_ylabel("Curah Hujan (mm/hari)")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_pola_musiman.png"), dpi=100)
plt.close()
print("Gambar tersimpan: outputs/eksplorasi/03_pola_musiman.png")

# %% [markdown]
# ## 7. Distribusi Data (Histogram)
# (banyak nilai nol/rendah, ekor panjang untuk nilai ekstrem)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

tas_vals = tas_obs.values.flatten()
tas_vals = tas_vals[~np.isnan(tas_vals)]
axes[0].hist(tas_vals, bins=50, color='firebrick', alpha=0.7)
axes[0].set_title("Distribusi Nilai Suhu (TAS)")
axes[0].set_xlabel("Suhu (°C)")
axes[0].set_ylabel("Frekuensi")

pr_vals = pr_obs.values.flatten()
pr_vals = pr_vals[~np.isnan(pr_vals)]
axes[1].hist(pr_vals, bins=50, color='steelblue', alpha=0.7)
axes[1].set_title("Distribusi Nilai Curah Hujan (PR)\n(zero-inflated)")
axes[1].set_xlabel("Curah Hujan (mm/hari)")
axes[1].set_ylabel("Frekuensi")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_distribusi_histogram.png"), dpi=100)
plt.close()
print("Gambar tersimpan: outputs/eksplorasi/04_distribusi_histogram.png")

# %% [markdown]
# ## 8. Bandingkan Karakteristik Model (DCPP) vs Observasi
# Cek apakah pola spasial kasar dari model konsisten dengan observasi
# (pemeriksaan awal sebelum downscaling - bukan evaluasi akurasi)

# %%
tas_model_path = os.path.join(PROCESSED_DIR, "tas_indonesia_alllead.nc")
pr_model_path = os.path.join(PROCESSED_DIR, "pr_indonesia_alllead.nc")

if os.path.exists(tas_model_path) and os.path.exists(pr_model_path):
    tas_model = xr.open_dataarray(tas_model_path)
    pr_model = xr.open_dataarray(pr_model_path)

    tas_model_clim = tas_model.mean(dim=["sample", "lead_year"])
    pr_model_clim = pr_model.mean(dim=["sample", "lead_year"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    tas_model_clim.plot(ax=axes[0, 0], cmap="coolwarm")
    axes[0, 0].set_title("Model DCPP - Suhu (grid kasar)")

    tas_clim.plot(ax=axes[0, 1], cmap="coolwarm")
    axes[0, 1].set_title("Observasi ERA5 - Suhu (grid halus)")

    pr_model_clim.plot(ax=axes[1, 0], cmap="Blues")
    axes[1, 0].set_title("Model DCPP - Curah Hujan (grid kasar)")

    pr_clim.plot(ax=axes[1, 1], cmap="Blues")
    axes[1, 1].set_title("Observasi CHIRPS - Curah Hujan (grid halus)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_perbandingan_model_vs_obs.png"), dpi=100)
    plt.close()
    print("Gambar tersimpan: outputs/eksplorasi/05_perbandingan_model_vs_obs.png")
    print("\nCatatan: perbedaan resolusi (kotak-kotak besar vs detail halus) di sini")
else:
    print("\nFile data model belum ditemukan, skip perbandingan model vs observasi.")