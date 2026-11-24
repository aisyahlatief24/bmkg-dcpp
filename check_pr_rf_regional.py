
import xarray as xr
import numpy as np
import pandas as pd

# ==========================================
# 1. LOAD DATA
# ==========================================

ds = xr.open_dataset(
    "outputs/pr_rf_spatial_validation.nc"
)

# ==========================================
# 2. DEFINISI WILAYAH
# ==========================================

regions = {
    "Sumatra": {
        "lat": (-6, 6),
        "lon": (95, 106)
    },
    "Jawa": {
        "lat": (-9, -5),
        "lon": (105, 115)
    },
    "Bali dan Nusa Tenggara": {
        "lat": (-12, -7),
        "lon": (114, 130)
    },
    "Kalimantan": {
        "lat": (-5, 5),
        "lon": (108, 119)
    },
    "Sulawesi": {
        "lat": (-6, 6),
        "lon": (119, 125)
    },
    "Maluku": {
        "lat": (-9, 3),
        "lon": (125, 132)
    },
    "Papua": {
        "lat": (-11, 1),
        "lon": (132, 145)
    }
}

# ==========================================
# 3. HITUNG STATISTIK REGIONAL
# ==========================================

results = []

for region, bounds in regions.items():

    subset = ds.where(
        (ds.lat >= bounds["lat"][0]) &
        (ds.lat <= bounds["lat"][1]) &
        (ds.lon >= bounds["lon"][0]) &
        (ds.lon <= bounds["lon"][1]),
        drop=True
    )

    chirps = subset["chirps_mean"].values
    raw = subset["raw_mean"].values
    rf = subset["rf_mean"].values

    bias_raw = subset["bias_raw"].values
    bias_rf = subset["bias_rf"].values

    # Ambil grid yang valid pada ketiga dataset
    valid = (
        np.isfinite(chirps) &
        np.isfinite(raw) &
        np.isfinite(rf)
    )

    if valid.sum() == 0:
        continue

    # Rata-rata nilai curah hujan
    chirps_avg = np.mean(chirps[valid])
    raw_avg = np.mean(raw[valid])
    rf_avg = np.mean(rf[valid])

    # Rata-rata bias
    raw_bias_avg = np.nanmean(bias_raw)
    rf_bias_avg = np.nanmean(bias_rf)

    results.append({
        "Wilayah": region,
        "Jumlah Grid": int(valid.sum()),
        "CHIRPS Mean": chirps_avg,
        "Raw Mean": raw_avg,
        "RF Mean": rf_avg,
        "Raw Bias": raw_bias_avg,
        "RF Bias": rf_bias_avg,
        "Perubahan Bias": abs(raw_bias_avg) - abs(rf_bias_avg)
    })

# ==========================================
# 4. TAMPILKAN HASIL
# ==========================================

df = pd.DataFrame(results)

print("\n=== HASIL ANALISIS REGIONAL PR RANDOM FOREST ===")
print(df.round(3).to_string(index=False))

# Simpan hasil
df.to_csv(
    "outputs/pr_rf_regional_summary.csv",
    index=False
)

print("\nFile tersimpan:")
print("outputs/pr_rf_regional_summary.csv")
