
import os
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pengaturan harus sama dengan script training RF
CUTOFF_YEAR = 2005
THIN_FACTOR = 4

# 1. Load data historis
raw_ds = xr.open_dataset(
    os.path.join(PROCESSED_DIR, "pr_model_regridded.nc")
)
obs_ds = xr.open_dataset(
    os.path.join(PROCESSED_DIR, "pr_obs_matched.nc")
)

model_da = raw_ds["pr_model"]
obs_da = obs_ds["pr_obs"]

# 2. Load hasil prediksi RF
rf_data = np.load(
    os.path.join(PROCESSED_DIR, "rf_pr_predictions.npz")
)

y_test_saved = rf_data["y_test"]
y_pred = rf_data["y_pred"]
y_baseline_saved = rf_data["y_baseline"]

# 3. Rekonstruksi train-test split
init_years = np.array([
    int(str(x).replace("s", ""))
    for x in model_da.init_date.values
])

test_mask = init_years >= CUTOFF_YEAR
test_indices = np.where(test_mask)[0]

print("Jumlah pasangan test:", len(test_indices))

# 4. Rekonstruksi spatial thinning
lat_vals = model_da.lat.values[::THIN_FACTOR]
lon_vals = model_da.lon.values[::THIN_FACTOR]

model_sub = model_da.isel(pair=test_indices).sel(
    lat=lat_vals, lon=lon_vals
)
obs_sub = obs_da.isel(pair=test_indices).sel(
    lat=lat_vals, lon=lon_vals
)

# 5. Buat ulang urutan sampel seperti script training
raw_list = []
obs_list = []

model_values = model_sub.values
obs_values = obs_sub.values

for i in range(len(test_indices)):
    m_flat = model_values[i].flatten()
    o_flat = obs_values[i].flatten()

    valid = (
        np.isfinite(m_flat) &
        np.isfinite(o_flat)
    )

    raw_list.append(m_flat[valid])
    obs_list.append(o_flat[valid])

raw_reconstructed = np.concatenate(raw_list)
obs_reconstructed = np.concatenate(obs_list)

# 6. Validasi kecocokan dengan file prediksi RF
print("\n=== VALIDASI REKONSTRUKSI ===")
print("Reconstructed:", len(raw_reconstructed))
print("Saved y_test:", len(y_test_saved))
print("Saved y_baseline:", len(y_baseline_saved))
print("Saved y_pred:", len(y_pred))

assert len(raw_reconstructed) == len(y_baseline_saved)
assert len(obs_reconstructed) == len(y_test_saved)
assert len(y_pred) == len(y_test_saved)

print(
    "y_test cocok:",
    np.allclose(obs_reconstructed, y_test_saved, equal_nan=True)
)
print(
    "y_baseline cocok:",
    np.allclose(raw_reconstructed, y_baseline_saved, equal_nan=True)
)

# Hentikan jika urutan atau nilai tidak cocok
if not np.allclose(obs_reconstructed, y_test_saved):
    raise ValueError("Rekonstruksi CHIRPS tidak cocok dengan y_test!")

if not np.allclose(raw_reconstructed, y_baseline_saved):
    raise ValueError("Rekonstruksi Raw tidak cocok dengan y_baseline!")

# 7. Statistik distribusi pada sampel yang sama
datasets = {
    "CHIRPS": obs_reconstructed,
    "Raw": y_baseline_saved,
    "Random Forest": y_pred
}

rows = []

for name, values in datasets.items():
    values = np.asarray(values)
    rows.append({
        "Dataset": name,
        "N": len(values),
        "Mean": np.mean(values),
        "Std": np.std(values),
        "Min": np.min(values),
        "P01": np.percentile(values, 1),
        "P05": np.percentile(values, 5),
        "Median": np.median(values),
        "P95": np.percentile(values, 95),
        "P99": np.percentile(values, 99),
        "Max": np.max(values),
        "Below_1_mm_day_pct": np.mean(values < 1) * 100,
        "Below_2_mm_day_pct": np.mean(values < 2) * 100,
        "Below_5_mm_day_pct": np.mean(values < 5) * 100,
        "Above_10_mm_day_pct": np.mean(values > 10) * 100,
        "Above_15_mm_day_pct": np.mean(values > 15) * 100
    })

summary = pd.DataFrame(rows)

print("\n=== HISTORICAL TEST DISTRIBUTION ===")
print(summary.round(4).to_string(index=False))

output_path = os.path.join(
    OUTPUT_DIR, "pr_historical_test_distribution.csv"
)

summary.to_csv(output_path, index=False)

print("\nCSV tersimpan:", output_path)

raw_ds.close()
obs_ds.close()
rf_data.close()
