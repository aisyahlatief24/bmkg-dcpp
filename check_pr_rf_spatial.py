
import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

CUTOFF_YEAR = 2005
THIN_FACTOR = 4

# ==================================================
# 1. LOAD DATA
# ==================================================

raw_ds = xr.open_dataset(
    os.path.join(PROCESSED_DIR, "pr_model_regridded.nc")
)

obs_ds = xr.open_dataset(
    os.path.join(PROCESSED_DIR, "pr_obs_matched.nc")
)

model_da = raw_ds["pr_model"]
obs_da = obs_ds["pr_obs"]

rf_data = np.load(
    os.path.join(PROCESSED_DIR, "rf_pr_predictions.npz")
)

# ==================================================
# 2. RECONSTRUCT TEST DATA
# ==================================================

init_years = np.array([
    int(str(x).replace("s", ""))
    for x in model_da.init_date.values
])

test_indices = np.where(init_years >= CUTOFF_YEAR)[0]

lat_vals = model_da.lat.values[::THIN_FACTOR]
lon_vals = model_da.lon.values[::THIN_FACTOR]

model_sub = model_da.isel(pair=test_indices).sel(
    lat=lat_vals, lon=lon_vals
)

obs_sub = obs_da.isel(pair=test_indices).sel(
    lat=lat_vals, lon=lon_vals
)

# ==================================================
# 3. RECONSTRUCT SPATIAL VALIDITY MASK
# ==================================================

model_values = model_sub.values
obs_values = obs_sub.values

valid_masks = (
    np.isfinite(model_values) &
    np.isfinite(obs_values)
)

# Pastikan jumlah valid sama dengan jumlah sampel RF
assert valid_masks.sum() == len(rf_data["y_pred"]), (
    "Jumlah sampel valid tidak sama dengan jumlah prediksi RF"
)

# ==================================================
# 4. MAP BACK RF PREDICTIONS TO GRID
# ==================================================

n_pairs, n_lat, n_lon = model_values.shape

raw_grid = np.full((n_pairs, n_lat, n_lon), np.nan)
obs_grid = np.full((n_pairs, n_lat, n_lon), np.nan)
rf_grid = np.full((n_pairs, n_lat, n_lon), np.nan)

raw_grid[valid_masks] = rf_data["y_baseline"]
obs_grid[valid_masks] = rf_data["y_test"]
rf_grid[valid_masks] = rf_data["y_pred"]

# ==================================================
# 5. AVERAGE OVER TEST PERIOD
# ==================================================

def safe_nanmean(data, axis=0):
    valid_count = np.isfinite(data).sum(axis=axis)
    total = np.nansum(data, axis=axis)

    return np.divide(
        total,
        valid_count,
        out=np.full(total.shape, np.nan, dtype=float),
        where=valid_count > 0
    )

raw_mean = safe_nanmean(raw_grid)
obs_mean = safe_nanmean(obs_grid)
rf_mean = safe_nanmean(rf_grid)

bias_raw = raw_mean - obs_mean
bias_rf = rf_mean - obs_mean

improvement = np.abs(bias_raw) - np.abs(bias_rf)

# ==================================================
# 5B. SPATIAL RMSE DAN IDENTIFIKASI GRID BERMASALAH
# ==================================================

# Menghitung RMSE temporal pada setiap grid
def safe_nanmean(data, axis=0):
    valid_count = np.isfinite(data).sum(axis=axis)
    total = np.nansum(data, axis=axis)

    return np.divide(
        total,
        valid_count,
        out=np.full(total.shape, np.nan, dtype=float),
        where=valid_count > 0
    )

rmse_raw = np.sqrt(
    safe_nanmean((raw_grid - obs_grid) ** 2)
)

rmse_rf = np.sqrt(
    safe_nanmean((rf_grid - obs_grid) ** 2)
)

# Positif = RMSE RF lebih baik
rmse_improvement = rmse_raw - rmse_rf

# Grid valid
valid_grid = (
    np.isfinite(rmse_raw) &
    np.isfinite(rmse_rf)
)

print("\n=== SPATIAL RMSE SUMMARY ===")
print("Raw mean RMSE:", np.nanmean(rmse_raw))
print("RF mean RMSE :", np.nanmean(rmse_rf))
print("Grid RMSE membaik:", np.sum(
    valid_grid & (rmse_improvement > 0)
))
print("Grid RMSE memburuk:", np.sum(
    valid_grid & (rmse_improvement < 0)
))

# Identifikasi grid dengan RMSE RF paling tinggi
lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")

problem_df = pd.DataFrame({
    "lat": lat_grid[valid_grid],
    "lon": lon_grid[valid_grid],
    "raw_bias": bias_raw[valid_grid],
    "rf_bias": bias_rf[valid_grid],
    "raw_rmse": rmse_raw[valid_grid],
    "rf_rmse": rmse_rf[valid_grid],
    "rmse_improvement": rmse_improvement[valid_grid]
})

problem_df = problem_df.sort_values(
    "rmse_improvement",
    ascending=True
)

problem_path = os.path.join(
    OUTPUT_DIR, "pr_rf_spatial_problem_locations.csv"
)

problem_df.to_csv(problem_path, index=False)

print("\n=== 15 GRID DENGAN PERBAIKAN RMSE TERBURUK ===")
print(problem_df.head(15).round(4).to_string(index=False))

print("\nCSV tersimpan:", problem_path)

# ==================================================
# 6. SUMMARY
# ==================================================

def spatial_stats(name, values):
    values = values[np.isfinite(values)]

    return {
        "Dataset": name,
        "Mean": np.mean(values),
        "Std": np.std(values),
        "Min": np.min(values),
        "Max": np.max(values)
    }

print("\n=== SPATIAL BIAS SUMMARY ===")

for name, values in [
    ("Raw Bias", bias_raw),
    ("RF Bias", bias_rf),
    ("Improvement", improvement)
]:
    stats = spatial_stats(name, values)
    print(stats)

print("\nJumlah grid:", np.isfinite(obs_mean).sum())
print(
    "Grid RF memperbaiki bias:",
    np.sum(improvement > 0)
)
print(
    "Grid RF memperburuk bias:",
    np.sum(improvement < 0)
)

# ==================================================
# 7. SAVE SPATIAL DATA
# ==================================================

spatial_ds = xr.Dataset(
    {
        "chirps_mean": (
            ("lat", "lon"), obs_mean
        ),
        "raw_mean": (
            ("lat", "lon"), raw_mean
        ),
        "rf_mean": (
            ("lat", "lon"), rf_mean
        ),
        "bias_raw": (
            ("lat", "lon"), bias_raw
        ),
        "bias_rf": (
            ("lat", "lon"), bias_rf
        ),
        "bias_improvement": (
            ("lat", "lon"), improvement
        )
    },
    coords={
        "lat": lat_vals,
        "lon": lon_vals
    }
)

spatial_path = os.path.join(
    OUTPUT_DIR, "pr_rf_spatial_validation.nc"
)

spatial_ds.to_netcdf(spatial_path)

# ==================================================
# 8. PLOT MAPS
# ==================================================

maps = [
    (obs_mean, "CHIRPS Reference", "mm/day"),
    (raw_mean, "Raw CMCC-CM2-SR5", "mm/day"),
    (rf_mean, "Random Forest", "mm/day"),
    (bias_raw, "Raw Bias (Raw - CHIRPS)", "mm/day"),
    (bias_rf, "RF Bias (RF - CHIRPS)", "mm/day"),
    (improvement, "Absolute Bias Improvement", "mm/day")
]

fig, axes = plt.subplots(
    2, 3,
    figsize=(18, 10),
    constrained_layout=True
)

for ax, (data, title, unit) in zip(axes.flat, maps):
    image = ax.pcolormesh(
        lon_vals,
        lat_vals,
        data,
        shading="auto",
        cmap="RdBu_r" if "Bias" in title or "Improvement" in title
        else "YlGnBu"
    )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(90, 145)
    ax.set_ylim(-13, 11)

    fig.colorbar(image, ax=ax, label=unit)

fig.suptitle(
    "PR Spatial Validation: CHIRPS vs Raw vs Random Forest",
    fontsize=15
)

plot_path = os.path.join(
    OUTPUT_DIR, "pr_rf_spatial_validation.png"
)

plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()

print("\nFile tersimpan:")
print(spatial_path)
print(plot_path)

raw_ds.close()
obs_ds.close()
rf_data.close()


print("\n=== RINGKASAN GRID SPASIAL ===")

for name, data in {
    "CHIRPS": obs_mean,
    "Raw": raw_mean,
    "RF": rf_mean,
    "Raw Bias": bias_raw,
    "RF Bias": bias_rf,
    "Improvement": improvement
}.items():
    valid = data[np.isfinite(data)]

    print(f"\n{name}")
    print("Min :", np.min(valid))
    print("Max :", np.max(valid))
    print("Mean:", np.mean(valid))
    print("Std :", np.std(valid))