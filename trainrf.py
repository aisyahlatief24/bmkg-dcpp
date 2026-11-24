
# %% [markdown]
# # 06 - Training Random Forest untuk Downscaling PR
# Dataset extended DCPP-A s1981-s2019
# Training: 1220 pairs
# Testing : 500 pairs
# Target  : CHIRPS precipitation (mm/day)

# %%
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DATASET_DIR = os.path.join(PROCESSED_DIR, "training_dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

VARNAME = "pr"
SPATIAL_THIN_FACTOR = 4

RF_ESTIMATORS = 100
RF_MAX_DEPTH = 15
RANDOM_STATE = 42

# =====================================================
# 2. LOAD EXTENDED TRAINING DATASET
# =====================================================

dataset = np.load(
    os.path.join(DATASET_DIR, "pr_dataset.npz")
)

X_train_grid = dataset["X_train"].astype(np.float32)
y_train_grid = dataset["Y_train"].astype(np.float32)

X_test_grid = dataset["X_test"].astype(np.float32)
y_test_grid = dataset["Y_test"].astype(np.float32)

train_indices = dataset["train_indices"].astype(int)
test_indices = dataset["test_indices"].astype(int)

print("=== EXTENDED DATASET ===")
print("X_train:", X_train_grid.shape)
print("Y_train:", y_train_grid.shape)
print("X_test :", X_test_grid.shape)
print("Y_test :", y_test_grid.shape)

assert X_train_grid.shape == y_train_grid.shape
assert X_test_grid.shape == y_test_grid.shape
assert len(train_indices) == len(X_train_grid)
assert len(test_indices) == len(X_test_grid)

# =====================================================
# 3. LOAD SPATIAL COORDINATES AND TEMPORAL METADATA
# =====================================================

model_da = xr.open_dataarray(
    os.path.join(PROCESSED_DIR, "pr_model_regridded.nc")
)

mask = np.load(
    os.path.join(DATASET_DIR, "pr_mask.npy")
).astype(bool)

lat_vals = model_da.lat.values
lon_vals = model_da.lon.values

assert mask.shape == (len(lat_vals), len(lon_vals))

# Pastikan indeks pasangan sesuai dengan dataset
assert np.all(train_indices < model_da.sizes["pair"])
assert np.all(test_indices < model_da.sizes["pair"])

# Koordinat grid valid
valid_flat_indices = np.flatnonzero(mask.ravel())

assert len(valid_flat_indices) == X_train_grid.shape[1]

# =====================================================
# 4. SPATIAL THINNING
# =====================================================

# Pertahankan pola thinning lama:
# mengambil satu titik setiap empat grid pada masing-masing sumbu.

thin_grid = np.zeros(mask.shape, dtype=bool)
thin_grid[::SPATIAL_THIN_FACTOR, ::SPATIAL_THIN_FACTOR] = True

selected_grid = mask & thin_grid

selected_flat_indices = np.flatnonzero(
    selected_grid.ravel()
)

# Posisi kolom pada dataset yang sudah difilter mask
selected_columns = np.searchsorted(
    valid_flat_indices,
    selected_flat_indices
)

assert np.all(
    valid_flat_indices[selected_columns] == selected_flat_indices
)

# Ambil hanya kolom grid terpilih
X_train_grid = X_train_grid[:, selected_columns]
y_train_grid = y_train_grid[:, selected_columns]

X_test_grid = X_test_grid[:, selected_columns]
y_test_grid = y_test_grid[:, selected_columns]

# Koordinat setiap grid terpilih
lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")

selected_lat = lat_grid.ravel()[selected_flat_indices]
selected_lon = lon_grid.ravel()[selected_flat_indices]

n_train, n_grid = X_train_grid.shape
n_test = len(X_test_grid)

print("\n=== SPATIAL THINNING ===")
print("Grid valid awal :", len(valid_flat_indices))
print("Grid setelah thinning:", n_grid)
print("Training pairs:", n_train)
print("Testing pairs :", n_test)

# =====================================================
# 5. BUILD FEATURE TABLE
# =====================================================

def get_init_year(indices):
    dates = model_da.init_date.values[indices]

    return np.array([
        int(str(date).replace("s", ""))
        for date in dates
    ])


def build_feature_table(X_grid, Y_grid, indices):
    n_pairs, n_points = X_grid.shape

    # Metadata waktu berdasarkan indeks pasangan
    lead_years = model_da.lead_year.values[indices]
    init_years = get_init_year(indices)

    # Fitur spasial dan temporal
    lat_feature = np.tile(selected_lat, n_pairs)
    lon_feature = np.tile(selected_lon, n_pairs)

    lead_feature = np.repeat(lead_years, n_points)
    init_feature = np.repeat(init_years, n_points)

    model_feature = X_grid.ravel()
    target = Y_grid.ravel()

    features = np.column_stack([
        model_feature,
        lat_feature,
        lon_feature,
        lead_feature,
        init_feature
    ])

    # Hapus nilai tidak valid
    valid = (
        np.isfinite(features).all(axis=1)
        & np.isfinite(target)
    )

    return features[valid], target[valid]


print("\nMenyusun feature table...")

X_train, y_train = build_feature_table(
    X_train_grid,
    y_train_grid,
    train_indices
)

X_test, y_test = build_feature_table(
    X_test_grid,
    y_test_grid,
    test_indices
)

print("\n=== FEATURE TABLE ===")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

print("Jumlah fitur:", X_train.shape[1])
print("Fitur: model_value, lat, lon, lead_year, init_year")

# =====================================================
# 6. TRAIN RANDOM FOREST
# =====================================================

print("\nMelatih Random Forest PR...")

rf_model = RandomForestRegressor(
    n_estimators=RF_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

rf_model.fit(X_train, y_train)

print("Training selesai.")

# =====================================================
# 7. PREDICTION AND EVALUATION
# =====================================================

y_pred = rf_model.predict(X_test)

# Baseline: nilai CMIP6 mentah
y_baseline = X_test[:, 0]


def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    bias = np.mean(y_pred - y_true)

    return {
        "Metode": name,
        "RMSE": rmse,
        "MAE": mae,
        "Correlation": corr,
        "Bias": bias
    }


results = [
    evaluate(y_test, y_baseline, "Model Mentah"),
    evaluate(y_test, y_pred, "Random Forest")
]

results_df = pd.DataFrame(results)

print("\n=== HASIL EVALUASI PR ===")
print(results_df.to_string(index=False))

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "06_rf_evaluation_pr.csv"),
    index=False
)

# =====================================================
# 8. FEATURE IMPORTANCE
# =====================================================

feature_names = [
    "model_value",
    "lat",
    "lon",
    "lead_year",
    "init_year"
]

importance_df = pd.DataFrame({
    "Fitur": feature_names,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(importance_df.to_string(index=False))

importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "06_rf_feature_importance_pr.csv"),
    index=False
)

# =====================================================
# 9. SCATTER PLOT
# =====================================================

plt.figure(figsize=(6, 6))

sample_idx = np.random.choice(
    len(y_test),
    size=min(5000, len(y_test)),
    replace=False
)

plt.scatter(
    y_test[sample_idx],
    y_pred[sample_idx],
    alpha=0.3,
    s=5,
    label="Random Forest"
)

plt.scatter(
    y_test[sample_idx],
    y_baseline[sample_idx],
    alpha=0.3,
    s=5,
    color="orange",
    label="CMIP6 Raw"
)

lims = [
    min(y_test.min(), y_pred.min()),
    max(y_test.max(), y_pred.max())
]

plt.plot(lims, lims, "k--", label="Ideal")

plt.xlabel("CHIRPS Observation (mm/day)")
plt.ylabel("Predicted Precipitation (mm/day)")
plt.title("Random Forest vs CMIP6 Raw - PR")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "06_rf_scatter_pr.png"),
    dpi=150
)

plt.close()

# =====================================================
# 10. SAVE MODEL AND PREDICTIONS
# =====================================================

joblib.dump(
    rf_model,
    os.path.join(MODELS_DIR, "rf_pr.pkl")
)

# Prediksi hasil thinning
np.savez_compressed(
    os.path.join(PROCESSED_DIR, "rf_pr_predictions.npz"),
    y_test=y_test,
    y_pred=y_pred,
    y_baseline=y_baseline
)

# Simpan metadata grid yang digunakan
np.savez_compressed(
    os.path.join(PROCESSED_DIR, "rf_pr_grid_metadata.npz"),
    selected_flat_indices=selected_flat_indices,
    selected_lat=selected_lat,
    selected_lon=selected_lon,
    train_indices=train_indices,
    test_indices=test_indices
)

model_da.close()

print("\nSemua output Random Forest PR berhasil disimpan.")
print("=" * 60)
print("SELESAI RANDOM FOREST UNTUK PR")
print("=" * 60)
