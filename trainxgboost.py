
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DATASET_DIR = os.path.join(PROCESSED_DIR, "training_dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================
# PENGATURAN
# ============================================================

VARNAME = "pr"
SPATIAL_THIN_FACTOR = 4

XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05

RANDOM_STATE = 42

# ============================================================
# 1. LOAD EXTENDED TRAINING DATASET
# ============================================================

dataset_path = os.path.join(DATASET_DIR, "pr_dataset.npz")

data = np.load(dataset_path)

X_train_grid = data["X_train"]
Y_train_grid = data["Y_train"]
X_test_grid = data["X_test"]
Y_test_grid = data["Y_test"]

train_indices = data["train_indices"]
test_indices = data["test_indices"]

print("=== EXTENDED DATASET ===")
print("X_train:", X_train_grid.shape)
print("Y_train:", Y_train_grid.shape)
print("X_test :", X_test_grid.shape)
print("Y_test :", Y_test_grid.shape)

# ============================================================
# 2. LOAD COORDINATES, METADATA, DAN MASK
# ============================================================

model_path = os.path.join(
    PROCESSED_DIR,
    "pr_model_regridded.nc"
)

mask_path = os.path.join(
    DATASET_DIR,
    "pr_mask.npy"
)

model_da = xr.open_dataarray(model_path)
valid_mask = np.load(mask_path).astype(bool)

n_lat = model_da.sizes["lat"]
n_lon = model_da.sizes["lon"]

if valid_mask.shape != (n_lat, n_lon):
    raise ValueError("Dimensi PR mask tidak sesuai dengan grid model.")

valid_flat_indices = np.flatnonzero(valid_mask.ravel())

if len(valid_flat_indices) != X_train_grid.shape[1]:
    raise ValueError(
        "Jumlah grid valid pada mask tidak sesuai dengan dataset."
    )

# ============================================================
# 3. SPATIAL THINNING
# ============================================================

thin_grid = np.zeros((n_lat, n_lon), dtype=bool)
thin_grid[::SPATIAL_THIN_FACTOR, ::SPATIAL_THIN_FACTOR] = True

selected_mask = valid_mask & thin_grid
selected_flat_indices = np.flatnonzero(selected_mask.ravel())

# Mapping indeks grid terpilih ke kolom dataset PR
selected_columns = np.searchsorted(
    valid_flat_indices,
    selected_flat_indices
)

X_train_selected = X_train_grid[:, selected_columns]
Y_train_selected = Y_train_grid[:, selected_columns]

X_test_selected = X_test_grid[:, selected_columns]
Y_test_selected = Y_test_grid[:, selected_columns]

print("\n=== SPATIAL THINNING ===")
print("Grid valid awal :", len(valid_flat_indices))
print("Grid setelah thinning:", len(selected_columns))
print("Training pairs:", len(train_indices))
print("Testing pairs :", len(test_indices))

# Pastikan grid yang digunakan sama dengan RF PR
if len(selected_columns) != 369:
    raise ValueError(
        f"Jumlah grid setelah thinning bukan 369, melainkan {len(selected_columns)}."
    )

# ============================================================
# 4. MENYUSUN FEATURE TABLE
# ============================================================

lat_vals = model_da.lat.values
lon_vals = model_da.lon.values

lat_grid, lon_grid = np.meshgrid(
    lat_vals,
    lon_vals,
    indexing="ij"
)

lat_selected = lat_grid.ravel()[selected_flat_indices]
lon_selected = lon_grid.ravel()[selected_flat_indices]

def get_init_year(values):
    return np.array([
        int(str(value).replace("s", ""))
        for value in values
    ])

init_year_all = get_init_year(model_da.init_date.values)
lead_year_all = model_da.lead_year.values

train_init_year = init_year_all[train_indices]
test_init_year = init_year_all[test_indices]

train_lead_year = lead_year_all[train_indices]
test_lead_year = lead_year_all[test_indices]


def build_feature_table(model_values, obs_values,
                        init_years, lead_years,
                        lat_values, lon_values):

    n_pairs, n_grid = model_values.shape

    lat_feature = np.tile(lat_values, n_pairs)
    lon_feature = np.tile(lon_values, n_pairs)

    lead_feature = np.repeat(lead_years, n_grid)
    init_feature = np.repeat(init_years, n_grid)

    model_feature = model_values.reshape(-1)
    target = obs_values.reshape(-1)

    features = np.column_stack([
        model_feature,
        lat_feature,
        lon_feature,
        lead_feature,
        init_feature
    ])

    valid = (
        np.isfinite(features).all(axis=1)
        & np.isfinite(target)
    )

    return features[valid], target[valid]


print("\nMenyusun feature table...")

X_train, y_train = build_feature_table(
    X_train_selected,
    Y_train_selected,
    train_init_year,
    train_lead_year,
    lat_selected,
    lon_selected
)

X_test, y_test = build_feature_table(
    X_test_selected,
    Y_test_selected,
    test_init_year,
    test_lead_year,
    lat_selected,
    lon_selected
)

print("\n=== FEATURE TABLE ===")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)
print("Jumlah fitur:", X_train.shape[1])
print("Fitur: model_value, lat, lon, lead_year, init_year")

# ============================================================
# 5. TRAIN XGBOOST PR
# ============================================================

print("\nMelatih XGBoost PR...")

xgb_model = XGBRegressor(
    n_estimators=XGB_N_ESTIMATORS,
    max_depth=XGB_MAX_DEPTH,
    learning_rate=XGB_LEARNING_RATE,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

xgb_model.fit(X_train, y_train)

print("Training selesai.")

# ============================================================
# 6. PREDIKSI DAN EVALUASI
# ============================================================

y_pred = xgb_model.predict(X_test)
y_baseline = X_test[:, 0]


def evaluate(y_true, y_prediction, name):

    rmse = np.sqrt(
        mean_squared_error(y_true, y_prediction)
    )

    mae = mean_absolute_error(
        y_true, y_prediction
    )

    corr = np.corrcoef(
        y_true, y_prediction
    )[0, 1]

    bias = np.mean(
        y_prediction - y_true
    )

    return {
        "Metode": name,
        "RMSE": rmse,
        "MAE": mae,
        "Correlation": corr,
        "Bias": bias
    }


results = [
    evaluate(
        y_test,
        y_baseline,
        "Model Mentah"
    ),
    evaluate(
        y_test,
        y_pred,
        "XGBoost"
    )
]

# Bandingkan dengan RF PR
rf_pred_path = os.path.join(
    PROCESSED_DIR,
    "rf_pr_predictions.npz"
)

if os.path.exists(rf_pred_path):

    rf_data = np.load(rf_pred_path)

    rf_y_test = rf_data["y_test"]
    rf_y_pred = rf_data["y_pred"]

    if (
        len(rf_y_test) == len(y_test)
        and np.allclose(rf_y_test, y_test)
    ):

        results.insert(
            1,
            evaluate(
                y_test,
                rf_y_pred,
                "Random Forest"
            )
        )

        print("\nRF PR berhasil dimasukkan ke perbandingan.")

    else:
        print(
            "\nPERINGATAN: Urutan atau nilai y_test RF berbeda."
        )
        print("RF tidak dimasukkan ke tabel perbandingan.")

else:
    print("\nFile RF PR tidak ditemukan.")

results_df = pd.DataFrame(results)

print("\n=== HASIL EVALUASI PR ===")
print(results_df.to_string(index=False))

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "07_xgb_evaluation_pr.csv"
    ),
    index=False
)

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================

feature_names = [
    "model_value",
    "lat",
    "lon",
    "lead_year",
    "init_year"
]

importance_df = pd.DataFrame({
    "Fitur": feature_names,
    "Importance": xgb_model.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

print("\n=== FEATURE IMPORTANCE ===")
print(importance_df.to_string(index=False))

importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "07_xgb_feature_importance_pr.csv"
    ),
    index=False
)

# ============================================================
# 8. SCATTER PLOT
# ============================================================

np.random.seed(RANDOM_STATE)

sample_idx = np.random.choice(
    len(y_test),
    size=min(5000, len(y_test)),
    replace=False
)

plt.figure(figsize=(6, 6))

plt.scatter(
    y_test[sample_idx],
    y_pred[sample_idx],
    alpha=0.3,
    s=5,
    label="XGBoost",
    color="green"
)

plt.scatter(
    y_test[sample_idx],
    y_baseline[sample_idx],
    alpha=0.3,
    s=5,
    label="Model Mentah",
    color="orange"
)

lims = [
    min(y_test.min(), y_pred.min(), y_baseline.min()),
    max(y_test.max(), y_pred.max(), y_baseline.max())
]

plt.plot(
    lims,
    lims,
    "k--",
    label="Garis ideal (y=x)"
)

plt.xlabel("Observasi CHIRPS (mm/day)")
plt.ylabel("Prediksi Curah Hujan (mm/day)")
plt.title("XGBoost vs Model Mentah - PR")
plt.legend()
plt.tight_layout()

scatter_path = os.path.join(
    OUTPUT_DIR,
    "07_xgb_scatter_pr.png"
)

plt.savefig(scatter_path, dpi=100)
plt.close()

print("\nGambar tersimpan:", scatter_path)

# ============================================================
# 9. SIMPAN MODEL DAN PREDIKSI
# ============================================================

model_save_path = os.path.join(
    MODELS_DIR,
    "xgb_pr.json"
)

xgb_model.save_model(model_save_path)

joblib.dump(
    xgb_model,
    os.path.join(MODELS_DIR, "xgb_pr.pkl")
)

np.savez(
    os.path.join(
        PROCESSED_DIR,
        "xgb_pr_predictions.npz"
    ),
    y_test=y_test,
    y_pred=y_pred,
    y_baseline=y_baseline
)

print("\nSemua output XGBoost PR berhasil disimpan.")

model_da.close()
data.close()

print("\n" + "=" * 60)
print("SELESAI XGBOOST UNTUK PR")
print("Metode ML selanjutnya: ANN PR")
print("=" * 60)
