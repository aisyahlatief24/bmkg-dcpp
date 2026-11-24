
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import joblib

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DATASET_DIR = os.path.join(PROCESSED_DIR, "training_dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================
# PENGATURAN ANN
# ============================================================

VARNAME = "pr"
SPATIAL_THIN_FACTOR = 4

ANN_HIDDEN_LAYERS = (64, 32)
ANN_MAX_ITER = 300

RANDOM_STATE = 42

# ============================================================
# 1. LOAD EXTENDED TRAINING DATASET
# ============================================================

dataset_path = os.path.join(
    DATASET_DIR,
    "pr_dataset.npz"
)

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
# 2. LOAD COORDINATES DAN SPATIAL MASK
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
    raise ValueError("Dimensi spatial mask tidak sesuai.")

valid_flat_indices = np.flatnonzero(
    valid_mask.ravel()
)

if len(valid_flat_indices) != X_train_grid.shape[1]:
    raise ValueError(
        "Jumlah grid valid tidak sesuai dengan dataset."
    )

# ============================================================
# 3. SPATIAL THINNING
# ============================================================

thin_grid = np.zeros(
    (n_lat, n_lon),
    dtype=bool
)

thin_grid[
    ::SPATIAL_THIN_FACTOR,
    ::SPATIAL_THIN_FACTOR
] = True

selected_mask = valid_mask & thin_grid

selected_flat_indices = np.flatnonzero(
    selected_mask.ravel()
)

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

if len(selected_columns) != 369:
    raise ValueError(
        f"Grid setelah thinning seharusnya 369, "
        f"tetapi ditemukan {len(selected_columns)}."
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

lat_selected = lat_grid.ravel()[
    selected_flat_indices
]

lon_selected = lon_grid.ravel()[
    selected_flat_indices
]

def get_init_year(values):
    return np.array([
        int(str(value).replace("s", ""))
        for value in values
    ])

init_year_all = get_init_year(
    model_da.init_date.values
)

lead_year_all = model_da.lead_year.values

train_init_year = init_year_all[train_indices]
test_init_year = init_year_all[test_indices]

train_lead_year = lead_year_all[train_indices]
test_lead_year = lead_year_all[test_indices]


def build_feature_table(
    model_values,
    obs_values,
    init_years,
    lead_years,
    lat_values,
    lon_values
):

    n_pairs, n_grid = model_values.shape

    model_feature = model_values.reshape(-1)
    target = obs_values.reshape(-1)

    lat_feature = np.tile(
        lat_values,
        n_pairs
    )

    lon_feature = np.tile(
        lon_values,
        n_pairs
    )

    lead_feature = np.repeat(
        lead_years,
        n_grid
    )

    init_feature = np.repeat(
        init_years,
        n_grid
    )

    features = np.column_stack([
    model_feature,
    lat_feature,
    lon_feature
])
    print(
    "Fitur yang digunakan: model_value, lat, lon")
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
print(
    "Fitur: model_value, lat, lon, lead_year, init_year"
)

# ============================================================
# 5. STANDARDISASI FITUR DAN TARGET
# ============================================================

print("\nMelakukan standardisasi fitur dan target...")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Scaler hanya fit menggunakan data training
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(
    y_train.reshape(-1, 1)
).ravel()

# Baseline tetap menggunakan skala asli
y_baseline = X_test[:, 0]

print("Standardisasi selesai.")

# ============================================================
# 6. TRAIN ANN (MLP REGRESSOR)
# ============================================================

print("\nMelatih ANN (MLPRegressor)...")

ann_model = MLPRegressor(
    hidden_layer_sizes=ANN_HIDDEN_LAYERS,
    activation="relu",
    solver="adam",
    max_iter=ANN_MAX_ITER,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    random_state=RANDOM_STATE,
    verbose=True
)

ann_model.fit(
    X_train_scaled,
    y_train_scaled
)

print("\nTraining selesai.")
print("Jumlah iterasi:", ann_model.n_iter_)

# ============================================================
# 7. PREDIKSI DAN EVALUASI
# ============================================================

y_pred_scaled = ann_model.predict(
    X_test_scaled
)

# Kembalikan prediksi ke satuan mm/day
y_pred = scaler_y.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).ravel()
# ============================================================
# DIAGNOSTIK PREDIKSI ANN
# ============================================================

print("\n=== DIAGNOSTIK TARGET DAN PREDIKSI ===")

def print_stats(name, values):
    values = np.asarray(values)
    print(f"\n{name}")
    print("Min  :", np.min(values))
    print("Max  :", np.max(values))
    print("Mean :", np.mean(values))
    print("Std  :", np.std(values))
    print("NaN  :", np.isnan(values).sum())
    print("Inf  :", np.isinf(values).sum())

print_stats("y_train asli", y_train)
print_stats("y_test asli", y_test)
print_stats("y_pred_scaled", y_pred_scaled)
print_stats("y_pred ANN (mm/day)", y_pred)
print_stats("y_baseline (mm/day)", y_baseline)

print("\n=== PARAMETER SCALER TARGET ===")
print("Mean training target:", scaler_y.mean_[0])
print("Scale training target:", scaler_y.scale_[0])

print("\n=== PERBANDINGAN RATA-RATA ===")
print("Mean observasi test :", np.mean(y_test))
print("Mean model mentah   :", np.mean(y_baseline))
print("Mean prediksi ANN   :", np.mean(y_pred))

print("\n=== PERSENTIL PREDIKSI ANN ===")
print(np.percentile(y_pred, [0, 1, 5, 25, 50, 75, 95, 99, 100]))

def evaluate(y_true, y_prediction, name):

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_prediction
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_prediction
    )

    corr = np.corrcoef(
        y_true,
        y_prediction
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
        "ANN"
    )
]

# ============================================================
# 8. PERBANDINGAN DENGAN RF DAN XGBOOST
# ============================================================

previous_models = [
    ("Random Forest", "rf_pr_predictions.npz"),
    ("XGBoost", "xgb_pr_predictions.npz")
]

for prev_name, prev_file in previous_models:

    prev_path = os.path.join(
        PROCESSED_DIR,
        prev_file
    )

    if os.path.exists(prev_path):

        prev_data = np.load(prev_path)

        prev_y_test = prev_data["y_test"]
        prev_y_pred = prev_data["y_pred"]

        if (
            len(prev_y_test) == len(y_test)
            and np.allclose(
                prev_y_test,
                y_test,
                rtol=1e-7,
                atol=1e-9
            )
        ):

            results.append(
                evaluate(
                    y_test,
                    prev_y_pred,
                    prev_name
                )
            )

            print(
                f"{prev_name} berhasil dimasukkan."
            )

        else:

            print(
                f"PERINGATAN: y_test {prev_name} "
                "tidak cocok. Dilewati."
            )

        prev_data.close()

    else:

        print(
            f"File {prev_path} tidak ditemukan."
        )


results_df = pd.DataFrame(results)

print("\n=== HASIL EVALUASI ANN PR ===")
print(results_df.to_string(index=False))

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "08_ann_evaluation_pr.csv"
    ),
    index=False
)

# ============================================================
# 9. FEATURE IMPORTANCE - PERMUTATION IMPORTANCE
# ============================================================

# MLPRegressor tidak menyediakan feature_importances_
# seperti Random Forest dan XGBoost.
# Oleh karena itu, gunakan permutation importance.

from sklearn.inspection import permutation_importance

print("\nMenghitung permutation importance...")

# Subsample data testing agar komputasi lebih ringan
rng = np.random.default_rng(RANDOM_STATE)

importance_sample_size = min(
    10000,
    len(y_test)
)

importance_idx = rng.choice(
    len(y_test),
    size=importance_sample_size,
    replace=False
)

perm_result = permutation_importance(
    ann_model,
    X_test_scaled[importance_idx],
    scaler_y.transform(
        y_test[importance_idx].reshape(-1, 1)
    ).ravel(),
    n_repeats=5,
    random_state=RANDOM_STATE,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

feature_names = [
    "model_value",
    "lat",
    "lon"
]

importance_df = pd.DataFrame({
    "Fitur": feature_names,
    "Importance": perm_result.importances_mean
}).sort_values(
    "Importance",
    ascending=False
)

print("\n=== PERMUTATION IMPORTANCE ===")
print(importance_df.to_string(index=False))

importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "08_ann_feature_importance_pr.csv"
    ),
    index=False
)

# ============================================================
# 10. SCATTER PLOT
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
    label="ANN",
    color="purple"
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
    min(
        y_test.min(),
        y_pred.min(),
        y_baseline.min()
    ),
    max(
        y_test.max(),
        y_pred.max(),
        y_baseline.max()
    )
]

plt.plot(
    lims,
    lims,
    "k--",
    label="Garis ideal (y=x)"
)

plt.xlabel("Observasi CHIRPS (mm/day)")
plt.ylabel("Prediksi Curah Hujan (mm/day)")
plt.title("ANN vs Model Mentah - PR")
plt.legend()
plt.tight_layout()

scatter_path = os.path.join(
    OUTPUT_DIR,
    "08_ann_scatter_pr.png"
)

plt.savefig(
    scatter_path,
    dpi=100
)

plt.close()

print("\nScatter plot tersimpan:", scatter_path)

# ============================================================
# 11. LOSS CURVE
# ============================================================

plt.figure(figsize=(6, 4))

plt.plot(
    ann_model.loss_curve_,
    label="Training loss"
)

if hasattr(ann_model, "validation_scores_"):
    plt.plot(
        np.arange(
            1,
            len(ann_model.validation_scores_) + 1
        ),
        [
            1 - score
            for score in ann_model.validation_scores_
        ],
        label="Validation loss (1 - R²)"
    )

plt.xlabel("Iterasi")
plt.ylabel("Loss")
plt.title("Kurva Training ANN PR")
plt.legend()
plt.tight_layout()

loss_path = os.path.join(
    OUTPUT_DIR,
    "08_ann_loss_curve_pr.png"
)

plt.savefig(
    loss_path,
    dpi=100
)

plt.close()

print("Loss curve tersimpan:", loss_path)

# ============================================================
# 12. SIMPAN MODEL, SCALER, DAN PREDIKSI
# ============================================================

joblib.dump(
    ann_model,
    os.path.join(
        MODELS_DIR,
        "ann_pr.pkl"
    )
)

joblib.dump(
    scaler_X,
    os.path.join(
        MODELS_DIR,
        "ann_pr_scaler_X.pkl"
    )
)

joblib.dump(
    scaler_y,
    os.path.join(
        MODELS_DIR,
        "ann_pr_scaler_y.pkl"
    )
)

np.savez(
    os.path.join(
        PROCESSED_DIR,
        "ann_pr_predictions.npz"
    ),
    y_test=y_test,
    y_pred=y_pred,
    y_baseline=y_baseline
)

print("\nSemua output ANN PR berhasil disimpan.")

model_da.close()
data.close()

print("\n" + "=" * 60)
print("SELESAI ANN UNTUK PR")
print("Eksperimen ML PR: RF, XGBoost, dan ANN")
print("=" * 60)
