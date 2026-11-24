# %% [markdown]
# # 09 - Training CNN (U-Net) untuk Downscaling TAS
#
# Metode ML KEEMPAT dari 5. BEDA pendekatan dari RF/XGBoost/ANN: CNN bekerja
# di grid SPASIAL UTUH (bukan per-pixel tabular), jadi bisa belajar pola antar
# piksel tetangga secara langsung -- makanya di sini TIDAK ADA thinning untuk
# training (RF/XGBoost/ANN perlu thinning karena tidak paham struktur spasial,
# CNN justru butuh struktur spasial itu).
#
# 1. Load data hasil regridding (grid PENUH: 97x221)
# 2. Split train/test berdasarkan init_date (sama: < 2005 train, >= 2005 test)
# 3. Susun jadi tensor 4D (pair, lat, lon, channel) - channel = model_value,
#    lat, lon, lead_year, init_year (SAMA fitur dengan RF/XGBoost/ANN, cuma
#    di sini bentuknya "gambar" bukan tabel)
# 4. Normalisasi channel (CNN, seperti ANN, sensitif skala)
# 5. Latih CNN (U-Net sederhana, 2x downsampling - 2x upsampling)
# 6. Evaluasi DUA cara:
#    a. Grid penuh (seluruh 97x221 titik) - ini metrik "asli" CNN
#    b. Titik yang SAMA dengan RF/XGBoost/ANN (grid ter-thin) - ini yang
#       dipakai untuk tabel perbandingan akhir supaya adil (apple-to-apple)
# 7. Simpan model & prediksi

# %%
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import mean_squared_error, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Pengaturan split (SAMA seperti 06/07/08) ---
VARNAME ="tas"
TRAIN_TEST_CUTOFF_YEAR = 2005
SPATIAL_THIN_FACTOR = 4  # dipakai HANYA untuk evaluasi pembanding, bukan training

# --- Pengaturan khusus CNN ---
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 8
CNN_PATIENCE = 8  # early stopping
DATASET_DIR = os.path.join(
    PROCESSED_DIR, "training_dataset"
)
# %% [markdown]
# ## 1. Load Data Hasil Regridding (Grid Penuh)

# =====================================================
# 1. LOAD EXTENDED TRAINING DATASET
# =====================================================

dataset_path = os.path.join(
    DATASET_DIR, "tas_dataset.npz"
)

data = np.load(dataset_path)

print("Dataset keys:", data.files)

X_train_raw = data["X_train"].astype(np.float32)
y_train_raw = data["Y_train"].astype(np.float32)

X_test_raw = data["X_test"].astype(np.float32)
y_test_raw = data["Y_test"].astype(np.float32)

print("\n=== DATASET EXTENDED ===")
print("X_train:", X_train_raw.shape)
print("Y_train:", y_train_raw.shape)
print("X_test :", X_test_raw.shape)
print("Y_test :", y_test_raw.shape)

assert X_train_raw.shape == y_train_raw.shape
assert X_test_raw.shape == y_test_raw.shape
assert len(X_train_raw) == 1220
assert len(X_test_raw) == 500

# Grid penuh
n_lat_full = X_train_raw.shape[1]
n_lon_full = X_train_raw.shape[2]

# =====================================================
# 2. ADD CHANNEL DIMENSION
# =====================================================

# Satu channel: nilai TAS dari CMIP6
X_train_raw = X_train_raw[..., np.newaxis]
X_test_raw = X_test_raw[..., np.newaxis]

y_train_raw = y_train_raw[..., np.newaxis]
y_test_raw = y_test_raw[..., np.newaxis]

print("\n=== CHANNEL DIMENSION ===")
print("X_train:", X_train_raw.shape)
print("X_test :", X_test_raw.shape)

# =====================================================
# 3. NORMALIZATION
# =====================================================

# Statistik dihitung dari training saja
x_mean = np.mean(X_train_raw, dtype=np.float64)
x_std = np.std(X_train_raw, dtype=np.float64)

y_mean = np.mean(y_train_raw, dtype=np.float64)
y_std = np.std(y_train_raw, dtype=np.float64)

assert x_std > 0
assert y_std > 0

X_train = ((X_train_raw - x_mean) / x_std).astype(np.float32)
X_test = ((X_test_raw - x_mean) / x_std).astype(np.float32)

y_train = ((y_train_raw - y_mean) / y_std).astype(np.float32)

print("\n=== NORMALIZATION ===")
print("X mean:", x_mean)
print("X std :", x_std)
print("Y mean:", y_mean)
print("Y std :", y_std)

# =====================================================
# 4. PADDING FOR U-NET
# =====================================================

pad_lat = (-n_lat_full) % 4
pad_lon = (-n_lon_full) % 4

if pad_lat or pad_lon:
    pad_width = (
        (0, 0),
        (0, pad_lat),
        (0, pad_lon),
        (0, 0)
    )

    X_train = np.pad(X_train, pad_width)
    X_test = np.pad(X_test, pad_width)
    y_train = np.pad(y_train, pad_width)

print("\n=== PADDING ===")
print("Padding latitude :", pad_lat)
print("Padding longitude:", pad_lon)
print("Final X_train:", X_train.shape)
print("Final X_test :", X_test.shape)
print("Final y_train:", y_train.shape)

# %% [markdown]
# ## 5. Bangun & Latih U-Net

# %%


def build_unet(input_shape):
    inp = layers.Input(shape=input_shape)

    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(inp)
    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    b = layers.Conv2D(64, 3, activation="relu", padding="same")(p2)
    b = layers.Conv2D(64, 3, activation="relu", padding="same")(b)

    u2 = layers.UpSampling2D()(b)
    u2 = layers.Concatenate()([u2, c2])
    u2 = layers.Conv2D(32, 3, activation="relu", padding="same")(u2)

    u1 = layers.UpSampling2D()(u2)
    u1 = layers.Concatenate()([u1, c1])
    u1 = layers.Conv2D(16, 3, activation="relu", padding="same")(u1)

    out = layers.Conv2D(1, 1, activation="linear", padding="same")(u1)
    return Model(inp, out)


print("\nMelatih CNN (U-Net))")
cnn_model = build_unet(X_train.shape[1:])
cnn_model.compile(optimizer="adam", loss="mse")
cnn_model.summary()

history = cnn_model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=CNN_EPOCHS,
    batch_size=CNN_BATCH_SIZE,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        patience=CNN_PATIENCE, restore_best_weights=True)],
    verbose=1,
)
print("Selesai training.")

# %% [markdown]
# ## 6. Prediksi & Evaluasi

# %%
pred_scaled = cnn_model.predict(
    X_test,
    batch_size=CNN_BATCH_SIZE
)

pred_scaled = pred_scaled[
    :, :n_lat_full, :n_lon_full, 0
]

pred_full = pred_scaled * y_std + y_mean

y_test_full = y_test_raw[..., 0]
baseline_full = X_test_raw[..., 0]

def evaluate(y_true, y_pred, name):
    yt, yp = y_true.flatten(), y_pred.flatten()
    mask = ~np.isnan(yt) & ~np.isnan(yp)
    yt, yp = yt[mask], yp[mask]
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    corr = np.corrcoef(yt, yp)[0, 1]
    bias = (yp - yt).mean()
    return {"Metode": name, "RMSE": rmse, "MAE": mae, "Correlation": corr, "Bias": bias}


# 6a. Evaluasi GRID PENUH (metrik "asli" CNN, semua ~97x221 titik)
result_full = evaluate(y_test_full, pred_full, "CNN_UNet (grid penuh)")
result_baseline_full = evaluate(
    y_test_full, baseline_full, "Model Mentah (grid penuh)")

print("\n=== HASIL EVALUASI - GRID PENUH ===")
print(pd.DataFrame([result_baseline_full, result_full]).to_string(index=False))

# 6b. Evaluasi TITIK YANG SAMA dengan RF/XGBoost/ANN (grid ter-thin)
#     -- supaya bisa dibandingkan apple-to-apple di 11_compare_all.py
pred_thin = pred_full[:, ::SPATIAL_THIN_FACTOR, ::SPATIAL_THIN_FACTOR]
y_test_thin = y_test_full[:, ::SPATIAL_THIN_FACTOR, ::SPATIAL_THIN_FACTOR]
baseline_thin = baseline_full[:, ::SPATIAL_THIN_FACTOR, ::SPATIAL_THIN_FACTOR]

y_test_thin_flat = y_test_thin.flatten()
pred_thin_flat = pred_thin.flatten()
baseline_thin_flat = baseline_thin.flatten()
valid = ~np.isnan(y_test_thin_flat) & ~np.isnan(pred_thin_flat)
y_test_thin_flat = y_test_thin_flat[valid]
pred_thin_flat = pred_thin_flat[valid]
baseline_thin_flat = baseline_thin_flat[valid]

results_compare = [
    evaluate(y_test_thin_flat, baseline_thin_flat,
             "Model Mentah (tanpa koreksi)"),
    evaluate(y_test_thin_flat, pred_thin_flat, "CNN_UNet"),
]

for prev_name, prev_file in [("Random Forest", f"rf_{VARNAME}_predictions.npz"),
                             ("XGBoost", f"xgb_{VARNAME}_predictions.npz"),
                             ("ANN", f"ann_{VARNAME}_predictions.npz")]:
    prev_path = os.path.join(PROCESSED_DIR, prev_file)
    if os.path.exists(prev_path):
        prev_data = np.load(prev_path)
        if len(prev_data["y_test"]) == len(y_test_thin_flat):
            results_compare.append(
                evaluate(prev_data["y_test"], prev_data["y_pred"], prev_name))
        else:
            print(f"Catatan: jumlah baris y_test {prev_name} beda ({len(prev_data['y_test'])} vs "
                  f"{len(y_test_thin_flat)}), dilewati dari tabel perbandingan.")
    else:
        print(f"Catatan: {prev_path} tidak ditemukan, {prev_name} dilewati.")

results_compare_df = pd.DataFrame(results_compare)
print("\n=== HASIL EVALUASI - TITIK SAMA DENGAN RF/XGBOOST/ANN (untuk perbandingan akhir) ===")
print(results_compare_df.to_string(index=False))

results_compare_df.to_csv(os.path.join(
    OUTPUT_DIR, f"09_cnn_evaluation_{VARNAME}.csv"), index=False)

# %% [markdown]
# ## 7. Visualisasi

# %%
# Kurva loss
plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE, skala ter-standarisasi)")
plt.title(f"Kurva Loss Training - CNN {VARNAME.upper()}")
plt.legend()
plt.savefig(os.path.join(
    OUTPUT_DIR, f"09_cnn_loss_curve_{VARNAME}.png"), dpi=100)
plt.close()

# Scatter (titik ter-thin, biar sebanding dengan gambar RF/XGB/ANN)
plt.figure(figsize=(6, 6))
sample_idx = np.random.choice(len(y_test_thin_flat), size=min(
    5000, len(y_test_thin_flat)), replace=False)
plt.scatter(y_test_thin_flat[sample_idx], pred_thin_flat[sample_idx],
            alpha=0.3, s=5, label="CNN_UNet", color="red")
plt.scatter(y_test_thin_flat[sample_idx], baseline_thin_flat[sample_idx],
            alpha=0.3, s=5, label="Model Mentah", color="orange")
lims = [min(y_test_thin_flat.min(), pred_thin_flat.min()),
        max(y_test_thin_flat.max(), pred_thin_flat.max())]
plt.plot(lims, lims, 'k--', label="Garis ideal (y=x)")
plt.xlabel("Observasi (°C)")
plt.ylabel("Prediksi (°C)")
plt.title(f"CNN (U-Net) vs Model Mentah - {VARNAME.upper()}")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, f"09_cnn_scatter_{VARNAME}.png"), dpi=100)
plt.close()

# Peta contoh 1 pair: obs vs prediksi vs model mentah (biar keliatan pola spasialnya)
sample_pair = 0
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
vmin, vmax = np.nanmin(y_test_full[sample_pair]), np.nanmax(
    y_test_full[sample_pair])
for ax, data, title in zip(
    axes,
    [baseline_full[sample_pair], pred_full[sample_pair], y_test_full[sample_pair]],
    ["Model Mentah", "CNN_UNet", "Observasi"],
):
    im = ax.imshow(data, origin="lower", vmin=vmin, vmax=vmax, cmap="coolwarm")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle(f"Contoh Peta Prediksi - Pair Test ke-{sample_pair}")
plt.tight_layout()
plt.savefig(os.path.join(
    OUTPUT_DIR, f"09_cnn_map_sample_{VARNAME}.png"), dpi=100)
plt.close()


# %% [markdown]
# ## 8. Simpan Model & Prediksi

# %%
cnn_model.save(os.path.join(MODELS_DIR, f"cnn_{VARNAME}.h5"))


np.savez(
    os.path.join(MODELS_DIR, f"cnn_{VARNAME}_norm_stats.npz"),
    ch_mean=ch_mean, ch_std=ch_std, y_mean=y_mean, y_std=y_std,
)
print(f"Model tersimpan: models/cnn_{VARNAME}.h5 (+ norm_stats.npz)")

# Simpan versi thinned (untuk 11_compare_all.py, konsisten dengan RF/XGB/ANN)
np.savez(
    os.path.join(PROCESSED_DIR, f"cnn_{VARNAME}_predictions.npz"),
    y_test=y_test_thin_flat, y_pred=pred_thin_flat, y_baseline=baseline_thin_flat,
)
print(
    f"Prediksi (thinned) tersimpan: data/processed/cnn_{VARNAME}_predictions.npz")

# Simpan juga versi grid penuh (untuk visualisasi peta / tahap forecast nanti)
np.savez(
    os.path.join(PROCESSED_DIR, f"cnn_{VARNAME}_predictions_fullgrid.npz"),
    y_test=y_test_full, y_pred=pred_full, y_baseline=baseline_full,
)
print(
    f"Prediksi (grid penuh) tersimpan: data/processed/cnn_{VARNAME}_predictions_fullgrid.npz")

print("\n" + "=" * 60)
print(f"SELESAI CNN (U-Net) untuk {VARNAME}.")
print("=" * 60)


print(X_train.shape)
print(X_test.shape)


import os
import numpy as np

BASE_DIR = r"c:\Users\Danella Nur Aisyah L\magangproject"
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Masukkan nilai statistik hasil training sebelumnya
x_mean = 27.49952367047912
x_std = 1.1985116680245143
y_mean = 26.593453548132942
y_std = 1.2771840418931144

np.savez(
    os.path.join(MODELS_DIR, "cnn_tas_norm_stats.npz"),
    ch_mean=x_mean,
    ch_std=x_std,
    y_mean=y_mean,
    y_std=y_std
)

print("Statistik normalisasi CNN TAS berhasil disimpan.")