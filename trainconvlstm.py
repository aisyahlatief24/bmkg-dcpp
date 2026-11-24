# %% [markdown]
# # 10 - Training ConvLSTM untuk Downscaling {VARNAME}
#
# Metode ML KELIMA (terakhir) dari 5. Beda dari 4 metode sebelumnya: ConvLSTM
# butuh SEQUENCE (urutan waktu), bukan snapshot tunggal. Di sini urutannya
# adalah 10 lead_year dalam satu init_date (mis. init 1990 -> prediksi
# 1991, 1992, ..., 2000 sebagai satu deret).
#
# KONSEKUENSI PENTING:
# - Satu "sample" untuk ConvLSTM = SATU init_date dengan SEMUA 10 lead_year-nya
#   lengkap. Kalau ada blok yang lead_year=1-nya hilang (kasus s1981 yang
#   sudah kita bahas), SELURUH blok itu (bukan cuma 1 titik) tidak bisa dipakai.
#   Makanya jumlah sample ConvLSTM jauh lebih sedikit dari RF/XGBoost/ANN/CNN.
# - Grid di-thin (sama seperti RF) karena ConvLSTM memproses 10 frame sekaligus
#   per sample -> jauh lebih berat secara komputasi daripada CNN biasa.
# - Populasi test-set ConvLSTM tidak 100% sama dengan 4 metode sebelumnya,
#   jadi tabel "HASIL EVALUASI" di sini TIDAK digabung otomatis dengan yang
#   lain (beda populasi = tidak apple-to-apple kalau digabung mentah-mentah).
#   Perbandingan akhir yang benar akan dilakukan di 11_compare_all.py dengan
#   membaca metrik ringkasan (RMSE/MAE/Corr/Bias) dari CSV tiap tahap, BUKAN
#   dengan mencocokkan baris data mentah.
#
# ALUR:
# 1. Load data hasil regridding
# 2. Deteksi blok sequence lengkap (10 lead_year berurutan) per sample asli
#    -- dilakukan lewat deteksi "reset" pada nilai lead_year, BUKAN sekadar
#    group-by init_date (karena kemungkinan ada >1 ensemble member per
#    init_date yang sama)
# 3. Split train/test berdasarkan init_date (sama: < 2005 train, >= 2005 test)
# 4. Thinning grid + normalisasi channel
# 5. Latih ConvLSTM (2 layer ConvLSTM2D, many-to-many: prediksi seluruh
#    10 frame sekaligus, bukan cuma frame terakhir)
# 6. Evaluasi (RMSE, MAE, Correlation, Bias) - dibandingkan baseline
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

VARNAME = "pr"
TRAIN_TEST_CUTOFF_YEAR = 2005
SPATIAL_THIN_FACTOR = 4
SEQUENCE_LENGTH = 10  # jumlah lead_year per sequence

CONVLSTM_EPOCHS = 30
CONVLSTM_BATCH_SIZE = 4
CONVLSTM_PATIENCE = 8

# %% [markdown]
# ## 1. Load Data Hasil Regridding

# %%
model_da = xr.open_dataarray(os.path.join(PROCESSED_DIR, f"{VARNAME}_model_regridded.nc"))
obs_da = xr.open_dataarray(os.path.join(PROCESSED_DIR, f"{VARNAME}_obs_matched.nc"))

print(f"Model shape: {dict(model_da.sizes)}")
print(f"Obs shape  : {dict(obs_da.sizes)}")

lead_year_arr = model_da.lead_year.values
init_years_arr = np.array([int(str(x).replace("s", "")) for x in model_da.init_date.values])

# %% [markdown]
# ## 2. Deteksi Blok Sequence Lengkap
#
# Sebuah blok baru dimulai setiap kali lead_year TIDAK naik dibanding
# sebelumnya (turun atau sama) -- ini menangani kasus banyak ensemble member
# dengan init_date yang sama, tanpa perlu kolom "member" eksplisit.

# %%
def find_complete_blocks(init_date_arr, member_arr, lead_year_arr, seq_len):
    complete_blocks = []

    groups = {}

    for i, (init_date, member) in enumerate(
        zip(init_date_arr, member_arr)
    ):
        key = (str(init_date), str(member))
        groups.setdefault(key, []).append(i)

    for (init_date, member), indices in groups.items():
        indices = np.array(indices)

        lead_years = lead_year_arr[indices]

        if (
            len(indices) == seq_len
            and np.array_equal(
                np.sort(lead_years),
                np.arange(1, seq_len + 1)
            )
        ):
            order = np.argsort(lead_years)
            sorted_indices = indices[order]

            init_year = int(init_date.replace("s", ""))

            complete_blocks.append((sorted_indices, init_year))

    return complete_blocks


blocks = find_complete_blocks(
    model_da.init_date.values,
    model_da.member.values,
    lead_year_arr,
    SEQUENCE_LENGTH
)
print(f"\nTotal blok sequence lengkap ({SEQUENCE_LENGTH} lead_year berurutan): {len(blocks)}")

block_init_years = np.array([b[1] for b in blocks])
train_block_idx = np.where(block_init_years < TRAIN_TEST_CUTOFF_YEAR)[0]
test_block_idx = np.where(block_init_years >= TRAIN_TEST_CUTOFF_YEAR)[0]

print(f"Blok training: {len(train_block_idx)} (init-date < {TRAIN_TEST_CUTOFF_YEAR})")
print(f"Blok testing : {len(test_block_idx)} (init-date >= {TRAIN_TEST_CUTOFF_YEAR})")
raise SystemExit("Pemeriksaan sequence selesai. Training belum dijalankan.")
# %% [markdown]
# ## 3. Thinning Grid + Susun Tensor 5D (n_sample, T, lat, lon, channel)
#
# Channel: model_value, lat (broadcast), lon (broadcast), init_year (broadcast
# konstan). lead_year TIDAK perlu jadi channel di sini karena posisinya di
# axis waktu (T) sudah merepresentasikan urutan lead_year secara implisit.

# %%
lat_vals = model_da.lat.values[::SPATIAL_THIN_FACTOR]
lon_vals = model_da.lon.values[::SPATIAL_THIN_FACTOR]
lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)  # (lat_thin, lon_thin)

model_thin = model_da.sel(lat=lat_vals, lon=lon_vals)
obs_thin = obs_da.sel(lat=lat_vals, lon=lon_vals)


def build_sequence_tensor(model_thin, obs_thin, blocks, block_idx):
    X_seq, y_seq = [], []
    model_vals = model_thin.values  # (pair, lat_thin, lon_thin)
    obs_vals = obs_thin.values

    for bi in block_idx:
        idx, init_year = blocks[bi]
        m = model_vals[idx]              # (T, lat_thin, lon_thin)
        o = obs_vals[idx]

        lat_ch = np.broadcast_to(lat_grid, m.shape)
        lon_ch = np.broadcast_to(lon_grid, m.shape)
        iy_ch = np.full_like(m, init_year)

        X_i = np.stack([m, lat_ch, lon_ch, iy_ch], axis=-1)  # (T, lat, lon, 4)
        X_seq.append(X_i)
        y_seq.append(o[..., None])                            # (T, lat, lon, 1)

    return np.array(X_seq, dtype="float32"), np.array(y_seq, dtype="float32")


print("\nMenyusun tensor sequence (thinning factor " + str(SPATIAL_THIN_FACTOR) + ")...")
X_train_raw, y_train_raw = build_sequence_tensor(model_thin, obs_thin, blocks, train_block_idx)
X_test_raw, y_test_raw = build_sequence_tensor(model_thin, obs_thin, blocks, test_block_idx)

print(f"X_train shape: {X_train_raw.shape}  (sample, T, lat, lon, channel)")
print(f"X_test shape : {X_test_raw.shape}")

# %% [markdown]
# ## 4. Normalisasi Channel (statistik dari TRAIN saja)

# %%
# Normalisasi channel menggunakan statistik TRAIN saja
# Gunakan float64 untuk menghindari kesalahan akumulasi presisi

ch_mean = np.nanmean(
    X_train_raw, axis=(0, 1, 2, 3), dtype=np.float64
).astype(np.float32)

ch_std = np.nanstd(
    X_train_raw, axis=(0, 1, 2, 3), dtype=np.float64
).astype(np.float32)

ch_std[ch_std == 0] = 1.0

y_mean = np.nanmean(y_train_raw, dtype=np.float64)
y_std = np.nanstd(y_train_raw, dtype=np.float64)

print("\n=== STATISTIK NORMALISASI CONVLSTM ===")
print("ch_mean:", ch_mean)
print("ch_std:", ch_std)
print("y_mean:", y_mean)
print("y_std:", y_std)

X_train = np.nan_to_num((X_train_raw - ch_mean) / ch_std)
X_test = np.nan_to_num((X_test_raw - ch_mean) / ch_std)
y_train = np.nan_to_num((y_train_raw - y_mean) / y_std)


print("NaN model:",
      np.isnan(X_test_raw).sum())

print("NaN observasi:",
      np.isnan(y_test_raw).sum())

# %% [markdown]
# ## 5. Bangun & Latih ConvLSTM (many-to-many: prediksi seluruh 10 frame)

# %%
def build_convlstm(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.ConvLSTM2D(16, 3, padding="same", return_sequences=True)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(16, 3, padding="same", return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    out = layers.TimeDistributed(layers.Conv2D(1, 1, activation="linear", padding="same"))(x)
    return Model(inp, out)


print("\nMelatih ConvLSTM... (paling berat dari 5 metode, wajar kalau lama)")
convlstm_model = build_convlstm(X_train.shape[1:])
convlstm_model.compile(optimizer="adam", loss="mse")
convlstm_model.summary()

history = convlstm_model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=CONVLSTM_EPOCHS,
    batch_size=CONVLSTM_BATCH_SIZE,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=CONVLSTM_PATIENCE, restore_best_weights=True)],
    verbose=1,
)
print("Selesai training.")

# %% [markdown]
# ## 6. Prediksi & Evaluasi

# %%
pred_scaled = convlstm_model.predict(X_test, batch_size=CONVLSTM_BATCH_SIZE)
pred = pred_scaled[..., 0] * y_std + y_mean   # (n_test, T, lat_thin, lon_thin), skala asli

y_test_arr = y_test_raw[..., 0]
baseline_arr = X_test_raw[..., 0]


def evaluate(y_true, y_pred, name):
    yt, yp = y_true.flatten(), y_pred.flatten()
    mask = ~np.isnan(yt) & ~np.isnan(yp)
    yt, yp = yt[mask], yp[mask]
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    corr = np.corrcoef(yt, yp)[0, 1]
    bias = (yp - yt).mean()
    return {"Metode": name, "RMSE": rmse, "MAE": mae, "Correlation": corr, "Bias": bias}


results = [
    evaluate(y_test_arr, baseline_arr, "Model Mentah (tanpa koreksi)"),
    evaluate(y_test_arr, pred, "ConvLSTM"),
]
results_df = pd.DataFrame(results)

print(f"\n(Catatan: evaluasi ini pakai {X_test_raw.shape[0]} sequence lengkap "
      f"= {X_test_raw.shape[0] * SEQUENCE_LENGTH} pair (di titik grid ter-thin). "
      f"Populasi ini kemungkinan sedikit lebih kecil dari RF/XGBoost/ANN/CNN, "
      f"karena ConvLSTM cuma bisa pakai blok init_date yang 10 lead_year-nya "
      f"lengkap. Ini normal -- lihat catatan di bagian atas script.)")
print("\n=== HASIL EVALUASI ===")
print(results_df.to_string(index=False))

results_df.to_csv(os.path.join(OUTPUT_DIR, f"10_convlstm_evaluation_{VARNAME}.csv"), index=False)

# %% [markdown]
# ## 7. Visualisasi

# %%
plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE, skala ter-standarisasi)")
plt.title(f"Kurva Loss Training - {VARNAME.upper()}")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, f"10_convlstm_loss_curve_{VARNAME}.png"), dpi=100)
plt.close()

yt_flat, yp_flat, yb_flat = y_test_arr.flatten(), pred.flatten(), baseline_arr.flatten()
valid = (
    np.isfinite(yt_flat)
    & np.isfinite(yp_flat)
    & np.isfinite(yb_flat)
)
yt_flat, yp_flat, yb_flat = yt_flat[valid], yp_flat[valid], yb_flat[valid]

plt.figure(figsize=(6, 6))
sample_idx = np.random.choice(len(yt_flat), size=min(5000, len(yt_flat)), replace=False)
plt.scatter(yt_flat[sample_idx], yp_flat[sample_idx], alpha=0.3, s=5, label="ConvLSTM", color="teal")
plt.scatter(yt_flat[sample_idx], yb_flat[sample_idx], alpha=0.3, s=5, label="Model Mentah", color="orange")
lims = [min(yt_flat.min(), yp_flat.min()), max(yt_flat.max(), yp_flat.max())]
plt.plot(lims, lims, 'k--', label="Garis ideal (y=x)")
plt.xlabel("Observasi (°C)")
plt.ylabel("Prediksi (°C)")
plt.title(f"ConvLSTM vs Model Mentah - {VARNAME.upper()}")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, f"10_convlstm_scatter_{VARNAME}.png"), dpi=100)
plt.close()

# RMSE per lead_year -- khas ConvLSTM, lihat apakah akurasi menurun makin jauh horizon prediksi
rmse_per_lead = []
for t in range(SEQUENCE_LENGTH):
    yt_t = y_test_arr[:, t].flatten()
    yp_t = pred[:, t].flatten()
    m = ~np.isnan(yt_t) & ~np.isnan(yp_t)
    rmse_per_lead.append(np.sqrt(mean_squared_error(yt_t[m], yp_t[m])))

plt.figure(figsize=(6, 4))
plt.plot(range(1, SEQUENCE_LENGTH + 1), rmse_per_lead, marker="o")
plt.xlabel("Lead Year")
plt.ylabel("RMSE (°C)")
plt.title(f"RMSE ConvLSTM per Lead Year - {VARNAME.upper()}")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, f"10_convlstm_rmse_per_leadyear_{VARNAME}.png"), dpi=100)
plt.close()

print(f"\nGambar tersimpan: outputs/10_convlstm_loss_curve_{VARNAME}.png, "
      f"10_convlstm_scatter_{VARNAME}.png, 10_convlstm_rmse_per_leadyear_{VARNAME}.png")

# %% [markdown]
# ## 8. Simpan Model & Prediksi

# %%
convlstm_model.save(os.path.join(MODELS_DIR, f"convlstm_{VARNAME}.h5"))
np.savez(
    os.path.join(MODELS_DIR, f"convlstm_{VARNAME}_norm_stats.npz"),
    ch_mean=ch_mean, ch_std=ch_std, y_mean=y_mean, y_std=y_std,
)
print(f"Model tersimpan: models/convlstm_{VARNAME}.h5 (+ norm_stats.npz)")

np.savez(
    os.path.join(PROCESSED_DIR, f"convlstm_{VARNAME}_predictions.npz"),
    y_test=yt_flat, y_pred=yp_flat, y_baseline=yb_flat,
)
print(f"Prediksi tersimpan: data/processed/convlstm_{VARNAME}_predictions.npz")

print("\n" + "=" * 60)
print(f"SELESAI ConvLSTM untuk {VARNAME}.")
print(f"SEMUA 5 METODE ML SUDAH DILATIH untuk {VARNAME}.")
print("=" * 60)

print("NaN model:",
      np.isnan(X_test_raw).sum())

print("NaN observasi:",
      np.isnan(y_test_raw).sum())