# %% [markdown]
# # 11 - Perbandingan Metode ML (dikelompokkan sesuai arahan mentor)
#
# Kelompok A - Tabular (per titik)  : Random Forest, XGBoost, ANN
# Kelompok B - Spasial (grid/urutan): CNN, ConvLSTM
#
# Aturan pembacaan:
# - Perbandingan yang ADIL dilakukan DI DALAM tiap kelompok.
# - Perbandingan ANTAR kelompok = insight tambahan, bukan klaim "menang telak",
#   karena kelompok B menerima informasi spasial yang tidak dimiliki kelompok A.
#
# Script ini hanya MEMBACA file *_predictions.npz yang sudah dihasilkan
# script training (06-10). Tidak ada training ulang di sini.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VARNAME = "tas"   # ganti ke "pr" untuk curah hujan
UNIT = "°C" if VARNAME == "tas" else "mm/hari"

# key = awalan nama file: {key}_{VARNAME}_predictions.npz
METHODS = [
    dict(key="rf", label="Random Forest", group="A", color="#1f77b4", marker="o"),
    dict(key="xgb", label="XGBoost", group="A", color="#ff7f0e", marker="s"),
    dict(key="ann", label="ANN", group="A", color="#2ca02c", marker="^"),
    dict(key="cnn", label="CNN (grid ter-thin)", group="B", color="#d62728", marker="D"),
    dict(key="cnn_fullgrid", label="CNN (grid penuh)", group="B", color="#8c564b", marker="d"),
    dict(key="convlstm", label="ConvLSTM", group="B", color="#9467bd", marker="P"),
]
GROUP_NAME = {
    "A": "Kelompok A - Tabular (per titik)",
    "B": "Kelompok B - Spasial (grid / urutan)",
}

# %% [markdown]
# ## 1. Fungsi bantu

# %%
def load_npz(key):
    if key == "cnn_fullgrid":
        filename = f"cnn_{VARNAME}_predictions_fullgrid.npz"
    else:
        filename = f"{key}_{VARNAME}_predictions.npz"

    path = os.path.join(PROCESSED_DIR, filename)

    if not os.path.exists(path):
        return None

    d = np.load(path)

    return (
        d["y_test"].astype("float64").ravel(),
        d["y_pred"].astype("float64").ravel(),
        d["y_baseline"].astype("float64").ravel()
    )

def compute_metrics(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[m], y_pred[m]
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae = float(np.mean(np.abs(yp - yt)))
    corr = float(np.corrcoef(yt, yp)[0, 1])
    bias = float(np.mean(yp - yt))
    std_ratio = float(np.std(yp) / np.std(yt))
    # RMSE terpusat (tanpa bias), dinormalisasi -> ini yang jadi kontur di Taylor diagram
    crmse_norm = float(np.sqrt(max(1 + std_ratio**2 - 2 * std_ratio * corr, 0)))
    return dict(RMSE=rmse, MAE=mae, Correlation=corr, Bias=bias,
                Std_ratio=std_ratio, cRMSE_norm=crmse_norm, n=int(m.sum()))

# %% [markdown]
# ## 2. Muat semua metode + cek apakah populasi data uji sama

# %%
loaded = {}
for cfg in METHODS:
    res = load_npz(cfg["key"])
    if res is None:
        print(f"[SKIP] {cfg['label']}: file {cfg['key']}_{VARNAME}_predictions.npz belum ada")
    else:
        loaded[cfg["key"]] = res

if not loaded:
    raise SystemExit("Tidak ada file prediksi ditemukan di data/processed/")

ref_key = next(iter(loaded))
ref_y = loaded[ref_key][0]
y_base = loaded[ref_key][2]

rows = []
# baris baseline (model mentah)
base_m = compute_metrics(ref_y, y_base)
rows.append(dict(Metode="Model mentah (tanpa koreksi)", Kelompok="-", **base_m, Populasi_sama="-"))

for cfg in METHODS:
    if cfg["key"] not in loaded:
        continue
    y_true, y_pred, _ = loaded[cfg["key"]]
    same = (len(y_true) == len(ref_y)) and np.allclose(y_true, ref_y, atol=1e-4)
    if not same:
        print(f"[PERINGATAN] {cfg['label']}: data uji TIDAK identik dengan {ref_key} "
              f"({len(y_true):,} vs {len(ref_y):,} titik). Metrik tetap dihitung, tapi "
              f"perbandingan dengan metode lain tidak 100% apple-to-apple.")
    m = compute_metrics(y_true, y_pred)
    rows.append(dict(Metode=cfg["label"], Kelompok=cfg["group"], **m,
                     Populasi_sama="ya" if same else "TIDAK"))

df = pd.DataFrame(rows)
df["Turun_RMSE_%"] = 100 * (1 - df["RMSE"] / base_m["RMSE"])
df.loc[df["Kelompok"] == "-", "Turun_RMSE_%"] = np.nan

show_cols = ["Metode", "RMSE", "MAE", "Correlation", "Bias", "Turun_RMSE_%", "Populasi_sama"]
fmt = lambda d: d[show_cols].round(3).to_string(index=False)

# %% [markdown]
# ## 3. Tabel per kelompok (perbandingan ADIL) + lintas kelompok (insight tambahan)

# %%
print(f"\n=== VARIABEL {VARNAME.upper()} ({UNIT}) ===")
print(f"Baseline model mentah: RMSE={base_m['RMSE']:.3f}, Korelasi={base_m['Correlation']:.3f}")

for g in ["A", "B"]:
    sub = df[df["Kelompok"] == g].sort_values("RMSE")
    print(f"\n--- {GROUP_NAME[g]} (perbandingan adil di dalam kelompok) ---")
    if sub.empty:
        print("  (belum ada metode di kelompok ini)")
        continue
    print(fmt(sub))
    best_rmse = sub.iloc[0]["Metode"]
    best_bias = sub.loc[sub["Bias"].abs().idxmin(), "Metode"]
    print(f"  RMSE terendah : {best_rmse}")
    print(f"  |Bias| terkecil: {best_bias}")

print("\n--- LINTAS KELOMPOK (insight tambahan, BUKAN klaim menang telak) ---")
print("Catatan: kelompok B menerima konteks spasial; kelompok A hanya nilai per titik.")
cross = df[df["Kelompok"] != "-"].sort_values("RMSE")
print(fmt(cross))

df.to_csv(os.path.join(OUTPUT_DIR, f"11_compare_{VARNAME}.csv"), index=False)
print(f"\nTabel tersimpan: outputs/11_compare_{VARNAME}.csv")

# %% [markdown]
# ## 4. Taylor diagram (korelasi + rasio std + RMSE terpusat dalam 1 gambar)

# %%
def draw_taylor(ax, cfgs, title, max_r):
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]
    ax.set_thetagrids(np.degrees(np.arccos(ticks)), labels=[str(t) for t in ticks])
    ax.set_ylim(0, max_r)
    ax.set_rlabel_position(0)

    # kontur RMSE terpusat: lingkaran berpusat di titik REF (r=1, theta=0)
    phi = np.linspace(0, np.pi, 400)
    for rho in [0.25, 0.5, 0.75, 1.0]:
        x, y = 1 + rho * np.cos(phi), rho * np.sin(phi)
        r, th = np.hypot(x, y), np.arctan2(y, x)
        keep = (th >= 0) & (th <= np.pi / 2) & (r <= max_r)
        ax.plot(th[keep], r[keep], color="gray", ls=":", lw=0.8)
    # busur std = 1
    t = np.linspace(0, np.pi / 2, 100)
    ax.plot(t, np.ones_like(t), color="gray", ls="--", lw=0.8)

    ax.plot(0, 1, marker="*", color="black", markersize=16, ls="", label="Observasi (REF)", zorder=5)
    for c in cfgs:
        r_ = df[df["Metode"] == c["label"]]
        if r_.empty:
            continue
        corr, sr = float(r_["Correlation"].iloc[0]), float(r_["Std_ratio"].iloc[0])
        ax.plot(np.arccos(np.clip(corr, 0, 1)), sr, marker=c["marker"], color=c["color"],
                markersize=11, ls="", label=c["label"], zorder=4)
    ax.plot(np.arccos(np.clip(base_m["Correlation"], 0, 1)), base_m["Std_ratio"], marker="x",
            color="gray", markersize=10, mew=2, ls="", label="Model mentah", zorder=4)
    ax.set_title(title, fontsize=10, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=8)


present = [c for c in METHODS if c["key"] in loaded]
max_std = max([base_m["Std_ratio"]] + [float(df[df["Metode"] == c["label"]]["Std_ratio"].iloc[0]) for c in present])
max_r = max(1.5, 1.15 * max_std)

fig = plt.figure(figsize=(18, 6.5))
groups = [("A", "Kelompok A - Tabular"), ("B", "Kelompok B - Spasial"), (None, "Semua metode (insight tambahan)")]
for i, (g, ttl) in enumerate(groups, start=1):
    ax = fig.add_subplot(1, 3, i, polar=True)
    cfgs = [c for c in present if (g is None or c["group"] == g)]
    draw_taylor(ax, cfgs, f"{ttl}\n{VARNAME.upper()}", max_r)

fig.suptitle("Taylor diagram - makin dekat ke bintang (REF) makin baik. "
             "Garis titik-titik = RMSE terpusat (ternormalisasi)", fontsize=11)
plt.tight_layout()
out_png = os.path.join(OUTPUT_DIR, f"11_taylor_{VARNAME}.png")
plt.savefig(out_png, dpi=110, bbox_inches="tight")
plt.close()
print(f"Taylor diagram tersimpan: {out_png}")
print("\nSELESAI perbandingan.")