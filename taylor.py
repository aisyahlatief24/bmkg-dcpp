# %% [markdown]
# # 11 - Taylor Diagram: Perbandingan ML dan DL
#
# Perbandingan dilakukan dalam 2 kelompok:
#
# 1. ML (tabular)
#    - Random Forest
#    - XGBoost
#    - ANN
#
# 2. DL (spasial)
#    - CNN (U-Net)
#    - ConvLSTM
#
# Taylor diagram menampilkan:
#   - Korelasi
#   - Standar deviasi prediksi yang dinormalisasi
#   - Kedekatan statistik model terhadap observasi
#
# Script ini TIDAK melakukan training ulang.
# Hanya membaca file prediksi hasil training dari tahap sebelumnya.
#
# Output:
#   outputs/11_taylor_ML_{VARNAME}.png
#   outputs/11_taylor_DL_{VARNAME}.png
#
# Ganti VARNAME menjadi "pr" untuk membuat diagram curah hujan.


# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as FA
import mpl_toolkits.axisartist.grid_finder as GF


# ============================================================
# 1. KONFIGURASI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pilih variabel:
# "tas" = temperatur
# "pr"  = curah hujan
VARNAME = "pr"


# ============================================================
# 2. KELOMPOK METODE
# ============================================================

METHOD_GROUPS = {

    # -------------------------
    # MACHINE LEARNING
    # -------------------------
    "Random Forest": {
        "file": f"rf_{VARNAME}_predictions.npz",
        "group": "ML",
        "marker": "o",
        "color": "#1C7293",
    },

    "XGBoost": {
        "file": f"xgb_{VARNAME}_predictions.npz",
        "group": "ML",
        "marker": "s",
        "color": "#3D8C99",
    },

    "ANN": {
        "file": f"ann_{VARNAME}_predictions.npz",
        "group": "ML",
        "marker": "^",
        "color": "#065A82",
    },

    # -------------------------
    # DEEP LEARNING
    # -------------------------
    "CNN (U-Net)": {
        "file": f"cnn_{VARNAME}_predictions.npz",
        "group": "DL",
        "marker": "D",
        "color": "#F2994A",
    },

    "ConvLSTM": {
        "file": f"convlstm_{VARNAME}_predictions.npz",
        "group": "DL",
        "marker": "P",
        "color": "#C0392B",
    },
}


# ============================================================
# 3. KELAS TAYLOR DIAGRAM
# ============================================================

class TaylorDiagram:

    def __init__(self, ref_std=1.0, fig=None, rect=111,
                 label="Observasi (REF)"):

        self.ref_std = ref_std

        # Transform polar
        tr = PolarAxes.PolarTransform()

        # Korelasi yang ditampilkan pada sumbu
        corr_ticks = np.array([
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            0.9,
            0.95,
            0.99,
            1.0
        ])

        angles = np.arccos(corr_ticks)

        grid_loc = GF.FixedLocator(angles.tolist())

        tick_fmt = GF.DictFormatter(
            dict(
                zip(
                    angles.tolist(),
                    [str(c) for c in corr_ticks]
                )
            )
        )

        # Batas radial
        r_max = 1.6 * ref_std

        grid_helper = FA.GridHelperCurveLinear(
            tr,
            extremes=(
                0,
                np.pi / 2,
                0,
                r_max
            ),
            grid_locator1=grid_loc,
            tick_formatter1=tick_fmt,
        )

        ax = FA.FloatingSubplot(
            fig,
            rect,
            grid_helper=grid_helper
        )

        fig.add_subplot(ax)

        # -------------------------
        # Pengaturan sumbu
        # -------------------------

        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(
            ticklabels=True,
            label=True
        )
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Korelasi")

        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text(
            "Standar Deviasi (dinormalisasi)"
        )

        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")

        ax.axis["bottom"].set_visible(False)

        self._ax = ax
        self.ax = ax.get_aux_axes(tr)

        # -------------------------
        # Titik observasi
        # -------------------------

        l, = self.ax.plot(
            [0],
            ref_std,
            "k*",
            ms=16,
            label=label,
            zorder=10
        )

        self.sample_points = [l]

        # -------------------------
        # Referensi std = 1
        # -------------------------

        t = np.linspace(
            0,
            np.pi / 2,
            100
        )

        r = np.ones_like(t) * ref_std

        self.ax.plot(
            t,
            r,
            "k--",
            alpha=0.4,
            linewidth=1
        )

    def add_sample(
        self,
        std,
        corr,
        *args,
        **kwargs
    ):

        # Korelasi -> sudut
        theta = np.arccos(
            np.clip(corr, -1, 1)
        )

        l, = self.ax.plot(
            theta,
            std,
            *args,
            **kwargs
        )

        self.sample_points.append(l)

        return l

    def add_grid(self):
        self._ax.grid(
            True,
            alpha=0.3
        )


# ============================================================
# 4. LOAD PREDIKSI & HITUNG STATISTIK
# ============================================================

stats = {}

for name, cfg in METHOD_GROUPS.items():

    path = os.path.join(
        PROCESSED_DIR,
        cfg["file"]
    )

    if not os.path.exists(path):

        print(
            f"Lewati {name}: "
            f"file {cfg['file']} tidak ditemukan."
        )

        continue

    data = np.load(path)

    y_true = data["y_test"].flatten()
    y_pred = data["y_pred"].flatten()

    # Buang NaN
    valid = (
        ~np.isnan(y_true)
        &
        ~np.isnan(y_pred)
    )

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    # Standar deviasi observasi
    obs_std = np.std(y_true)

    # Standar deviasi prediksi
    pred_std = np.std(y_pred)

    # Normalisasi terhadap std observasi
    std_norm = pred_std / obs_std

    # Korelasi
    corr = np.corrcoef(
        y_true,
        y_pred
    )[0, 1]

    stats[name] = {
        "std_norm": std_norm,
        "corr": corr,
        "group": cfg["group"],
        "marker": cfg["marker"],
        "color": cfg["color"],
        "n": len(y_true),
    }

    print(
        f"{name:15s} | "
        f"n={len(y_true):>8,} | "
        f"std_norm={std_norm:.3f} | "
        f"corr={corr:.3f}"
    )


# ============================================================
# 5. FUNGSI MEMBUAT TAYLOR DIAGRAM
# ============================================================

def create_taylor_diagram(
    selected_methods,
    group_name
):

    # --------------------------------
    # Buat figure
    # --------------------------------

    fig = plt.figure(
        figsize=(8, 7)
    )

    dia = TaylorDiagram(
        ref_std=1.0,
        fig=fig,
        label="Observasi (REF)"
    )

    dia.add_grid()

    # --------------------------------
    # Tambahkan setiap metode
    # --------------------------------

    for name in selected_methods:

        if name not in stats:
            continue

        s = stats[name]

        label = name

        dia.add_sample(
            s["std_norm"],
            s["corr"],
            marker=s["marker"],
            markersize=11,
            color=s["color"],
            linestyle="none",
            label=label,
            markeredgecolor="black",
            markeredgewidth=0.6,
        )

    # --------------------------------
    # Legend
    # --------------------------------

    fig.legend(
        dia.sample_points,
        [
            l.get_label()
            for l in dia.sample_points
        ],
        numpoints=1,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=9,
        frameon=True,
    )

    # --------------------------------
    # Judul
    # --------------------------------

    plt.suptitle(
        f"Taylor Diagram — "
        f"Perbandingan {group_name} "
        f"({VARNAME.upper()})",
        fontsize=13,
        y=0.98
    )

    # --------------------------------
    # Keterangan
    # --------------------------------

    if group_name == "ML":

        note = (
            "Perbandingan metode Machine Learning "
            "(Random Forest, XGBoost, dan ANN)."
        )

    else:

        note = (
            "Perbandingan metode Deep Learning "
            "(CNN U-Net dan ConvLSTM)."
        )

    plt.figtext(
        0.5,
        0.01,
        "Titik yang semakin dekat ke "
        "bintang Observasi (REF) menunjukkan "
        "karakteristik statistik yang semakin mendekati observasi.\n"
        + note,
        ha="center",
        fontsize=8.5,
        color="dimgray"
    )

    # --------------------------------
    # Simpan
    # --------------------------------

    output_name = (
        f"11_taylor_"
        f"{group_name}_"
        f"{VARNAME}.png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Taylor diagram {group_name} "
        f"tersimpan: outputs/{output_name}"
    )


# ============================================================
# 6. PERBANDINGAN ML
# ============================================================

ML_METHODS = [
    "Random Forest",
    "XGBoost",
    "ANN"
]

create_taylor_diagram(
    ML_METHODS,
    "ML"
)


# ============================================================
# 7. PERBANDINGAN DL
# ============================================================

DL_METHODS = [
    "CNN (U-Net)",
    "ConvLSTM"
]

create_taylor_diagram(
    DL_METHODS,
    "DL"
)


# ============================================================
# 8. SELESAI
# ============================================================

print("\n" + "=" * 60)
print(
    f"SELESAI Taylor Diagram "
    f"untuk {VARNAME.upper()}"
)
print("=" * 60)

print(
    "\nOutput:"
)

print(
    f"1. outputs/11_taylor_ML_{VARNAME}.png"
)

print(
    f"2. outputs/11_taylor_DL_{VARNAME}.png"
)

print(
    "\nUntuk membuat versi PR, "
    "ubah VARNAME = 'tas' menjadi VARNAME = 'pr'."
)