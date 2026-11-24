
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

for var in ["tas", "pr"]:
    print(f"\n{'='*60}")
    print(f"AUDIT POPULASI PENGUJIAN: {var.upper()}")
    print("="*60)

    methods = ["rf", "xgb", "ann", "cnn", "convlstm"]
    results = {}

    for method in methods:
        path = os.path.join(
            PROCESSED_DIR,
            f"{method}_{var}_predictions.npz"
        )

        if not os.path.exists(path):
            print(f"[TIDAK ADA] {method}")
            continue

        data = np.load(path)
        yt = data["y_test"].astype("float64").ravel()
        yp = data["y_pred"].astype("float64").ravel()
        yb = data["y_baseline"].astype("float64").ravel()

        results[method] = (yt, yp, yb)

        print(f"\n{method.upper()}")
        print("Jumlah elemen :", len(yt))
        print("NaN y_test    :", np.isnan(yt).sum())
        print("NaN y_pred    :", np.isnan(yp).sum())
        print("NaN baseline  :", np.isnan(yb).sum())
        print("Data valid    :", np.isfinite(yt).sum())

        if len(yt) != len(yp) or len(yt) != len(yb):
            print("[PERINGATAN] Panjang array tidak sama!")

    if "rf" not in results:
        continue

    ref = results["rf"][0]

    print("\n--- PERBANDINGAN DENGAN RF ---")

    for method, (yt, yp, yb) in results.items():
        if len(yt) != len(ref):
            print(f"{method.upper()}: PANJANG BERBEDA")
            continue

        same_values = np.allclose(
            yt, ref, atol=1e-4, rtol=0, equal_nan=True
        )

        same_mask = np.array_equal(
            np.isfinite(yt), np.isfinite(ref)
        )

        print(
            f"{method.upper()}: "
            f"nilai y_test sama={same_values}, "
            f"mask NaN sama={same_mask}"
        )
