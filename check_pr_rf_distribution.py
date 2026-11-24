
import os
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_DIR = os.path.join(
    BASE_DIR, "data", "processed", "forecast_s2020"
)

raw_path = os.path.join(
    FORECAST_DIR,
    "pr_dcppB_forecast_s2020_regridded_to_chirps025.nc"
)

rf_path = os.path.join(
    FORECAST_DIR,
    "rf_corrected",
    "pr_rf_corrected_dcppB_s2020.nc"
)

raw_ds = xr.open_dataset(raw_path)
rf_ds = xr.open_dataset(rf_path)

raw = raw_ds["pr"]
rf = rf_ds["pr_rf_corrected"]

# Pastikan koordinat dan dimensi sama
xr.align(raw, rf, join="exact")

thresholds = [1, 2, 5, 10, 15, 20]
rows = []

for lead in raw.lead_year.values:

    raw_values = raw.sel(lead_year=lead).values.flatten()
    rf_values = rf.sel(lead_year=lead).values.flatten()

    raw_values = raw_values[np.isfinite(raw_values)]
    rf_values = rf_values[np.isfinite(rf_values)]

    row = {
        "lead_year": int(lead),

        "raw_min": np.min(raw_values),
        "rf_min": np.min(rf_values),

        "raw_mean": np.mean(raw_values),
        "rf_mean": np.mean(rf_values),

        "raw_median": np.median(raw_values),
        "rf_median": np.median(rf_values),

        "raw_p01": np.percentile(raw_values, 1),
        "rf_p01": np.percentile(rf_values, 1),

        "raw_p95": np.percentile(raw_values, 95),
        "rf_p95": np.percentile(rf_values, 95),

        "raw_p99": np.percentile(raw_values, 99),
        "rf_p99": np.percentile(rf_values, 99),

        "raw_max": np.max(raw_values),
        "rf_max": np.max(rf_values),
    }

    for threshold in thresholds:
        row[f"raw_below_{threshold}"] = (
            np.mean(raw_values < threshold) * 100
        )
        row[f"rf_below_{threshold}"] = (
            np.mean(rf_values < threshold) * 100
        )

    rows.append(row)

result = pd.DataFrame(rows)

output_dir = os.path.join(BASE_DIR, "outputs")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir, "pr_rf_distribution_by_leadyear.csv"
)

result.to_csv(output_path, index=False)

print("\n=== DISTRIBUSI PR: RAW VS RANDOM FOREST ===")

columns = [
    "lead_year",
    "raw_min", "rf_min",
    "raw_mean", "rf_mean",
    "raw_median", "rf_median",
    "raw_p01", "rf_p01",
    "raw_p99", "rf_p99",
    "raw_max", "rf_max"
]

print(
    result[columns].to_string(
        index=False,
        float_format="%.4f"
    )
)

print("\n=== PROPORSI NILAI DI BAWAH AMBANG (%) ===")

for threshold in thresholds:
    print(f"\nAmbang < {threshold} mm/hari")

    print(
        result[
            [
                "lead_year",
                f"raw_below_{threshold}",
                f"rf_below_{threshold}"
            ]
        ].to_string(
            index=False,
            float_format="%.2f"
        )
    )

print(f"\nCSV tersimpan: {output_path}")

raw_ds.close()
rf_ds.close()
