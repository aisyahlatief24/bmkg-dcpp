
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

corrected_path = os.path.join(
    FORECAST_DIR, "rf_corrected",
    "pr_rf_corrected_dcppB_s2020.nc"
)

raw_ds = xr.open_dataset(raw_path)
corrected_ds = xr.open_dataset(corrected_path)

raw = raw_ds["pr"]
corrected = corrected_ds["pr_rf_corrected"]

# Samakan nama dimensi untuk memudahkan perhitungan
raw = raw.assign_coords(
    member=("sample", raw.member.values)
).swap_dims({"sample": "member"})

corrected = corrected.assign_coords(
    member=("sample", corrected.member.values)
).swap_dims({"sample": "member"})

# Selisih hasil koreksi terhadap data mentah
correction = corrected - raw

rows = []

for lead in raw.lead_year.values:
    raw_lead = raw.sel(lead_year=lead)
    corrected_lead = corrected.sel(lead_year=lead)
    correction_lead = correction.sel(lead_year=lead)

    # Variasi antar-member pada setiap grid
    raw_spread = raw_lead.std("member")
    corrected_spread = corrected_lead.std("member")

    rows.append({
        "lead_year": int(lead),
        "raw_mean": float(raw_lead.mean()),
        "corrected_mean": float(corrected_lead.mean()),
        "raw_ensemble_spread": float(raw_spread.mean()),
        "corrected_ensemble_spread": float(
            corrected_spread.mean()
        ),
        "spread_ratio": float(
            corrected_spread.mean() / raw_spread.mean()
        ),
        "mean_correction": float(correction_lead.mean()),
        "correction_min": float(correction_lead.min()),
        "correction_max": float(correction_lead.max())
    })

result = pd.DataFrame(rows)

print("\n=== ENSEMBLE SPREAD PR: RAW VS RF ===")
print(result.to_string(index=False, float_format="%.4f"))

output = os.path.join(
    BASE_DIR, "outputs", "pr_rf_ensemble_spread.csv"
)

result.to_csv(output, index=False)

print(f"\nCSV tersimpan: {output}")

raw_ds.close()
corrected_ds.close()
