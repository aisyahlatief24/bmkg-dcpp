
import os
import numpy as np
import xarray as xr
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_DIR = os.path.join(
    BASE_DIR, "data", "processed", "forecast_s2020"
)

FILE = os.path.join(
    FORECAST_DIR, "rf_corrected",
    "pr_rf_corrected_dcppB_s2020.nc"
)

ds = xr.open_dataset(FILE)
da = ds["pr_rf_corrected"]

# Pastikan dimensi member tersedia
if "sample" in da.dims:
    da = da.assign_coords(
        member=("sample", da.member.values)
    ).swap_dims({"sample": "member"})

members = da.member.values
rows = []

for lead in da.lead_year.values:
    data = da.sel(lead_year=lead)

    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a = data.sel(member=members[i]).values
            b = data.sel(member=members[j]).values

            diff = a - b

            rows.append({
                "lead_year": int(lead),
                "member_1": str(members[i]),
                "member_2": str(members[j]),
                "mean_abs_difference": np.nanmean(np.abs(diff)),
                "max_abs_difference": np.nanmax(np.abs(diff)),
                "correlation": np.corrcoef(
                    a.ravel(), b.ravel()
                )[0, 1]
            })

result = pd.DataFrame(rows)

print("\n=== PERBEDAAN ANTAR-MEMBER SETELAH RF ===")
print(
    result.groupby("lead_year")[
        ["mean_abs_difference", "max_abs_difference", "correlation"]
    ].mean().to_string(float_format="%.5f")
)

output = os.path.join(
    BASE_DIR, "outputs", "pr_rf_member_difference.csv"
)
result.to_csv(output, index=False)

print(f"\nCSV tersimpan: {output}")

ds.close()
