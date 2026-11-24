
import os
import xarray as xr

base_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(
    base_dir, "data", "processed", "forecast_s2020"
)

files = {
    "TAS": os.path.join(
        folder, "tas_dcppB_forecast_s2020_indonesia_alllead.nc"
    ),
    "PR": os.path.join(
        folder, "pr_dcppB_forecast_s2020_indonesia_alllead.nc"
    ),
}

for name, path in files.items():
    print(f"\n=== {name} ===")

    if not os.path.exists(path):
        print("FILE TIDAK DITEMUKAN:", path)
        continue

    with xr.open_dataset(path) as ds:
        print("File:", os.path.basename(path))
        print("Dimensi:", dict(ds.sizes))
        print("Variabel:", list(ds.data_vars))

        var = name.lower()
        print("Satuan:", ds[var].attrs.get("units"))
        print("Ada missing value:", bool(ds[var].isnull().any().compute().item()))
        print("Nilai minimum:", float(ds[var].min().compute()))
        print("Nilai maksimum:", float(ds[var].max().compute()))

print("\nPengecekan selesai.")
