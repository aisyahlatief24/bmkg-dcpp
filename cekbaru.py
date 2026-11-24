import xarray as xr
import numpy as np

paths = {
    "obs": "data/processed/pr_obs_matched.nc",
    "raw": "data/processed/forecast_s2020/pr_dcppB_forecast_s2020_regridded_to_chirps025.nc",
    "corrected": "data/processed/forecast_s2020/rf_corrected/pr_rf_corrected_dcppB_s2020.nc",
}

datasets = {
    name: xr.open_dataset(path)
    for name, path in paths.items()
}

ref = datasets["obs"]

for name in ["raw", "corrected"]:
    ds = datasets[name]

    print(f"\nOBS vs {name.upper()}")
    print("Latitude identik:",
          np.array_equal(ref.lat.values, ds.lat.values))
    print("Longitude identik:",
          np.array_equal(ref.lon.values, ds.lon.values))

for ds in datasets.values():
    ds.close()