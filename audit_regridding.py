
import os
import numpy as np
import xarray as xr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

files = {
    "TAS Model": "tas_model_regridded.nc",
    "TAS Observasi ERA5": "tas_obs_matched.nc",
    "PR Model": "pr_model_regridded.nc",
    "PR Observasi CHIRPS": "pr_obs_matched.nc",
}

for name, filename in files.items():
    path = os.path.join(DATA_DIR, filename)

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    with xr.open_dataarray(path) as da:
        values = da.values

        print("Dimensi:", da.dims)
        print("Shape:", da.shape)
        print("NaN:", int(np.isnan(values).sum()))
        print("Inf:", int(np.isinf(values).sum()))
        print("Minimum:", float(np.nanmin(values)))
        print("Maximum:", float(np.nanmax(values)))
        print("Koordinat:", list(da.coords))

        if "pair" in da.coords:
            print("Jumlah pasangan:", da.sizes["pair"])

        if "init_date" in da.coords:
            dates= sorted(
                np.unique(da.init_date.values.astype(str))
            )
            print("rentang inisialisasi:",
                  dates[0], "sampai", dates[-1])
        if "lead_year" in da.coords:
            print("Lead year:",
                  np.unique(da.lead_year.values))

        if "member" in da.coords:
            print("Ensemble member:",
                  np.unique(da.member.values))

        assert np.isfinite(values).all(), (
            f"Masih ada NaN atau Inf pada {name}"
        )

print("\nAUDIT SELESAI.")
