
import os
import xarray as xr

# ==================================================
# 1. PENGATURAN FOLDER
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FORECAST_DIR = os.path.join(PROCESSED_DIR, "forecast_s2020")

# File forecast hasil preprocessing
TAS_INPUT = os.path.join(
    FORECAST_DIR,
    "tas_dcppB_forecast_s2020_indonesia_alllead.nc",
)
PR_INPUT = os.path.join(
    FORECAST_DIR,
    "pr_dcppB_forecast_s2020_indonesia_alllead.nc",
)

# File target observasi
ERA5_PATH = os.path.join(
    RAW_DIR, "era5", "era5_tas_indonesia_1982_2020.nc"
)
CHIRPS_PATH = os.path.join(
    RAW_DIR, "chirps", "chirps_monthly_indonesia_1982_2020.nc"
)

# ==================================================
# 2. LOAD GRID TARGET
# ==================================================
print("Memuat grid target observasi...")

era5_ds = xr.open_dataset(ERA5_PATH)
era5_tas = era5_ds["t2m"].rename({
    "valid_time": "time",
    "latitude": "lat",
    "longitude": "lon",
})

target_lat_tas = era5_tas.lat
target_lon_tas = era5_tas.lon

chirps_ds = xr.open_dataset(CHIRPS_PATH)
chirps_pr = chirps_ds["precip"].rename({
    "latitude": "lat",
    "longitude": "lon",
})

# Harus sama dengan pengaturan saat regridding hindcast
chirps_pr = chirps_pr.coarsen(
    lat=5, lon=5, boundary="trim"
).mean()

target_lat_pr = chirps_pr.lat
target_lon_pr = chirps_pr.lon

print(
    "Grid ERA5:", target_lat_tas.size,
    "lat x", target_lon_tas.size, "lon"
)
print(
    "Grid CHIRPS:", target_lat_pr.size,
    "lat x", target_lon_pr.size, "lon"
)

# ==================================================
# 3. FUNGSI REGRID
# ==================================================
def regrid_forecast(input_path, output_path, variable,
                   target_lat, target_lon):

    print(f"\nMemproses {variable.upper()}...")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"File forecast tidak ditemukan: {input_path}"
        )

    with xr.open_dataarray(input_path) as source:
        source = source.load()

    print("Dimensi awal:", dict(source.sizes))

    # Pastikan koordinat terurut untuk interpolasi
    source = source.sortby("lat").sortby("lon")

    result = source.interp(
        lat=target_lat,
        lon=target_lon,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )

    result.attrs = source.attrs.copy()
    result.attrs["regridding_method"] = "linear interpolation"
    result.attrs["description"] = (
        "DCPP-B forecast s2020 regridded to observational grid. "
        "No observational time matching performed."
    )

    result.name = variable
    result.to_netcdf(output_path)

    print("Dimensi akhir:", dict(result.sizes))
    print("Satuan:", result.attrs.get("units"))
    print(
        "Missing value:",
        bool(result.isnull().any().compute().item())
    )
    print("Tersimpan:", output_path)

    return result


# ==================================================
# 4. REGRID TAS DAN PR
# ==================================================
tas_output = os.path.join(
    FORECAST_DIR,
    "tas_dcppB_forecast_s2020_regridded_to_era5.nc",
)

pr_output = os.path.join(
    FORECAST_DIR,
    "pr_dcppB_forecast_s2020_regridded_to_chirps025.nc",
)

tas_result = regrid_forecast(
    TAS_INPUT, tas_output, "tas",
    target_lat_tas, target_lon_tas
)

pr_result = regrid_forecast(
    PR_INPUT, pr_output, "pr",
    target_lat_pr, target_lon_pr
)

print("\n=== REGRIDDING FORECAST SELESAI ===")
print("TAS:", tas_output)
print("PR :", pr_output)

era5_ds.close()
chirps_ds.close()
