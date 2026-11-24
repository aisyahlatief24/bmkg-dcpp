
import os
import glob
import numpy as np
import xarray as xr

# =========================================================
# 1. KONFIGURASI FOLDER
# Skrip disimpan di folder utama magangproject
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = {
    "tas": os.path.join(BASE_DIR, "data", "raw", "tas", "dcppb_tas"),
    "pr": os.path.join(BASE_DIR, "data", "raw", "pr", "dcppb_pr"),
}

PROCESSED_DIR = os.path.join(
    BASE_DIR, "data", "processed", "forecast_s2020"
)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Domain Indonesia, sama dengan preprocessing hindcast
LON_MIN, LON_MAX = 90, 145
LAT_MIN, LAT_MAX = -13, 11

# Forecast mulai November 2020.
# 10 lead year x 12 bulan = 120 bulan.
N_LEAD_YEARS = 10
MONTHS_PER_YEAR = 12
N_MONTHS_REQUIRED = N_LEAD_YEARS * MONTHS_PER_YEAR

print("Folder output:", PROCESSED_DIR)
print("Domain Indonesia:", LON_MIN, "-", LON_MAX, "BT;",
      LAT_MIN, "-", LAT_MAX, "derajat lintang")


# =========================================================
# 2. AMBIL DAFTAR FILE FORECAST
# =========================================================

PATTERNS = {
    "tas": "tas_Amon_CMCC-CM2-SR5_dcppB-forecast_s2020-r*i1p1f1_gn_202011-203012.nc",
    "pr": "pr_Amon_CMCC-CM2-SR5_dcppB-forecast_s2020-r*i1p1f1_gn_202011-203012.nc",
}

files_by_var = {}

for varname in ["tas", "pr"]:
    pattern = os.path.join(RAW_DIR[varname], PATTERNS[varname])
    files = sorted(glob.glob(pattern))
    files_by_var[varname] = files

    print(f"Jumlah file {varname}: {len(files)}")

    if len(files) != 5:
        raise ValueError(
            f"Seharusnya ada 5 file {varname}, tetapi ditemukan {len(files)}. "
            f"Periksa folder: {RAW_DIR[varname]}"
        )


# =========================================================
# 3. SUBSET INDONESIA, KONVERSI, DAN LEAD YEAR
# =========================================================

def load_subset_convert(filepath, varname):
    """Memproses satu file forecast menjadi 10 lead-year means."""

    filename = os.path.basename(filepath)

    with xr.open_dataset(filepath) as ds:
        da = ds[varname]

        # Pastikan satuan sesuai dengan asumsi konversi
        units = da.attrs.get("units", "").strip()

        if varname == "tas" and units.lower() not in ["k", "kelvin"]:
            raise ValueError(
                f"Satuan tas tidak sesuai pada {filename}: {units}"
            )

        if varname == "pr" and units not in [
            "kg m-2 s-1", "kg m**-2 s**-1"
        ]:
            raise ValueError(
                f"Satuan pr tidak sesuai pada {filename}: {units}"
            )

        # Subset spasial terlebih dahulu agar data lebih kecil
        da = da.sel(
            lat=slice(LAT_MIN, LAT_MAX),
            lon=slice(LON_MIN, LON_MAX),
        ).load()

        # Konversi satuan
        if varname == "tas":
            da = da - 273.15
            da.attrs["units"] = "degC"

        elif varname == "pr":
            da = da * 86400
            da.attrs["units"] = "mm/day"

        # Validasi waktu
        n_months = da.sizes["time"]

        if n_months < N_MONTHS_REQUIRED:
            raise ValueError(
                f"{filename} hanya memiliki {n_months} bulan; "
                f"dibutuhkan minimal {N_MONTHS_REQUIRED} bulan."
            )

        # Ambil 120 bulan pertama untuk lead year 1-10.
        # Dua bulan terakhir (Nov-Des 2030) tidak masuk ke
        # 10 blok 12 bulan ini, konsisten dengan metode hindcast.
        da = da.isel(time=slice(0, N_MONTHS_REQUIRED))

        lead_year_means = []

        for ly in range(N_LEAD_YEARS):
            start = ly * MONTHS_PER_YEAR
            end = start + MONTHS_PER_YEAR

            chunk = da.isel(time=slice(start, end)).mean(
                dim="time",
                skipna=True,
            )
            lead_year_means.append(chunk)

        result = xr.concat(
            lead_year_means,
            dim=xr.IndexVariable(
                "lead_year", np.arange(1, N_LEAD_YEARS + 1)
            ),
        )

        # Identitas forecast dan ensemble
        member = ds.attrs.get("variant_label")

        if member is None:
            raise ValueError(f"variant_label tidak ditemukan: {filename}")

        result = result.expand_dims(sample=[0])
        result = result.assign_coords(
            member=("sample", [member]),
            init_date=("sample", ["s2020"]),
        )

        result.name = varname
        result.attrs["units"] = (
            "degC" if varname == "tas" else "mm/day"
        )
        result.attrs["description"] = (
            "Indonesia subset, monthly data averaged into "
            "lead years 1-10, DCPP-B forecast initialized at s2020"
        )

        return result


# =========================================================
# 4. PROSES SEMUA ENSEMBLE UNTUK SETIAP VARIABEL
# =========================================================

def process_all_files(file_list, varname):
    results = []

    for i, filepath in enumerate(file_list, start=1):
        print(f"[{varname}] Memproses {i}/{len(file_list)}: "
              f"{os.path.basename(filepath)}")

        da = load_subset_convert(filepath, varname)

        # Beri indeks sample yang unik untuk setiap ensemble
        da = da.assign_coords(sample=[i - 1])
        results.append(da)

    combined = xr.concat(results, dim="sample")

    return combined


# =========================================================
# 5. PROSES DAN SIMPAN OUTPUT
# =========================================================

for varname in ["tas", "pr"]:
    print(f"\n=== Memproses forecast {varname.upper()} ===")

    combined = process_all_files(
        files_by_var[varname],
        varname,
    )

    output_path = os.path.join(
        PROCESSED_DIR,
        f"{varname}_dcppB_forecast_s2020_indonesia_alllead.nc",
    )

    combined.to_netcdf(output_path)

    print("\nStruktur output:")
    print(combined)
    print("Disimpan:", output_path)

print("\nSelesai preprocessing forecast tas dan pr.")