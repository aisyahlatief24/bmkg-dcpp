# %% [markdown]
# # Regridding + Pencocokan Waktu
#
# Tujuan:
# 1. Interpolasi (regrid) grid model DCPP (kasar) ke grid observasi (halus)
# 2. Ambil rata-rata observasi pada WINDOW WAKTU YANG SAMA PERSIS dengan
#    tiap (init_date, lead_year) di model -> ini jadi TARGET/LABEL untuk model ML
# 3. Kalau window waktu sebagian tidak tersedia di data observasi
#    (misal init 1981 lead_year 1 butuh Nov1981-Okt1982, tapi observasi
#    baru mulai Jan1982), pasangan itu DIBUANG (NaN) - bukan dipaksa
#    dihitung dari data tidak lengkap yang bisa bias.

# %%
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Kalau True, CHIRPS diturunkan resolusinya ke ~0.25 derajat (dari 0.05 asli)
COARSEN_CHIRPS = True
CHIRPS_COARSEN_FACTOR = 5  # 0.05 x 5 = 0.25 derajat, mirip resolusi ERA5

# %% [markdown]
# ## 1. Load Semua Data

# %%
tas_model = xr.open_dataarray(
    os.path.join(
        PROCESSED_DIR,
        "tas_indonesia_alllead_1981_2019.nc"
    )
)
print("Model tas dimuat:", dict(tas_model.sizes))

era5_ds = xr.open_dataset(os.path.join(RAW_DIR, "era5", "era5_tas_indonesia_1982_2020.nc"))
era5_tas = era5_ds["t2m"] - 273.15  # Kelvin -> Celsius
era5_tas = era5_tas.rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})
era5_tas.attrs["units"] = "degC"
era5_tas.attrs["long_name"] = "2 metre temperature"
print("ERA5 dimuat:", dict(era5_tas.sizes))
print(f"ERA5 rentang waktu: {era5_tas.time.values[0]} sampai {era5_tas.time.values[-1]}")

pr_model = None
chirps_pr = None
pr_path = os.path.join(
    PROCESSED_DIR,
    "pr_indonesia_alllead_1981_2019.nc"
)
chirps_path = os.path.join(RAW_DIR, "chirps", "chirps_monthly_indonesia_1982_2020.nc")

if os.path.exists(pr_path):
    pr_model = xr.open_dataarray(pr_path)
    print("Model pr dimuat:", dict(pr_model.sizes))

if os.path.exists(chirps_path):
    chirps_ds = xr.open_dataset(chirps_path)
    chirps_pr = chirps_ds["precip"]
    chirps_pr = chirps_pr.rename({"latitude": "lat", "longitude": "lon"})
    # Konversi mm/bulan -> mm/hari (bagi jumlah hari di tiap bulan)
    days_in_month = chirps_pr["time"].dt.days_in_month
    chirps_pr = chirps_pr / days_in_month
    chirps_pr.attrs["units"] = "mm/day"

    if COARSEN_CHIRPS:
        chirps_pr = chirps_pr.coarsen(
            lat=CHIRPS_COARSEN_FACTOR, lon=CHIRPS_COARSEN_FACTOR, boundary="trim"
        ).mean()
        print(f"CHIRPS diturunkan resolusi (coarsen factor {CHIRPS_COARSEN_FACTOR})")
    print("CHIRPS dimuat:", dict(chirps_pr.sizes))

# %% [markdown]
# ## 2. Fungsi Bantuan: Hitung Window Waktu Nov-Okt untuk 1 (init_date, lead_year)

# %%
def get_lead_year_window(init_date_str, lead_year):
    """
    init_date_str: contoh 's1981'
    lead_year: 1 sampai 10
    Return: (start_date, end_date_exclusive) - window 12 bulan Nov s.d Okt tahun berikutnya
    """
    init_year = int(init_date_str.replace("s", ""))
    start_year = init_year + (lead_year - 1)
    start_date = pd.Timestamp(year=start_year, month=11, day=1)
    end_date = start_date + pd.DateOffset(years=1)  # exclusive
    return start_date, end_date


def get_obs_mean_for_window(obs_da, start_date, end_date):
    """
    Mengambil rata-rata observasi untuk 12 bulan Nov-Okt.
    Pasangan hanya diterima jika seluruh bulan tersedia.
    """
    mask = (
        (obs_da.time >= np.datetime64(start_date)) &
        (obs_da.time < np.datetime64(end_date))
    )

    subset = obs_da.sel(time=mask)

    expected_dates = pd.date_range(
        start=start_date,
        periods=12,
        freq="MS"
    )

    actual_dates = pd.DatetimeIndex(
        pd.to_datetime(subset.time.values)
    )

    if len(actual_dates) != 12:
        return None

    if not actual_dates.equals(expected_dates):
        return None

    return subset.mean(dim="time")

# %% [markdown]
# ## 3. Fungsi Bantuan: Regrid 1 Field Model (Kasar) ke Grid Observasi (Halus)

# %%
def regrid_to_target(da_coarse, target_lat, target_lon):
    """
    Interpolasi bilinear grid kasar ke grid target (observasi).

    kwargs fill_value='extrapolate' PENTING: tanpa ini, titik-titik di
    PINGGIR domain (misal lat paling utara/selatan) akan jadi NaN kalau
    grid asli model tidak PERSIS menyentuh batas domain target (ini
    hampir selalu terjadi karena resolusi grid berbeda). Dengan
    extrapolate, titik pinggir itu diisi dengan ekstrapolasi linear
    ringan dari titik-titik terdekat - jauh lebih baik daripada
    kehilangan seluruh baris/kolom data di pinggir Indonesia.
    """
    return da_coarse.interp(
        lat=target_lat,
        lon=target_lon,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )

# %% [markdown]
# ## 4. Proses Semua Kombinasi (sample x lead_year) untuk TAS

# %%
def process_variable(model_da, obs_da, varname):
    """
    Loop semua (sample, lead_year), regrid model ke grid obs,
    ambil obs yang match waktu, kumpulkan jadi 2 array besar:
    model_regridded dan obs_matched, dengan bentuk sama persis.
    """
    n_samples = model_da.sizes["sample"]
    n_lead = model_da.sizes["lead_year"]

    model_regridded_list = []
    obs_matched_list = []
    valid_mask_list = []  # True kalau window waktu lengkap tersedia

    target_lat = obs_da.lat
    target_lon = obs_da.lon

    n_skipped = 0
    for i in range(n_samples):
        init_date = str(model_da.init_date.values[i])
        for ly in range(1, n_lead + 1):
            field = model_da.isel(sample=i).sel(lead_year=ly)

            # Regrid model ke grid observasi
            field_regridded = regrid_to_target(field, target_lat, target_lon)

            # Ambil observasi yang cocok waktunya
            start_date, end_date = get_lead_year_window(init_date, ly)
            obs_mean = get_obs_mean_for_window(obs_da, start_date, end_date)

            if obs_mean is None:
                n_skipped += 1
                continue  # skip kombinasi ini, window waktu tidak lengkap di observasi

            model_regridded_list.append(field_regridded)
            obs_matched_list.append(obs_mean)
            valid_mask_list.append((init_date, ly, str(model_da.member.values[i])))

        if (i + 1) % 30 == 0:
            print(f"  [{varname}] Selesai proses {i+1}/{n_samples} sample...")

    print(f"  [{varname}] Total pasangan valid: {len(model_regridded_list)}")
    print(f"  [{varname}] Total di-skip (window waktu tidak lengkap): {n_skipped}")

    model_combined = xr.concat(model_regridded_list, dim="pair")
    obs_combined = xr.concat(obs_matched_list, dim="pair")

    # Tambahkan metadata pasangan (init_date, lead_year, member) sebagai koordinat
    init_dates, lead_years, members = zip(*valid_mask_list)
    model_combined = model_combined.assign_coords(
        pair=np.arange(len(valid_mask_list)),
        init_date=("pair", list(init_dates)),
        lead_year=("pair", list(lead_years)),
        member=("pair", list(members)),
    )
    obs_combined = obs_combined.assign_coords(pair=np.arange(len(valid_mask_list)))

    return model_combined, obs_combined

# %%
# %% [markdown]
# ## 5. Simpan Hasil TAS

# %%# %% [markdown]
# ## 5. Simpan Hasil TAS

# %%
print("\n=== Memproses TAS ===")

tas_model_regridded, tas_obs_matched = process_variable(
    tas_model, era5_tas, "tas"
)

tas_model_regridded.name = "tas_model"
tas_obs_matched.name = "tas_obs"

# Pemeriksaan koordinat sebelum menyimpan
assert np.array_equal(
    tas_model_regridded.lat.values,
    tas_obs_matched.lat.values
), "Latitude TAS tidak sama!"

assert np.array_equal(
    tas_model_regridded.lon.values,
    tas_obs_matched.lon.values
), "Longitude TAS tidak sama!"

print("Koordinat TAS model dan ERA5 identik.")

# Simpan hasil
tas_model_regridded.to_netcdf(
    os.path.join(PROCESSED_DIR, "tas_model_regridded.nc")
)
tas_obs_matched.to_netcdf(
    os.path.join(PROCESSED_DIR, "tas_obs_matched.nc")
)

print("Disimpan: tas_model_regridded.nc dan tas_obs_matched.nc")

# %% [markdown]
# ## 6. Visualisasi Cepat: Bandingkan 1 Contoh Pasangan (Model vs Obs)

# %%
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
tas_model_regridded.isel(pair=0).plot(cmap="coolwarm")
plt.title("Model (setelah regrid ke grid ERA5)")

plt.subplot(1, 2, 2)
tas_obs_matched.isel(pair=0).plot(cmap="coolwarm")
plt.title("Observasi ERA5 (window waktu cocok)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_regrid_compare_tas.png"), dpi=100)
plt.close()
print("Gambar tersimpan: outputs/05_regrid_compare_tas.png")

# %% [markdown]
# ## 7. Ulangi untuk PR (Kalau Data Tersedia)

# %%
if pr_model is not None and chirps_pr is not None:

    print("\n=== Memproses PR ===")

    pr_model_regridded, pr_obs_matched = process_variable(
        pr_model, chirps_pr, "pr"
    )

    pr_model_regridded.name = "pr_model"
    pr_obs_matched.name = "pr_obs"

    # Pemeriksaan koordinat sebelum menyimpan
    assert np.array_equal(
        pr_model_regridded.lat.values,
        pr_obs_matched.lat.values
    ), "Latitude PR tidak sama!"

    assert np.array_equal(
        pr_model_regridded.lon.values,
        pr_obs_matched.lon.values
    ), "Longitude PR tidak sama!"

    print("Koordinat PR model dan CHIRPS identik.")

    # Simpan hasil
    pr_model_regridded.to_netcdf(
        os.path.join(PROCESSED_DIR, "pr_model_regridded.nc")
    )
    pr_obs_matched.to_netcdf(
        os.path.join(PROCESSED_DIR, "pr_obs_matched.nc")
    )

    print("Disimpan: pr_model_regridded.nc dan pr_obs_matched.nc")

else:
    print("\nData PR/CHIRPS belum ditemukan, skip proses PR.")