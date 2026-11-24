# %% [markdown]
# # Preprocessing: Subset Area Indonesia + Konversi Satuan + Semua Lead Year
#
# Lanjutan dari 01_explore_data.py. Bedanya di sini:
# 1. Data langsung DIPOTONG ke area Indonesia (bukan global lagi) -> jauh lebih kecil
# 2. Konversi satuan: Kelvin->Celsius (tas), kg/m2/s->mm/hari (pr)
# 3. Ambil SEMUA lead year 1-10 (bukan cuma lead year 1 seperti sebelumnya)
# 4. Proses KEDUA variabel (tas dan pr), bukan cuma tas

# %%
import os
import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = {
    'tas': os.path.join(BASE_DIR, "data", "raw", "tas"),
    'pr': os.path.join(BASE_DIR, "data", "raw", "pr"),
}
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Domain Indonesia (mengikuti domain yang dipakai studi kating sebelumnya) ---
# PENTING: longitude data DCPP formatnya 0-360 (bukan -180 sampai 180),
# tapi domain Indonesia (90-145 BT) kebetulan tetap positif di kedua format,
# jadi TIDAK PERLU konversi khusus untuk kasus ini.
LON_MIN, LON_MAX = 90, 145
LAT_MIN, LAT_MAX = -13, 11

print(f"Domain Indonesia yang dipakai: {LON_MIN}-{LON_MAX} BT, {LAT_MIN}-{LAT_MAX} LU/LS")

# %% [markdown]
# ## 1. Fungsi Bantuan (Parsing Nama File & Load Satu File)

# %%
def parse_dcpp_filename(filepath):
    """Ekstrak init_date dan member dari nama file DCPP standar CMIP6."""
    fname = os.path.basename(filepath)
    parts = fname.split('_')
    s_r_part = parts[4]
    init_date = s_r_part.split('-')[0]
    member = s_r_part.split('-')[1]
    return init_date, member


def load_subset_convert(filepath, varname):
    """
    Buka 1 file, SUBSET ke area Indonesia, konversi satuan,
    dan hitung rata-rata per LEAD YEAR (1 sampai 10) alih-alih cuma lead year 1.
    """
    init_date, member = parse_dcpp_filename(filepath)
    ds = xr.open_dataset(filepath)

    # --- Subset spasial ke area Indonesia ---
    da = ds[varname].sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))

    # --- Konversi satuan ---
    if varname == 'tas':
        da = da - 273.15  # Kelvin -> Celsius
        da.attrs['units'] = 'degC'
    elif varname == 'pr':
        da = da * 86400  # kg/m2/s -> mm/hari
        da.attrs['units'] = 'mm/day'

    # --- Hitung rata-rata per lead year (1-10), bukan cuma lead year 1 ---
    # Data dimulai November tahun init, jadi lead year 1 = bulan ke-0 s.d 11 (Nov thn0 - Okt thn1)
    # Kita sederhanakan: kelompokkan per BLOK 12 BULAN berurutan dari awal file.
    n_months = da.sizes['time']
    n_lead_years = n_months // 12  # biasanya 10 (kadang bisa 10 pas, sisa dibuang)

    lead_year_means = []
    for ly in range(n_lead_years):
        start = ly * 12
        end = start + 12
        chunk = da.isel(time=slice(start, end)).mean(dim='time')
        lead_year_means.append(chunk)

    # Gabungkan jadi 1 DataArray dengan dimensi baru 'lead_year' (1 sampai n_lead_years)
    da_leadyears = xr.concat(lead_year_means, dim='lead_year')
    da_leadyears = da_leadyears.assign_coords(lead_year=np.arange(1, n_lead_years + 1))
    da_leadyears = da_leadyears.assign_coords(init_date=init_date, member=member)

    ds.close()
    return da_leadyears

# %% [markdown]
# ## 2. Test Dulu di 1 File (Cek Ukuran Sebelum-Sesudah Subset)

# %%
tas_files = sorted(glob.glob(os.path.join(RAW_DIR['tas'], "*.nc")))
pr_files = sorted(glob.glob(os.path.join(RAW_DIR['pr'], "*.nc")))

print(f"File tas ditemukan: {len(tas_files)}")
print(f"File pr ditemukan : {len(pr_files)}")

if len(tas_files) > 0:
    contoh = load_subset_convert(tas_files[0], 'tas')
    print("\n--- Hasil subset + konversi (1 file contoh, variabel tas) ---")
    print(contoh)
    print(f"\nUkuran grid setelah subset Indonesia: lat={contoh.sizes['lat']}, lon={contoh.sizes['lon']}")
    print(f"Jumlah lead year yang berhasil dihitung: {contoh.sizes['lead_year']}")
    print(f"Rentang nilai suhu (°C): {float(contoh.min()):.2f} sampai {float(contoh.max()):.2f}")

# %% [markdown]
# ## 3. Visualisasi Cepat: Cek Domain Indonesia Sudah Benar Terpotong

# %%
if len(tas_files) > 0:
    plt.figure(figsize=(8, 5))
    contoh.isel(lead_year=0).plot(cmap='coolwarm')
    plt.title('Suhu Rata-rata Lead Year 1 (°C) - Area Indonesia (hasil subset)')
    plt.savefig(os.path.join(OUTPUT_DIR, '02_subset_indonesia_tas.png'), dpi=100)
    plt.close()  # pakai close() bukan show() supaya tidak nge-block proses lanjut
    print("Gambar tersimpan: outputs/02_subset_indonesia_tas.png")
    print("(dipakai plt.close() bukan plt.show(), supaya script tidak berhenti nunggu window ditutup)")

# %% [markdown]
# ## 4. Proses SEMUA File untuk KEDUA Variabel (tas & pr)
# Peringatan: proses ini makan waktu (sudah dites ~150 file tas butuh beberapa menit).
# Untuk 300 file (tas+pr) dengan subset area, sebenarnya harusnya LEBIH CEPAT dari sebelumnya
# karena ukuran data per file jauh lebih kecil setelah dipotong ke Indonesia.

# %%
def process_all_files(file_list, varname):
    """Proses semua file untuk 1 variabel, gabungkan jadi 1 DataArray besar."""
    result_list = []
    for i, f in enumerate(file_list):
        try:
            da = load_subset_convert(f, varname)
            result_list.append(da)
        except Exception as e:
            print(f"  Gagal proses {os.path.basename(f)}: {e}")

        if (i + 1) % 30 == 0:
            print(f"  [{varname}] Selesai {i+1}/{len(file_list)} file...")

    if len(result_list) == 0:
        return None

    combined = xr.concat(result_list, dim='sample')
    combined = combined.assign_coords(sample=np.arange(len(result_list)))
    return combined

# %%
print("\n=== Memproses semua file TAS ===")
tas_combined = process_all_files(tas_files, 'tas') if len(tas_files) > 0 else None

print("\n=== Memproses semua file PR ===")
pr_combined = process_all_files(pr_files, 'pr') if len(pr_files) > 0 else None

# %% [markdown]
# ## 5. Simpan Hasil Akhir

# %%
if tas_combined is not None:
    print("\n--- Struktur akhir TAS (subset Indonesia, semua lead year) ---")
    print(tas_combined)
    out_path = os.path.join(PROCESSED_DIR, 'tas_indonesia_alllead.nc')
    tas_combined.to_netcdf(out_path)
    print(f"Disimpan: {out_path}")

if pr_combined is not None:
    print("\n--- Struktur akhir PR (subset Indonesia, semua lead year) ---")
    print(pr_combined)
    out_path = os.path.join(PROCESSED_DIR, 'pr_indonesia_alllead.nc')
    pr_combined.to_netcdf(out_path)
    print(f"Disimpan: {out_path}")

print("\n" + "="*60)
print("SELESAI preprocessing tahap ini.")
print("Data sudah: di-subset ke Indonesia, dikonversi satuan, semua lead year 1-10.")
print("Langkah selanjutnya: download data observasi (ERA5 untuk tas, CHIRPS untuk pr)")
print("lalu regridding supaya grid model & observasi cocok.")
print("="*60)

