# %% [markdown]
# # Download CHIRPS (Curah Hujan) - Subset Langsung Area Indonesia
#
# BEDA dari cara biasa: kita TIDAK download file CHIRPS global (~6.4GB),
# tapi pakai server ERDDAP (NOAA) yang bisa subset by area & tahun
# LANGSUNG di sisi server, jadi yang didownload cuma bagian Indonesia saja.
#
# Sumber: https://coastwatch.pfeg.noaa.gov/erddap/griddap/chirps20GlobalMonthlyP05.html

# %%
import os
import urllib.request
import xarray as xr
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "chirps")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Domain sama persis dengan yang dipakai untuk data DCPP ---
LAT_MIN, LAT_MAX = -13, 11
LON_MIN, LON_MAX = 90, 145

# --- Rentang waktu ---
# Hindcast kamu: init 1981-2010, lead year 1-10 -> butuh observasi sampai 2020
TIME_START = "1982-01-01T00:00:00Z"
TIME_END = "2020-12-01T00:00:00Z"

# %% [markdown]
# ## 1. Bangun URL Request (Subset Terjadi di Server, Bukan di Laptop Kita)

# %%
base_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/chirps20GlobalMonthlyP05.nc"

query = (
    f"?precip"
    f"[({TIME_START}):({TIME_END})]"
    f"[({LAT_MIN}):({LAT_MAX})]"
    f"[({LON_MIN}):({LON_MAX})]"
)

full_url = base_url + query
print("URL request yang akan dipakai:")
print(full_url)

# %% [markdown]
# ## 2. Download (Cuma Bagian Indonesia, Harusnya Jauh Lebih Kecil dari 6.4GB)

# %%
output_path = os.path.join(RAW_DIR, "chirps_monthly_indonesia_1982_2020.nc")

print(f"\nMendownload ke: {output_path}")
print("Mohon tunggu, tergantung koneksi mungkin beberapa menit...")

try:
    urllib.request.urlretrieve(full_url, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nBerhasil! Ukuran file: {file_size_mb:.2f} MB")
except Exception as e:
    print(f"\nGagal download: {e}")
    print("\nKemungkinan penyebab:")
    print("1. Server ERDDAP sedang down/lambat - coba lagi beberapa saat")
    print("2. Format URL berubah - cek manual di https://coastwatch.pfeg.noaa.gov/erddap/griddap/chirps20GlobalMonthlyP05.html")
    print("3. Koneksi internet terputus di tengah proses")

# %% [markdown]
# ## 3. Cek Hasilnya

# %%
if os.path.exists(output_path):
    ds = xr.open_dataset(output_path)
    print(ds)

    print("\n--- Info Variabel precip ---")
    print(ds['precip'].attrs)

# %% [markdown]
# ## 4. Visualisasi Cepat untuk Verifikasi

# %%
if os.path.exists(output_path):
    plt.figure(figsize=(8, 5))
    ds['precip'].mean(dim='time').plot(cmap='Blues')
    plt.title('Rata-rata Curah Hujan CHIRPS (mm/bulan) - Area Indonesia')
    plt.savefig(os.path.join(OUTPUT_DIR, '03_chirps_indonesia.png'), dpi=100)
    plt.close()
    print("Gambar tersimpan: outputs/03_chirps_indonesia.png")

    ds.close()

print("\n" + "="*60)
print("SELESAI download CHIRPS.")
print("Catatan satuan: CHIRPS asli dalam mm/bulan (mm/month),")
print("sedangkan data DCPP pr sudah dikonversi ke mm/hari.")
print("Perlu disamakan satuannya nanti saat tahap perbandingan/regridding.")
print("="*60)