# %% [markdown]
# # Download ERA5 (Suhu) - Subset Langsung Area Indonesia via CDS API
#
# Beda dengan CHIRPS tadi (pakai urllib biasa), ERA5 lewat CDS API resmi
# yang bisa langsung minta subset area saat request - server yang motong
# datanya, bukan kita download global dulu.

# %%
import os
import cdsapi
import xarray as xr
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "era5")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Domain sama persis dengan data DCPP & CHIRPS sebelumnya ---
# Format CDS: [North, West, South, East]
AREA = [11, 90, -13, 145]

output_path = os.path.join(RAW_DIR, "era5_tas_indonesia_1982_2020.nc")

# %% [markdown]
# ## 1. Request Data ke CDS
# PENTING: sebelum ini jalan, kamu HARUS sudah buka halaman dataset
# "ERA5 monthly averaged data on single levels from 1940 to present"
# di web CDS dan klik "Accept Terms of Use" MINIMAL SEKALI, kalau belum
# request ini akan gagal dengan error 403.

# %%
client = cdsapi.Client()

if os.path.exists(output_path):
    print(f"File sudah ada, SKIP download: {output_path}")
    print("(Hapus file ini manual kalau memang ingin download ulang)")
else:
    print("Mengirim request ke CDS... (request besar bisa antre beberapa menit di server)")
    try:
        client.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable": "2m_temperature",
                "year": [str(y) for y in range(1982, 2021)],  # 1982-2020
                "month": [f"{m:02d}" for m in range(1, 13)],
                "time": "00:00",
                "area": AREA,
                "format": "netcdf",
            },
            output_path
        )
        print(f"\nBerhasil! File disimpan di: {output_path}")
    except Exception as e:
        print(f"\nGagal download: {e}")
        print("\nKemungkinan penyebab paling umum:")
        print("1. Belum accept Terms of Use dataset ini di web CDS -> buka:")
        print("   https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means")
        print("   klik dataset itu, scroll ke bawah, cari tombol accept terms")
        print("2. Nama dataset/parameter berubah -> cek ulang di web CDS,")
        print("   pilih opsi manual, klik 'Show API request' untuk dapat kode terbaru")

# %% [markdown]
# ## 2. Cek Hasilnya

# %%
if os.path.exists(output_path):
    ds = xr.open_dataset(output_path)
    print(ds)

    print("\n--- Info Variabel Suhu ---")
    # Nama variabel di ERA5 biasanya 't2m', cek juga kemungkinan nama lain
    var_name = 't2m' if 't2m' in ds.data_vars else list(ds.data_vars)[0]
    print(f"Nama variabel: {var_name}")
    print(ds[var_name].attrs)

# %% [markdown]
# ## 3. Visualisasi Cepat

# %%
if os.path.exists(output_path):
    plt.figure(figsize=(8, 5))
    # Konversi Kelvin -> Celsius langsung saat plot
    # Catatan: ERA5 kadang kasih nama dimensi waktu 'valid_time', bukan 'time' -
    # deteksi otomatis supaya kode tetap jalan di kedua kemungkinan.
    time_dim = 'time' if 'time' in ds[var_name].dims else 'valid_time'
    (ds[var_name].mean(dim=time_dim) - 273.15).plot(cmap='coolwarm')
    plt.title('Rata-rata Suhu ERA5 (°C) - Area Indonesia')
    plt.savefig(os.path.join(OUTPUT_DIR, '04_era5_indonesia.png'), dpi=100)
    plt.close()
    print("Gambar tersimpan: outputs/04_era5_indonesia.png")

    ds.close()

print("\n" + "="*60)
print("SELESAI download ERA5.")
print("Sekarang kamu punya data observasi untuk KEDUA variabel:")
print("  - CHIRPS (curah hujan)")
print("  - ERA5 (suhu)")
print("Langkah selanjutnya: regridding (samakan grid model DCPP vs observasi)")
print("="*60)