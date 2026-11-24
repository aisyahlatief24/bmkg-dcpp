
import os
import numpy as np
import xarray as xr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_DIR = os.path.join(
    BASE_DIR, "data", "processed", "forecast_s2020"
)

RAW_FILE = os.path.join(
    FORECAST_DIR,
    "pr_dcppB_forecast_s2020_regridded_to_chirps025.nc"
)

CORRECTED_FILE = os.path.join(
    FORECAST_DIR, "rf_corrected",
    "pr_rf_corrected_dcppB_s2020.nc"
)


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    ds = xr.open_dataset(path)

    print(f"\nFile: {os.path.basename(path)}")
    print("Data variables:", list(ds.data_vars))

    # Ambil variabel pertama jika nama variabel belum diketahui
    da = ds[list(ds.data_vars)[0]]

    print("Variable:", da.name)
    print("Dimensions:", da.dims)
    print("Shape:", da.shape)
    print("Units:", da.attrs.get("units", "Tidak tercantum"))

    return ds, da


def describe(da, label):
    values = np.asarray(da.values, dtype=np.float64)
    valid = values[np.isfinite(values)]

    print(f"\n=== {label} ===")
    print("Total data:", values.size)
    print("NaN:", np.isnan(values).sum())
    print("Inf:", np.isinf(values).sum())

    if valid.size == 0:
        print("Tidak ada nilai valid.")
        return

    print(f"Minimum : {valid.min():.4f} mm/day")
    print(f"Maksimum: {valid.max():.4f} mm/day")
    print(f"Mean    : {valid.mean():.4f} mm/day")
    print(f"Median  : {np.median(valid):.4f} mm/day")
    print(f"P01     : {np.percentile(valid, 1):.4f} mm/day")
    print(f"P99     : {np.percentile(valid, 99):.4f} mm/day")
    print("Nilai negatif:", (valid < 0).sum())
    print("Nilai nol:", (valid == 0).sum())


def per_member_lead(da, label):
    print(f"\n=== RINGKASAN {label}: MEMBER DAN LEAD YEAR ===")

    for member in da.member.values:
        for lead in da.lead_year.values:
            values = da.sel(
                member=member,
                lead_year=lead
            ).values

            valid = values[np.isfinite(values)]

            if valid.size == 0:
                print(member, lead, "Tidak ada data valid")
                continue

            print(
                f"{str(member):12s} LY{int(lead):02d} | "
                f"min={valid.min():8.3f} | "
                f"max={valid.max():8.3f} | "
                f"mean={valid.mean():8.3f} | "
                f"negatif={(valid < 0).sum():5d} | "
                f"NaN={np.isnan(values).sum():5d}"
            )


def main():
    raw_ds, raw_da = load_data(RAW_FILE)
    corrected_ds, corrected_da = load_data(CORRECTED_FILE)

    describe(raw_da, "FORECAST PR MENTAH")
    describe(corrected_da, "FORECAST PR SETELAH RF")

    per_member_lead(raw_da, "PR MENTAH")
    per_member_lead(corrected_da, "PR RF")

    print("\n=== PEMERIKSAAN KOORDINAT ===")

    for coord in ["member", "lead_year", "lat", "lon"]:
        if coord in raw_da.coords and coord in corrected_da.coords:
            same = np.array_equal(
                raw_da[coord].values,
                corrected_da[coord].values
            )
            print(f"{coord}: {'SAMA' if same else 'BERBEDA'}")

    print("\n=== SELESAI ===")

    raw_ds.close()
    corrected_ds.close()


if __name__ == "__main__":
    main()
