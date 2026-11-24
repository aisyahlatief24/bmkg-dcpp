import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_DIR = os.path.join(
    BASE_DIR, "data", "processed", "forecast_s2020"
)

raw_path = os.path.join(
    FORECAST_DIR,
    "tas_dcppB_forecast_s2020_regridded_to_era5.nc"
)

cnn_path = os.path.join(
    FORECAST_DIR,
    "cnn_corrected",
    "tas_cnn_corrected_dcppB_s2020.nc"
)

output_dir = os.path.join(BASE_DIR, "outputs")
os.makedirs(output_dir, exist_ok=True)

raw_ds = xr.open_dataset(raw_path)
cnn_ds = xr.open_dataset(cnn_path)

raw = raw_ds["tas"].isel(sample=0, lead_year=0)
cnn = cnn_ds["tas_cnn_corrected"].isel(
    sample=0, lead_year=0
)

difference = cnn - raw

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

maps = [
    (raw, "RAW TAS - DCPP-B"),
    (cnn, "CNN U-Net Corrected"),
    (difference, "Perubahan (CNN - RAW)")
]

for ax, (data, title) in zip(axes, maps):
    if "Perubahan" in title:
        vmax = float(np.nanmax(np.abs(data.values)))
        im = ax.pcolormesh(
            data.lon, data.lat, data,
            cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
            shading="auto"
        )
    else:
        im = ax.pcolormesh(
            data.lon, data.lat, data,
            cmap="coolwarm",
            shading="auto"
        )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, label="°C")

plt.suptitle(
    "Perbandingan Forecast TAS - Member r1, Lead Year 1",
    fontsize=14
)

plt.tight_layout()

save_path = os.path.join(
    output_dir, "tas_raw_vs_cnn_s2020_r1_ly1.png"
)

plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close()

print("Peta tersimpan:", save_path)

raw_ds.close()
cnn_ds.close()
