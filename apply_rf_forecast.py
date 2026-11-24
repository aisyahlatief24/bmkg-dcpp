import os
import joblib
import numpy as np
import xarray as xr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FORECAST_DIR = os.path.join(PROCESSED_DIR, "forecast_s2020")
OUTPUT_DIR = os.path.join(FORECAST_DIR, "rf_corrected")

os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = {
    "tas": {
        "input": "tas_dcppB_forecast_s2020_regridded_to_era5.nc",
        "units": "degC",
    },
    "pr": {
        "input": "pr_dcppB_forecast_s2020_regridded_to_chirps025.nc",
        "units": "mm/day",
    },
}

BATCH_SIZE = 100_000


def predict_in_batches(model, X):
    """Prediksi per batch agar penggunaan memori lebih terkendali."""
    result = np.full(X.shape[0], np.nan, dtype=np.float32)

    valid = np.isfinite(X).all(axis=1)
    valid_idx = np.where(valid)[0]

    for start in range(0, len(valid_idx), BATCH_SIZE):
        idx = valid_idx[start:start + BATCH_SIZE]
        result[idx] = model.predict(X[idx]).astype(np.float32)

    return result


def apply_rf(variable, config):
    print("\n" + "=" * 60)
    print(f"APPLY RANDOM FOREST: {variable.upper()}")
    print("=" * 60)

    model_path = os.path.join(BASE_DIR, "models", f"rf_{variable}.pkl")
    input_path = os.path.join(FORECAST_DIR, config["input"])

    rf = joblib.load(model_path)
    ds = xr.open_dataset(input_path)

    da = ds[variable]

    lat = da.lat.values
    lon = da.lon.values
    lead_years = da.lead_year.values
    members = da.member.values

    n_sample = da.sizes["sample"]
    n_lead = da.sizes["lead_year"]
    n_lat = da.sizes["lat"]
    n_lon = da.sizes["lon"]

    print("Input shape:", da.shape)
    print("Members:", members)
    print("Lead years:", lead_years)
    print("Grid:", n_lat, "x", n_lon)
    print("RF features:", rf.n_features_in_)

    lon_grid, lat_grid = np.meshgrid(lon, lat)

    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()

    raw = da.values
    corrected = np.full(raw.shape, np.nan, dtype=np.float32)

    init_year = int(str(ds["init_date"].values[0]).replace("s", ""))

    for s in range(n_sample):
        member = str(members[s])

        for l in range(n_lead):
            ly = int(lead_years[l])

            model_flat = raw[s, l].ravel()

            X = np.column_stack([
                model_flat,
                lat_flat,
                lon_flat,
                np.full(model_flat.shape, ly),
                np.full(model_flat.shape, init_year),
            ])

            pred = predict_in_batches(rf, X)
            corrected[s, l] = pred.reshape(n_lat, n_lon)

            print(
                f"Member {member}, LY{ly}: "
                f"{np.isfinite(pred).sum():,} titik diprediksi"
            )

    corrected_da = xr.DataArray(
        corrected,
        dims=("sample", "lead_year", "lat", "lon"),
        coords={
            "sample": da.sample.values,
            "lead_year": lead_years,
            "lat": lat,
            "lon": lon,
            "member": ("sample", members),
            "init_date": ("sample", ds.init_date.values),
        },
        name=f"{variable}_rf_corrected",
        attrs={
            "long_name": f"{variable.upper()} bias-corrected using Random Forest",
            "units": config["units"],
            "method": "Random Forest",
            "training_spatial_thinning_factor": 4,
            "forecast_initialization": "s2020",
            "note": (
                "Full-grid prediction using RF trained on spatially thinned "
                "historical grid points."
            ),
        },
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{variable}_rf_corrected_dcppB_s2020.nc"
    )

    corrected_da.to_dataset().to_netcdf(output_path)

    print("\nOutput:", output_path)
    print("Output shape:", corrected_da.shape)
    print("Minimum:", float(corrected_da.min(skipna=True)))
    print("Maximum:", float(corrected_da.max(skipna=True)))
    print("NaN count:", int(np.isnan(corrected).sum()))

    ds.close()


if __name__ == "__main__":
    for variable, config in CONFIG.items():
        apply_rf(variable, config)

    print("\nSELESAI: RF FORECAST TAS DAN PR")
