import glob
import xarray as xr
import numpy as np

eupp_files = glob.glob("./data/EUPP_merged/output.sfc.*.nc")
era5_files = glob.glob("./data/ERA5/era.sfc.*.nc")

ssrd6_min, ssrd6_max = np.inf, -np.inf
ssrd6_obs_min, ssrd6_obs_max = np.inf, -np.inf

# Eerst EUPP-bestanden
for file in eupp_files:
    ds = xr.open_dataset(file)
    if "tp6" in ds:
        values = ds["tp6"].values
        values = values[np.isfinite(values)]  # verwijder NaNs
        ssrd6_min = min(ssrd6_min, values.min())
        ssrd6_max = max(ssrd6_max, values.max())
    ds.close()
print(f"tp6 value range: min = {ssrd6_min}, max = {ssrd6_max}")
# Nu ERA5 observaties
for file in era5_files:
    ds = xr.open_dataset(file)
    if "ssrd6_obs" in ds:
        values = ds["ssrd6_obs"].values
        values = values[np.isfinite(values)]
        ssrd6_obs_min = min(ssrd6_obs_min, values.min())
        ssrd6_obs_max = max(ssrd6_obs_max, values.max())
    ds.close()

print(f"ssrd6_obs value range: min = {ssrd6_obs_min}, max = {ssrd6_obs_max}")
