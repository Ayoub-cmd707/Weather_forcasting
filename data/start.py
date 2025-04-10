import numpy as np
import climetlab as cml
import xarray as xr 
import pandas as pd
import os
import zarr

# #### Surface variables 

sfc_rfc =cml.load_dataset('EUPPBench-training-data-gridded-reforecasts-surface')

# #### presure level variables 

pl500_rfc = cml.load_dataset('EUPPBench-training-data-gridded-reforecasts-pressure', level='500')
pl700_rfc = cml.load_dataset('EUPPBench-training-data-gridded-reforecasts-pressure', level='700')
pl850_rfc=cml.load_dataset('EUPPBench-training-data-gridded-reforecasts-pressure', level='850')

# #### processed variables

proc_rfc = cml.load_dataset('EUPPBench-training-data-gridded-reforecasts-surface-processed')

t2m=sfc_rfc.to_xarray()[['t2m']] 
u10=sfc_rfc.to_xarray()[['u10']]
v10=sfc_rfc.to_xarray()[['v10']]
tcc=sfc_rfc.to_xarray()[['tcc']]
u100 =sfc_rfc.to_xarray()[['u100']]
v100 =sfc_rfc.to_xarray()[['v100']]

z=pl500_rfc.to_xarray()[['z']] 
t=pl850_rfc.to_xarray()[['t']]

p10fg6=proc_rfc.to_xarray()[['p10fg6']]
ssrd6=proc_rfc.to_xarray()[['ssrd6']]
strd6=proc_rfc.to_xarray()[['strd6']]


# #### Getting the observations 
 
ssrd6_obs = xr.Dataset({'ssrd6_obs': proc_rfc.get_observations_as_xarray()['ssrd6']})

datasets = [t2m, u10, v10, tcc, u100, v100, z, t, p10fg6, ssrd6, strd6, ssrd6_obs]

for i in range(len(datasets)):
    # Convert 'step' from nanoseconds to hours
    datasets[i]['step'] = pd.to_timedelta(datasets[i]['step'], unit='ns').total_seconds() / 3600
    # Select specific time steps
    if i<=7:   #remove first forecasting step because it is absent in the processed variables 
         datasets[i] = datasets[i].isel(step=slice(1,None))
    # Squeeze unnecessary dimensions
    if 'depthBelowLandLayer' in datasets[i].dims:
        datasets[i] = datasets[i].squeeze('depthBelowLandLayer')
    if 'surface' in datasets[i].dims:
        datasets[i] = datasets[i].squeeze('surface')
    if 'isobaricInhPa' in datasets[i].variables:
        datasets[i] = datasets[i].drop_vars('isobaricInhPa')
    if "isobaricInhPa" in datasets[i].dims:
        datasets[i] = datasets[i].squeeze(dim="isobaricInhPa")
    if 'surface' in datasets[i].variables:
        datasets[i] = datasets[i].drop_vars('surface')
    if "surface" in datasets[i].dims:
        datasets[i] = datasets[i].squeeze(dim="surface")

rfc_all = xr.merge(datasets[0:12])

output_dir_forecast = "./EUPP"
output_dir_era5 ="./ERA5"



for year in range(3):
    yeardata=rfc_all.isel(year=year)
    for time in yeardata.time:
        time_str = pd.to_datetime(str(time.values)).strftime('%Y%m%d')
        filename = f"output.sfc.{int(year)}.{time_str}.nc"
        filepath = os.path.join(output_dir_forecast, filename)
        # Select the data for the current time step
        xds_at_time=yeardata.sel(time=time)
        xds_at_time.to_netcdf(filepath)
        print('EUPP printing ',year)


#observations
for year in range(3): 
    yeardata=ssrd6_obs.isel(year=year)
    for time in yeardata.time:
        time_str = pd.to_datetime(str(time.values)).strftime('%Y%m%d')
        filename = f"era.sfc.{int(year)}.{time_str}.nc"
        filepath = os.path.join(output_dir_era5, filename)
        # Select the data for the current time step
        xds_at_time=yeardata.sel(time=time)
        xds_at_time.to_netcdf(filepath)
        print('ERA5 printing ',year)