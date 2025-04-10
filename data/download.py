import numpy as np
import climetlab as cml
import xarray as xr 
import pandas as pd
import os

# #### Surface variables 

sfc_rfc =cml.load_dataset('EUPPBench-training-data-gridded-reforecasts-surface')


t2m=sfc_rfc.to_xarray()[['t2m']] 
u10=sfc_rfc.to_xarray()[['u10']]
v10=sfc_rfc.to_xarray()[['v10']]
tcc=sfc_rfc.to_xarray()[['tcc']]
u100 =sfc_rfc.to_xarray()[['u100']]
v100 =sfc_rfc.to_xarray()[['v100']]

print('Finished reforcast surface')

# #### presure level variables 


pl500_rfc = cml.load_dataset('EUPPBench-training-data-gridded-reforecasts-pressure', level='500')
pl700_rfc = cml.load_dataset('EUPPBench-training-data-gridded-reforecasts-pressure', level='700')
pl850_rfc=cml.load_dataset('EUPPBench-training-data-gridded-reforecasts-pressure', level='850')


z=pl500_rfc.to_xarray()[['z']] 
u700=pl700_rfc.to_xarray()[['u']]
v700= pl700_rfc.to_xarray()[['v']]
t=pl850_rfc.to_xarray()[['t']] 
print('Finished reforcast pressure')

proc_rfc = cml.load_dataset('EUPPBench-training-data-gridded-reforecasts-surface-processed')


p10fg6=proc_rfc.to_xarray()[['p10fg6']]
ssr6=proc_rfc.to_xarray()[['ssr6']]
ssrd6=proc_rfc.to_xarray()[['ssrd6']]
print('Finished reforcast surface processed')

#generating the wind speeds
w10_calc = np.sqrt(u10['u10']**2 + v10['v10']**2)
w10 = xr.Dataset(
    {'w10': (('time', 'number', 'year', 'step', 'surface', 'latitude', 'longitude'), w10_calc.data)},
    coords=u10.coords
)
w100_calc = np.sqrt(u100['u100']**2 + v100['v100']**2)
w100 = xr.Dataset({'w100': (('time', 'number', 'year', 'step', 'surface', 'latitude', 'longitude'), w100_calc.data)},
                coords=u100.coords)
w700_calc =  np.sqrt(u700['u']**2 + v700['v']**2)
w700 = xr.Dataset({'w700': (('time', 'number', 'year', 'step', 'surface', 'latitude', 'longitude'), w700_calc.data)},
                 coords=u700.coords)
print('Finished generating wind speeds')

# #### Getting the observations 


u100_obs=sfc_rfc.get_observations_as_xarray()[['u100']]
v100_obs=sfc_rfc.get_observations_as_xarray()[['v100']]
w100_obs_calc= np.sqrt(u100_obs['u100']**2 + v100_obs['v100']**2)

w100_obs_DA = xr.DataArray(
    w100_obs_calc,
    dims=('time', 'number', 'year', 'step', 'surface', 'latitude', 'longitude'),
    coords=u100_obs.coords,
    name='w100_obs'
)

# Now create the Dataset with `w100_obs`
w100_obs = xr.Dataset({'w100_obs': w100_obs_DA})

print('Finished observations')
datasets = [t2m, u10, w10, v10, tcc, u100, w100, v100, z, u700,w700, t, p10fg6, w100_obs, ssr6, ssrd6] 
for i in range(len(datasets)):
    # Convert 'step' from nanoseconds to hours
    datasets[i]['step'] = pd.to_timedelta(datasets[i]['step'], unit='ns').total_seconds() / 3600
    # Select specific time steps
    if i!=10:  #remove first forecasting step because it is absent in the processed variables 
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

rfc_all = xr.merge(datasets[0:17])

# Verkrijg alle variabelen in de dataset
variable_names = list(rfc_all.variables)

# Print enkel de namen van de variabelen
Print(variable_names)