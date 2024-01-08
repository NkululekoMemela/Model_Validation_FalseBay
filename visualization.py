#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 00:49:11 2024

@author: nkululeko
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:35:41 2024

@author: nkululeko
"""

import xarray as xr
import numpy as np
from datetime import datetime

# Open the NetCDF file with xarray
nc_file = xr.open_dataset('../OutPut_Gambu_Jan_24.nc')

# Display dataset information
print("Dataset information:")
print(nc_file)

# Iterate over all variables in the dataset
for variable_name in nc_file.variables:
    # Access the variable
    variable = nc_file[variable_name]

    # Print variable name
    print(f"\nVariable: {variable_name}")

    # Print variable attributes
    for attribute_name in variable.attrs:
        attribute_value = variable.attrs[attribute_name]
        print(f"  Attribute: {attribute_name}, Value: {attribute_value}")

    # Access the variable data
    variable_data = variable.values

    # Convert time variable to datetime if it exists
    if variable_name == 'time':
        # Assuming time is stored as timedelta64
        datetime_data = [np.datetime64('1970-01-01-00:00:00')  for float_time in variable_data]
        print(f"Top 5 data points for variable '{variable_name}':")
        print(datetime_data[:5])

        # Convert time variable to datetime64[s]
        variable_data_datetime64 = np.array([np.datetime64('1970-01-01-00:00:00')  for float_time in variable_data])
        nc_file[variable_name] = variable_data_datetime64

    else:
        # Print the top 5 data points for other variables
        print(f"Top 5 data points for variable '{variable_name}':")
        print(variable_data[:5])

# Extract global attributes
std_dev_model = nc_file.attrs.get('std_dev_model', None)
std_dev_obs_model = nc_file.attrs.get('std_dev_obs_model', None)
correlation_model_obs = nc_file.attrs.get('correlation_model_obs', None)
rmse_model_obs = nc_file.attrs.get('rmse_model_obs', None)
total_bias = nc_file.attrs.get('total_bias', None)

# Print the extracted values
print(f'\nstd_dev_model: {std_dev_model}')
print(f'std_dev_obs_model: {std_dev_obs_model}')
print(f'correlation_model_obs: {correlation_model_obs}')
print(f'rmse_model_obs: {rmse_model_obs}')
print(f'total_bias: {total_bias}')

# Close the xarray dataset
nc_file.close()



# %%
import os
import xarray as xr 
import netCDF4 as nc
from datetime import datetime

# %% DAta import
output_directory = "/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/scripts/"
fname_out = "OutPut_Gambu_Jan_24.nc"
# output_path = os.path.join(output_directory, fname_out)
model_insitu_dataset = xr.open_dataset(output_directory+fname_out)



#Loading lon and lat 
lon = model_insitu_dataset.longitude
lat = model_insitu_dataset.latitude

eval_time = model_insitu_dataset.time
eval_time

datetime_data = [eval_time.datetime.fromtimestamp(float_time) for float_time in model_insitu_dataset]
datetime_data
# %%  closest model point to SWC-CROCO

#Station Points at dep

# This is an x-array conversion method
ds_model = xr.DataArray(temp_depth, coords=[model_time, lat_croco, lon_croco], dims=["time", "lat","lon"])
# This is an output of the above conversion and it captures the closest [point between model and observation
model_pp = ds_model.sel(lon=18.8,lat=-34.4, method= 'nearest')


# %% now I'm bringing in the In situ dataset  

dataset_path = '/home/nkululeko/croco/FalseBaydata_FB001.nc'
ds = xr.open_dataset(dataset_path)

# Cleaning out duplicates from the dataset
ds_clean = ds.drop_duplicates(dim="time")

# Confirming the success of this cleaning effort
if len(set(ds_clean.time.values)) == len(ds_clean.time.values):
    print("success")
else:
    print("duplicate found")
    
ds_sst = ds_clean.temperature.sel(time=slice(start_date_in_situ,end_date_in_situ))
ds_sst_daily = ds_sst.resample(time="1D").mean()
#Remove the following line as it is redundent. It converts to an xarray data that is already in xarray.
#ds_InSitu = xr.DataArray(ds.temperature, coords=[ds.time, ds.longitude, ds.latitude], dims=["time", "longitude","latitude"])

#SST
temp = np.squeeze(ds_sst.values)
temp_daily = np.squeeze(ds_sst_daily.values)
time = ds_sst.time
time_daily = ds_sst_daily.time


# %% Anomalies and clim

model_pp = model_pp

model_ano = model_pp - model_pp.mean("time")
model_ano = model_ano
model_clim =  model_pp.mean("time")

sst_ano =  ds_sst_daily - ds_sst_daily.mean("time")
sst_clim =  ds_sst_daily.mean("time")

# %% Correltation and rmse 
#loading values for dask

temp_corr = temp_daily[~np.isnan(temp_daily)]
model_corr = model_pp[~np.isnan(temp_daily)]

model_in_situ_corr = np.corrcoef(np.array(temp_corr),np.array(model_corr.values))[1][0]
model_in_situ_rmse = sqrt(mean_squared_error(temp_corr, model_corr)) 

# %% Correlation and rmse ano

sst_ano_reshaped = sst_ano.values.reshape(model_ano.shape)

temp_ano_corr = sst_ano.values[~np.isnan(sst_ano.values)]
model_ano_corr = model_ano.values[~np.isnan(sst_ano_reshaped)]

model_in_situ_ano_corr = np.corrcoef(temp_ano_corr,model_ano_corr)[1][0]
model_in_situ_ano_rmse = sqrt(mean_squared_error(temp_ano_corr, model_ano_corr)) 


#%%

fig = plt.figure(figsize=(10, 5),facecolor='white')

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

#plt.contourf(model_dataset.lon_rho,model_dataset.lat_rho,model_dataset.temp[0,0,:,:],transform=ccrs.PlateCarree());
plt.plot(18.8006,-34.35667,'or',label='In situ')
plt.plot(model_pp.lon,model_pp.lat,'oy',label='CROCO model')
#plt.plot(model_glorys_pp.lonT,model_glorys_pp.latT,'ob',label='GLORYS model')
plt.legend()

plt.title('Position of in situ data') 
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, zorder=0)
ax.set_extent([16, 20, -32, -35])
#ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

gl = ax.gridlines(crs=ccrs.PlateCarree(),linewidth=0, color='black', draw_labels=True)
gl.left_labels = True
gl.top_labels = False
gl.right_labels=False
gl.xlines = True
gl.ylines = True

#plt.savefig(savepath+'Timeseries_location.png')
       
# %%

fig, ax = plt.subplots(figsize=(12, 15),nrows=3, ncols=1)
#plt.savefig(savepath+'Postion_of_timeseries_HYCOM.png')

ax[0].plot(time_daily ,temp_daily,label='ATAP in situ');
ax[0].plot(model_pp.time,model_pp,'--',label='SCC_model', color='red', linewidth=2.5, linestyle='-');
ax[0].text(0.01, 0.95, '\n SCC_model r='+str(np.round(model_in_situ_corr,2)) +'\n RMSE='+str(np.round(model_in_situ_rmse,2))
            , fontsize=16,
            horizontalalignment='left', verticalalignment='center', transform=ax[0].transAxes)
ax[0].set_title('SCC_model and '+str(dataset_path.split("/")[-1].split("_")[0])+'\n ATAP in situ  SST ',fontsize=18)
ax[0].legend(loc="upper right")
ax[0].set_ylabel('SST') 

ax[1].plot(time_daily,sst_ano_reshaped,label='ATAP in situ Anomaly', color='blue', linewidth=2.5, linestyle='--');
ax[1].plot(model_pp.time,model_ano,'--',label='SCC_model Anomaly', color='red', linewidth=2.5, linestyle='--');
ax[1].text(0.01, 0.95, '\n SCC_model r='+str(np.round(model_in_situ_ano_corr,2))+'\n RMSE='+str(np.round(model_in_situ_ano_rmse,2))
            ,fontsize=16,
            horizontalalignment='left', verticalalignment='center', transform=ax[1].transAxes)
ax[1].set_title('SCC_model and '+str(dataset_path.split("/")[-1].split("_")[0])+'\n ATAP in situ SST anomaly',fontsize=18)
ax[1].legend(loc="upper right")
ax[1].set_ylabel('SST') 

ax[2].plot(sst_clim,label='ATAP in situ', color='blue', linewidth=2.5, linestyle='-');
ax[2].plot(model_clim,'--',label='SCC_model', color='red', linewidth=2.5, linestyle='-');
ax[2].set_title('SCC_model and '+str(dataset_path.split("/")[-1].split("_")[0])+' \n ATAP in situ SST climatology ',fontsize=18)
ax[2].legend(loc="upper right")
ax[2].set_xlabel('Time')
ax[2].set_ylabel('SST') 

fig.tight_layout()
#plt.savefig(savepath+savename)
