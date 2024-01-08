#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 22:22:08 2024

@author: nkululeko
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import glob

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Open the NetCDF file
nc_file = nc.Dataset('../OutPut_Memela4!_Jan_24.nc', 'r')

# Access the time variable
time_variable = nc_file.variables['time']
datetime_data = [datetime.fromtimestamp(float_time) for float_time in time_variable[:]]

# Squeeze the dimensions for data_model and data_obs_model_timeaxis
data_model = np.squeeze(nc_file.variables['data_model'][:])
data_obs_model_timeaxis = np.squeeze(nc_file.variables['data_obs_model_timeaxis'][:])

# Mask gaps in the timeline for data_model
mask_model = np.isnan(data_model)
masked_data_model = np.ma.masked_array(data_model, mask=mask_model)

# Mask gaps in the timeline for data_obs_model_timeaxis
mask_obs_model = np.isnan(data_obs_model_timeaxis)
masked_data_obs_model_timeaxis = np.ma.masked_array(data_obs_model_timeaxis, mask=mask_obs_model)

# Plot time series for data_model
plt.figure()
plt.plot(datetime_data, masked_data_model, label='data_model')
plt.plot(datetime_data, masked_data_obs_model_timeaxis, label='data_obs_model_timeaxis')
plt.ylabel('Temperature')
plt.title('Time series of data_model and insitu observations')
plt.legend()
plt.show()

# Extract global attributes
std_dev_model = nc_file.getncattr('std_dev_model')
std_dev_obs_model = nc_file.getncattr('std_dev_obs_model')
correlation_model_obs = nc_file.getncattr('correlation_model_obs')
rmse_model_obs = nc_file.getncattr('rmse_model_obs')
total_bias = nc_file.getncattr('total_bias')

# Print the extracted values
print(f'std_dev_model: {std_dev_model}')
print(f'std_dev_obs_model: {std_dev_obs_model}')
print(f'correlation_model_obs: {correlation_model_obs}')
print(f'rmse_model_obs: {rmse_model_obs}')
print(f'total_bias: {total_bias}')

# Close the NetCDF file
nc_file.close()