#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:35:41 2024

@author: nkululeko
"""

import netCDF4 as nc
from datetime import datetime

# Open the NetCDF file
nc_file = nc.Dataset('../OutPut_Memela6!_Jan_24.nc', 'r')

# List all variables in the NetCDF file
print("Variables in the NetCDF file:")
print(nc_file.variables.keys())

# Iterate over all variables in the NetCDF file
for variable_name in nc_file.variables.keys():
    # Access the variable
    variable = nc_file.variables[variable_name]

    # Print variable name
    print(f"Variable: {variable_name}")

    # Print variable attributes
    for attribute_name in variable.ncattrs():
        attribute_value = getattr(variable, attribute_name)
        print(f"  Attribute: {attribute_name}, Value: {attribute_value}")

    # Access the variable data
    variable_data = variable[:]

    # Convert time variable to datetime if it exists
    if variable_name == 'time':
        # Assuming time is stored as float timestamps
        datetime_data = [datetime.fromtimestamp(float_time) for float_time in variable_data]
        print(f"Top 5 data points for variable '{variable_name}':")
        print(datetime_data[:5])
    else:
        # Print the top 5 data points for other variables
        print(f"Top 5 data points for variable '{variable_name}':")
        print(variable_data[:5])

    print()
    
    

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
        