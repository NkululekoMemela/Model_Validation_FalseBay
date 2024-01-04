#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:35:41 2024

@author: nkululeko
"""

import netCDF4 as nc
# Open the NetCDF file
nc_file = nc.Dataset('../OutPut_05_Jan_24.nc', 'r')

# List all variables in the NetCDF file
print("Variables in the NetCDF file:")
print(nc_file.variables.keys())

# Iterate over all variables in the NetCDF file
for variable_name in nc_file.variables.keys():
    # Access the variable data
    variable_data = nc_file.variables[variable_name][:]
    
    # Print the top 5 data points for each variable
    print(f"Top 5 data points for variable '{variable_name}':")
    print(variable_data[:5])
    print()

# Close the NetCDF file
nc_file.close()