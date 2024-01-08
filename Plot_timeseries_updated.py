#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:43:04 2023
@author: nkululeko
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import glob

# Directory containing NetCDF files
dataDir = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/scripts/'
# Initialize empty arrays to store hourly data and daily means
filenames = sorted(glob.glob('OutPut_Memela3!_Jan_24.nc')) #load multiple files
filenames = filenames[:]

# Set the figure size (adjust these values as needed)
fig_width = 8  # Adjust the width as needed
fig_height = 5  # Adjust the height as needed

# Loop through the NetCDF files in the directory
for filename in filenames:
    print(filename)
    if filename.endswith('.nc'):
        file_path = os.path.join(dataDir, filename)
        ds = xr.open_dataset(file_path)
        ds=ds.sortby("time")
        
        # Assuming 'temperature' is the variable name 'temp'
        hourly_temperatures = ds.temp
        
        plt.figure(figsize=(fig_width, fig_height))  # Set the figure size        
        # Calculate daily means and append to the daily_means array
        daily_mean = ds.temperature.resample(time="D").mean()      
        # Create a time array for the hourly data
        time = np.arange(len(hourly_temperatures))
        #time = hourly_temperatures.extend(ds['temp'].values)
        
        # Create a scatter plot for hourly temperature readings
        hourly_temperatures.plot.scatter(label='Hourly Temperatures',color="b", s=12,alpha=1)
        daily_mean.plot(label='Hourly Temperatures',color="r",alpha=0.9)

        # Set labels and legend
        plt.xlabel('Time (hours / daily)')
        plt.ylabel('Temperature (°C)')
        plt.legend(['Hourly Observed Temperatures', 'Daily Mean Temperature'])
        plt.title(os.path.basename(filename)[:-3])
        plt.savefig(os.path.basename(filename)[:-3]+'.png')
        plt.close()  # Close the figure to clear legend entries
