#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:42:07 2024

@author: nkululeko
"""
import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature

# Specify the directory where your NetCDF files are located
directory_path = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/model_validation/'
savepath = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/model_validation/'
# Get a list of all NetCDF files in the directory
netcdf_files = [file for file in os.listdir(directory_path) if file.endswith('.nc')]

# Initialize lists to store latitude and longitude for all stations
all_latitudes_insitu = []
all_longitudes_insitu = []
all_latitudes_on_model = []
all_longitudes_on_model = []
all_filenames = []

# Iterate through each NetCDF file
for filename in netcdf_files:
    file_path = os.path.join(directory_path, filename)

    # Open the NetCDF file
    ds = xr.open_dataset(file_path)

    # Extract latitude and longitude from the NetCDF file
    latitude_insitu = ds.latitude.values  # Replace with the correct coordinate name
    longitude_insitu = ds.longitude.values  # Replace with the correct coordinate name
    latitude_on_model = ds.latitude_on_model.values
    longitude_on_model = ds.longitude_on_model.values

    # Append the latitude and longitude to the lists
    all_latitudes_insitu.extend(latitude_insitu)
    all_longitudes_insitu.extend(longitude_insitu)
    all_latitudes_on_model.extend(latitude_on_model)
    all_longitudes_on_model.extend(longitude_on_model)
    all_filenames.extend([filename[-8:-3]] * len(latitude_insitu))  # Repeat the filename for each station

# Plot all stations
fig = plt.figure(figsize=(8, 7), facecolor='white')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

# Define colors for each station
colors_insitu = iter(plt.cm.tab10.colors)
colors_on_model = iter(plt.cm.tab10.colors)

# Plot in situ and model data
for i, (lat_insitu, lon_insitu, lat_on_model, lon_on_model, filename) in enumerate(
    zip(all_latitudes_insitu, all_longitudes_insitu, all_latitudes_on_model, all_longitudes_on_model, all_filenames)
):
    color_insitu = next(colors_insitu)
    color_on_model = next(colors_on_model)

    # Get depth value for the specific station
    station_depth = ds.sel(latitude=lat_insitu, longitude=lon_insitu).depth.item()  # Replace with the correct coordinate names

    plt.plot(lon_insitu, lat_insitu, 'o', color=color_insitu, label=f'{filename} - In situ: $={station_depth}$ (m) ')
    plt.plot(lon_on_model, lat_on_model, 's', color=color_on_model, label=f'{filename} - CROCO ')

# Customize the plot as needed
plt.title('Position of in situ and model data validated')
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, zorder=0)
ax.set_extent([18, 19.5, -34, -35])


# Add legend
# legend_text_insitu = r'$\bf{Sensor\ Depth}$' + f' $=\\bf{{{station_depth}}}$ (m) - In situ'
# legend_text_on_model = r'$\bf{Sensor\ Depth}$' + f' $=\\bf{{{ds.depth}}}$ (m) - CROCO'
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(plt.Line2D([0], [0], marker='None', linestyle='None'))  # Adding an empty entry for spacing
# labels.append(legend_text_insitu)
handles.append(plt.Line2D([0], [0], marker='None', linestyle='None'))  # Adding another empty entry for spacing
# labels.append(legend_text_on_model)
plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=11)

# Add gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0, color='black', draw_labels=True)
gl.left_labels = True
gl.top_labels = False
gl.right_labels = False
gl.xlines = True
gl.ylines = True

# Save the plot
plt.savefig(os.path.join(savepath, 'all_stations_map.png'))

# Show the plot
plt.show()