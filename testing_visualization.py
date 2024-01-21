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
from sklearn.metrics import mean_squared_error
from math import sqrt
import netCDF4 as nc
from datetime import datetime
import cartopy.crs as ccrs
import cartopy
import pandas as pd
from pandas.plotting import table
import calendar

# %%
filename = 'Validation_'+'CapePoint_CP003.nc'
savepath = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/model_validation/'
# dataset_path = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/scripts/OutPut_Memela10!_Jan_24.nc'
dataset_path = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/model_validation/'
ds = xr.open_dataset(dataset_path+filename)

time_variable = ds.time
data_model = ds.data_model.squeeze()
data_obs_model_timeaxis = ds.data_obs_model_timeaxis.squeeze()

print(sum(data_model)/len(data_model))
print(sum(data_obs_model_timeaxis)/len(ds.data_obs_model_timeaxis.values.squeeze()))

latitude_insitu = ds.latitude_insitu.values
latitude_insitu
latitude_on_model = ds.latitude_on_model.values
latitude_on_model

# %% Anomalies and clim

model_ano = data_model.groupby("time.month") - data_model.groupby("time.month").mean("time")
model_clim =  data_model.groupby("time.month").mean("time")

obs_ano =  data_obs_model_timeaxis.groupby("time.month") - data_obs_model_timeaxis.groupby("time.month").mean("time")
obs_clim =  data_obs_model_timeaxis.groupby("time.month").mean("time")

# %% Correltation and rmse 
#loading values for dask

temp_corr = data_obs_model_timeaxis[~np.isnan(data_obs_model_timeaxis)]
model_corr = data_model[~np.isnan(data_obs_model_timeaxis)]

model_in_situ_corr = np.corrcoef(np.array(temp_corr),np.array(model_corr.values))[1][0]
model_in_situ_rmse = sqrt(mean_squared_error(temp_corr, model_corr)) 

# %% Correlation and rmse ano

obs_ano_reshaped = obs_ano.values.reshape(model_ano.shape)

temp_ano_corr = obs_ano.values[~np.isnan(obs_ano.values)]
model_ano_corr = model_ano.values[~np.isnan(obs_ano_reshaped)]

#The following corrcoef function is based on the linked documentation
#https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
model_in_situ_ano_corr = np.corrcoef(temp_ano_corr,model_ano_corr)[1][0]
model_in_situ_ano_rmse = sqrt(mean_squared_error(temp_ano_corr, model_ano_corr)) 


#%%
#Station Points at dep

fig = plt.figure(figsize=(8, 7),facecolor='white')

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

#plt.contourf(model_dataset.lon_rho,model_dataset.lat_rho,model_dataset.temp[0,0,:,:],transform=ccrs.PlateCarree());
# plt.plot(18.8006,-34.35667,'or',label='In situ')
# plt.plot(data_model.longitude,data_model.latitude,'oy',label='CROCO model')
plt.plot(ds.longitude_insitu ,ds.latitude_insitu,'or',label='In situ')
plt.plot(ds.longitude_on_model,ds.latitude_on_model,'oy',label='CROCO model')
#plt.plot(model_glorys_pp.lonT,model_glorys_pp.latT,'ob',label='GLORYS model')

# Add custom legend entry with sensor depth
# legend_text = f'Sensor Depth={ds.depth.values}'
legend_text = r'$\bf{Sensor\ Depth}$' + f' $=\\bf{{{ds.depth}}}$ (m)'
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(plt.Line2D([0], [0], marker='None', linestyle='None'))  # Adding an empty entry for spacing
labels.append(legend_text)
plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1), fontsize=11)

plt.title('Position of in situ and model data validated') 
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

plt.savefig(savepath+filename[:-3]+'_'+ 'map')
       
# %% Model evaluation 
# savename = 'Model_evaluation.png'
fig, ax = plt.subplots(figsize=(12, 15),nrows=3, ncols=1)

# Convert the integer month values to month names
month_names = [calendar.month_abbr[i] for i in range(1, 13)]

# Update the 'month' coordinate with month names
obs_clim['month'] = month_names
model_clim['month'] = month_names

plt.title(filename,fontsize=22)

ax[0].plot(time_variable ,data_obs_model_timeaxis,label='ATAP in situ', color='blue');
ax[0].plot(data_model.time,data_model,'--',label='SCC_model', color='red', linewidth=2.5, linestyle='-');
ax[0].text(0.01, 0.95, '\n SCC_model r='+str(np.round(model_in_situ_corr,2)) +'\n RMSE='+str(np.round(model_in_situ_rmse,2))
            , fontsize=16,
            horizontalalignment='left', verticalalignment='center', transform=ax[0].transAxes)
ax[0].set_title('SCC_model and '+str(dataset_path.split("/")[-1].split("_")[0])+'\n ATAP in situ  SST ',fontsize=18)
ax[0].legend(loc="upper right")
ax[0].set_xlabel('Time (days)')
ax[0].set_ylabel('SST (°C)') 

ax[1].plot(time_variable,obs_ano_reshaped,label='ATAP in situ Anomaly', color='blue', linewidth=2.5, linestyle='--');
ax[1].plot(data_model.time,model_ano,'--',label='SCC_model Anomaly', color='red', linewidth=2.5, linestyle='--');
ax[1].text(0.01, 0.95, '\n SCC_model r='+str(np.round(model_in_situ_ano_corr,2))+'\n RMSE='+str(np.round(model_in_situ_ano_rmse,2))
            ,fontsize=16,
            horizontalalignment='left', verticalalignment='center', transform=ax[1].transAxes)
ax[1].set_title('SCC_model and '+str(dataset_path.split("/")[-1].split("_")[0])+'\n ATAP in situ SST anomaly',fontsize=18)
ax[1].legend(loc="upper right")
ax[1].set_xlabel('Time (days)')
ax[1].set_ylabel('SST Anomaly') 

ax[2].plot(obs_clim.month,obs_clim,label='ATAP in situ', color='blue', linewidth=2.5, linestyle='-');
ax[2].plot(model_clim.month,model_clim,'--',label='SCC_model', color='red', linewidth=2.5, linestyle='-');
ax[2].set_title('SCC_model and '+str(dataset_path.split("/")[-1].split("_")[0])+' \n ATAP in situ SST climatology ',fontsize=18)
ax[2].legend(loc="upper right")
ax[2].set_xlabel('Time (months)')
ax[2].set_ylabel('SST (°C)') 



fig.tight_layout()
plt.savefig(savepath+filename[:-3]+'_'+ 'Timeseries_location.png')


# %% Creating a statistical table
savename_tbl = 'stats_table.png'
# Create a pandas DataFrame
table_data = {
    'Statistic': ['Depth','Correlation', 'Std Dev Model', 'Std Dev Obs', 'RMSE ', 'Total Bias', 'Total Bias', 'Model mean', 'observations mean', 'observations min', 'Model max', 'observations max'],
    'Value': [ds.depth,round(ds.correlation_model_obs,3), ds.std_dev_model, round(ds.std_dev_obs_model,2), ds.rmse_model_obs, ds.total_bias,ds.model_mean ,ds.obs_mean ,ds.model_min ,ds.obs_min ,ds.model_max ,round(ds.obs_max,2) ]
}

table_df = pd.DataFrame(table_data)

# Update the DataFrame index to start at 1
table_df.index = table_df.index + 1

# Increase figure size for better readability
fig, ax = plt.subplots(figsize=(8, 7))

# Plot the DataFrame as a table with left-aligned text and wider columns
ax.axis('off')
tbl = table(ax, table_df, loc='center', cellLoc='left', colWidths=[0.4, 0.4], rowLoc='center', fontsize=12)

# Adjust the row heights
for i, key in enumerate(tbl.get_celld().keys()):
    cell = tbl[key]
    cell.set_fontsize(12)
    cell.set_height(0.07)  # Adjust the height as needed
    
    # Add a title to the table
# plt.title("Table 1: Model Evaluation Statistics over ATAP Data in False Bay", fontsize=14, y=0.75)

# Save the figure as a PNG file
# fig.savefig('output_table.png', bbox_inches='tight', dpi=300)  # Increased DPI for better quality

# Display the table
plt.savefig(savepath+filename[:-3]+'_'+savename_tbl)
plt.show()

