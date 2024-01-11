#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:42:57 2024

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

def analyze_and_visualize_data(dataset_path, savepath):
    ds = xr.open_dataset(dataset_path)

    time_variable = ds.time
    data_model = ds.data_model.squeeze()
    data_obs_model_timeaxis = ds.data_obs_model_timeaxis.squeeze()

    # Anomalies and clim
    model_ano = data_model.groupby("time.month") - data_model.groupby("time.month").mean("time")
    model_clim =  data_model.groupby("time.month").mean("time")

    sst_ano =  data_obs_model_timeaxis.groupby("time.month") - data_obs_model_timeaxis.groupby("time.month").mean("time")
    sst_clim =  data_obs_model_timeaxis.groupby("time.month").mean("time")

    # Correlation and rmse
    temp_corr = data_obs_model_timeaxis[~np.isnan(data_obs_model_timeaxis)]
    model_corr = data_model[~np.isnan(data_obs_model_timeaxis)]

    model_in_situ_corr = np.corrcoef(np.array(temp_corr), np.array(model_corr.values))[1][0]
    model_in_situ_rmse = sqrt(mean_squared_error(temp_corr, model_corr)) 

    # Correlation and rmse ano
    sst_ano_reshaped = sst_ano.values.reshape(model_ano.shape)

    temp_ano_corr = sst_ano.values[~np.isnan(sst_ano.values)]
    model_ano_corr = model_ano.values[~np.isnan(sst_ano_reshaped)]

    model_in_situ_ano_corr = np.corrcoef(temp_ano_corr, model_ano_corr)[1][0]
    model_in_situ_ano_rmse = sqrt(mean_squared_error(temp_ano_corr, model_ano_corr)) 

    # Station Points at dep
    fig = plt.figure(figsize=(10, 5), facecolor='white')
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    plt.plot(18.8006, -34.35667, 'or', label='In situ')
    plt.plot(data_model.longitude, data_model.latitude, 'oy', label='CROCO model')
    plt.legend()
    plt.title('Position of in situ data') 
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND, zorder=0)
    ax.set_extent([16, 20, -32, -35])
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0, color='black', draw_labels=True)
    gl.left_labels = True
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = True
    gl.ylines = True
    plt.savefig(os.path.join(savepath, 'Timeseries_location.png'))
    
    #  Model evaluation 
    savename = 'Model_evaluation.png'
    fig, ax = plt.subplots(figsize=(12, 15),nrows=3, ncols=1)

    ax[0].plot(time_variable ,data_obs_model_timeaxis,label='ATAP in situ', color='blue');
    ax[0].plot(data_model.time,data_model,'--',label='SCC_model', color='red', linewidth=2.5, linestyle='-');
    ax[0].text(0.01, 0.95, '\n SCC_model r='+str(np.round(model_in_situ_corr,2)) +'\n RMSE='+str(np.round(model_in_situ_rmse,2))
                , fontsize=16,
                horizontalalignment='left', verticalalignment='center', transform=ax[0].transAxes)
    ax[0].set_title('SCC_model and '+str(dataset_path.split("/")[-1].split("_")[0])+'\n ATAP in situ  SST ',fontsize=18)
    ax[0].legend(loc="upper right")
    ax[0].set_ylabel('SST') 

    ax[1].plot(time_variable,sst_ano_reshaped,label='ATAP in situ Anomaly', color='blue', linewidth=2.5, linestyle='--');
    ax[1].plot(data_model.time,model_ano,'--',label='SCC_model Anomaly', color='red', linewidth=2.5, linestyle='--');
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
    plt.savefig(savepath+savename)

    # Creating a statistical table
    savename_tbl = 'Model_evaluation_statistics_Tbl.png'
    table_data = {
        'Statistic': ['Correlation Model-Obs', 'Std Dev Model', 'Std Dev Obs-Model', 'RMSE Model-Obs', 'Total Bias'],
        'Value': [ds.correlation_model_obs, ds.std_dev_model, ds.std_dev_obs_model, ds.rmse_model_obs, ds.total_bias]
    }

    table_df = pd.DataFrame(table_data)
    table_df.index = table_df.index + 1

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    tbl = table(ax, table_df, loc='center', cellLoc='left', colWidths=[0.4, 0.4], rowLoc='center', fontsize=12)

    for i, key in enumerate(tbl.get_celld().keys()):
        cell = tbl[key]
        cell.set_fontsize(12)
        cell.set_height(0.07)  

    plt.title("Table 1: Model Evaluation Statistics over ATAP Data in False Bay", fontsize=14, y=0.75)
    fig.savefig(os.path.join(savepath, savename_tbl), bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    dataset_path = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/scripts/OutPut_Memela10!_Jan_24.nc'
    savepath = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/scripts/'
    analyze_and_visualize_data(dataset_path, savepath)