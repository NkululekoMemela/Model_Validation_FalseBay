#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:50:44 2024

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

def calculate_anomalies_and_climatology(data):
    anomalies = data.groupby("time.month") - data.groupby("time.month").mean("time")
    climatology = data.groupby("time.month").mean("time")
    return anomalies, climatology

def calculate_correlation_and_rmse(observed, model):
    observed_values = observed[~np.isnan(observed)]
    model_values = model[~np.isnan(observed)]
    correlation = np.corrcoef(np.array(observed_values), np.array(model_values.values))[1][0]
    rmse = sqrt(mean_squared_error(observed_values, model_values))
    return correlation, rmse

def plot_station_points(ax):
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND, zorder=0)
    ax.set_extent([16, 20, -32, -35])
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0, color='black', draw_labels=True)
    gl.left_labels = True
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = True
    gl.ylines = True
    return ax

def plot_model_evaluation(observed, model, title, save_path, anom_corr=None, anom_rmse=None):
    savename = f"{title.replace(' ', '_').lower()}.png"
    fig, ax = plt.subplots(figsize=(12, 15), nrows=3, ncols=1)

    ax[0].plot(observed.time, observed, label='Observed', color='blue')
    ax[0].plot(model.time, model, '--', label='Model', color='red', linewidth=2.5, linestyle='-')
    ax[0].text(0.01, 0.95, f'\n Model r={anom_corr}\n RMSE={anom_rmse}', fontsize=16,
               horizontalalignment='left', verticalalignment='center', transform=ax[0].transAxes)
    ax[0].set_title(title, fontsize=18)
    ax[0].legend(loc="upper right")
    ax[0].set_ylabel('Value')

    ax[1].plot(observed.time, observed, label='Observed', color='blue', linewidth=2.5, linestyle='--')
    ax[1].plot(model.time, model, '--', label='Model', color='red', linewidth=2.5, linestyle='--')
    ax[1].text(0.01, 0.95, f'\n Model r={anom_corr}\n RMSE={anom_rmse}', fontsize=16,
               horizontalalignment='left', verticalalignment='center', transform=ax[1].transAxes)
    ax[1].set_title(f'{title} Anomaly', fontsize=18)
    ax[1].legend(loc="upper right")
    ax[1].set_ylabel('Value')

    ax[2].plot(observed.groupby("time.month").mean("time"), label='Observed', color='blue', linewidth=2.5, linestyle='-')
    ax[2].plot(model.groupby("time.month").mean("time"), '--', label='Model', color='red', linewidth=2.5, linestyle='-')
    ax[2].set_title(f'{title} Climatology', fontsize=18)
    ax[2].legend(loc="upper right")
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Value')

    fig.tight_layout()
    plt.savefig(os.path.join(save_path, savename))

def create_statistical_table(data, save_path):
    savename_tbl = 'model_evaluation_statistics_tbl.png'
    table_df = pd.DataFrame(data)
    table_df.index = table_df.index + 1

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    tbl = table(ax, table_df, loc='center', cellLoc='left', colWidths=[0.4, 0.4], rowLoc='center', fontsize=12)

    for i, key in enumerate(tbl.get_celld().keys()):
        cell = tbl[key]
        cell.set_fontsize(12)
        cell.set_height(0.07)

    plt.title("Model Evaluation Statistics", fontsize=14, y=0.75)
    fig.savefig(os.path.join(save_path, savename_tbl))
    plt.show()

def analyze_and_visualize_data(dataset_path, savepath):
    ds = xr.open_dataset(dataset_path)

    time_variable = ds.time
    data_model = ds.data_model.squeeze()
    data_obs_model_timeaxis = ds.data_obs_model_timeaxis.squeeze()

    anomalies_model, climatology_model = calculate_anomalies_and_climatology(data_model)
    anomalies_obs, climatology_obs = calculate_anomalies_and_climatology(data_obs_model_timeaxis)

    correlation_model, rmse_model = calculate_correlation_and_rmse(data_obs_model_timeaxis, data_model)
    correlation_anom_model, rmse_anom_model = calculate_correlation_and_rmse(anomalies_obs.values, anomalies_model.values)

    # Station Points at dep
    fig = plt.figure(figsize=(10, 5), facecolor='white')
    ax = plt.axes(projection=ccrs.PlateCarree())
    plot_station_points(ax)
    plt.plot(18.8006, -34.35667, 'or', label='In situ')
    plt.plot(data_model.longitude, data_model.latitude, 'oy', label='CROCO model')
    plt.legend()
    plt.title('Position of in situ data')
    plt.savefig(os.path.join(savepath, 'Timeseries_location.png'))

    # Model evaluation
    model_eval_data = {
        'Correlation Model-Obs': correlation_model,
        'Std Dev Model': climatology_model.std().values,
        'Std Dev Obs-Model': (climatology_obs - climatology_model).std().values,
        'RMSE Model-Obs': rmse_model,
        'Total Bias': (data_model - data_obs_model_timeaxis).mean().values
    }

    plot_model_evaluation(data_obs_model_timeaxis, data_model, 'SCC_model and ATAP in situ SST', savepath, correlation_model, rmse_model)
    plot_model_evaluation(anomalies_obs, anomalies_model, 'SCC_model and ATAP in situ SST Anomaly', savepath, correlation_anom_model, rmse_anom_model)
    plot_model_evaluation(climatology_obs, climatology_model, 'SCC_model and ATAP in situ SST Climatology', savepath)

    create_statistical_table(model_eval_data, savepath)

if __name__ == "__main__":
    dataset_path = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/scripts/OutPut_Memela10!_Jan_24.nc'
    savepath = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/scripts/'
    analyze_and_visualize_data(dataset_path, savepath)
    
    