#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:39:09 2023

@author: nkululeko
"""

"""
       Libraries:
"""
import numpy as np
import xarray as xr 
from glob import glob
from natsort import natsorted
from datetime import datetime
import sys
directory_path = '/home/nkululeko/somisana/toolkit/cli/applications/croco'
sys.path.append(directory_path)
import postprocess as post
import netCDF4 as nc
import pandas as pd
import os
# import your libraries here
from datetime import datetime,timedelta
# import your libraries here

# %%

"""
The next Function is for matching time axis of both the insitu and model datasets:
    
    D/12- denotes 2 HOURLY  conversion
    D/6 - denotes 4 HOURLY  conversion
    D/4 - denotes 6 HOURLY conversion
    D/2 - denotes half-day conversion
    D - denotes daily conversion
    W - denotes weekly conversion
    M - denotes monthly conversion
"""


def hourly_2_frequency(fname_obs, conversionType):
    print("1. Im in hourly_2_frequency")
    # Load xarray dataset
    obs = xr.open_dataset(fname_obs)
    if conversionType == 'D/12':
        return obs.resample(time="2H").mean()
    if conversionType == 'D/6':
        return obs.resample(time="4H").mean()
    if conversionType == 'D/4':
        return obs.resample(time="6H").mean()
    if conversionType == 'D/2':
        return obs.resample(time="12").mean()
    if conversionType == 'D':
        return obs.resample(time="1D").mean()
    if conversionType == 'W':
        return obs.resample(time="1W").mean()
    if conversionType == 'M':
        return obs.resample(time="1M").mean()
    return obs


"""
       The next Function is for loading IN SITU time series:
"""


def get_ts_obs(fname_obs, var, obs):
    print("2. Im in get_ts_obs")
    # A generic function like this will work if we process all the observations
    # from different sources into the same format
    # Cleaning out duplicates from the dataset
    # obs = post.get_ds(fname_obs)
    # obs = xr.open_dataset(fname_obs)
    data_obs = np.squeeze(obs[var].values)

    # Temperature is the data observation
    time_obs = obs.time.values
    # time_obs = pd.to_datetime(time_obs)

    time_obs = time_obs.astype('datetime64[s]').astype(datetime)

    long_obs = obs.longitude.values
    lat_obs = obs.latitude.values

    return time_obs, data_obs, long_obs, lat_obs


"""
       The next METHOD is for finding the nearest point to insitu data in the model:
"""


def find_nearest_point(fname, Longi, Latit):
    print("3. Im in find_nearest_point")

    # for effeciency we shouldn't use open_mfdataset for this function
    # only use the first file
    if ('*' in fname) or ('?' in fname) or ('[' in fname):
        fname = glob(fname)[0]

    with post.get_ds(fname) as ds:
        # Calculate the distance between (Longi, lat0) and all grid points
        distance = ((ds['lon_rho'].values - Longi) ** 2 +
                    (ds['lat_rho'].values - Latit) ** 2) ** 0.5

    # Find the indices of the minimum distance
    min_index = np.unravel_index(distance.argmin(), distance.shape)
    # min_index is a tuple containing the row and column indices of the minimum value
    j, i = min_index

    return j, i


"""
       The next Function is for getting MODEL time series:
"""

# def get_ts(dir_model,var,depth=-1,model_suffix='',time_lims=[]):


def get_ts(fname, var, lon, lat, ref_date, depth=-1, time_lims=[]):
    print("4. Im in get_ts")
    # this function will eventually get moved into the postprocess.py file of the somisana repo
    # which is why I'd call it just 'get_ts', as it will be obvious that it relates to croco model output
    # you can use xr.open_mfdataset(dir_model+*nc+model_suffix) as you've been doing or rather loop through the correct months
    # time_lims=[datetime(2012,11,1),datetime(2018,11,1)]
    # Convert datetime64 array to datetime.datetime objects

    time_model = post.get_time(fname, ref_date, time_lims=time_lims)

    j, i = find_nearest_point(fname, lon, lat)

    data_model = post.get_var(fname, var,
                              tstep=time_lims,
                              level=depth,
                              eta=j,
                              xi=i,
                              ref_date=ref_date)

    return time_model, data_model


"""
       The next Function is for matching time axis of both the insitu and model datasets:
"""


def obs_2_new_timeaxis(time_model, time_obs, data_obs, time_threshold=timedelta(hours=12)):
    print("5. Im in obs_2_new_timeaxis")

    # obs = post.get_ds(fname_obs)
    obs =  hourly_2_frequency(fname_obs,conversionType="D")
    # model = post.get_ds(dir_model+SelectFiles)
    # data_obs = np.squeeze(obs[var].values)
    # time_obs = obs.time.values
    # get the observations time-series
    time_obs, data_obs, long_obs, lat_obs = get_ts_obs(fname_obs,var,obs)
    # time_model, data_model = get_ts(dir_model,var,depth=depth,time_lims=[time_obs[0],time_obs[-1]])

    # # Merge datasets on the model time axis, keeping only rows where the time values match
    # CombineDateTimes_df = pd.merge(time_model,time_obs,tolerance=time_threshold)

    # # Calculate the average temperature between consequetive observation timesteps
    # # obs['average_temperature'] = (obs['temperature'] + obs['temperature'].shift(-1)) / 2
    # CombineDateTimes_df['average_temperature'] = (data_obs+ data_obs.shift(-1)) / 2

    # # Use the time axis from the first dataset
    # data_obs_model_timeaxis = CombineDateTimes_df['time', 'average_temperature']

    # Approach:
        # Steps
        # Steps -2: Convert the hourly data of time_obs to daily data.
        # Steps -1:
    data_obs_model_timeaxis = [None for i in range(len(time_model))]
        # Step 0: Create data_obs_model_timeaxis and set it to be data_obs
        # Step 1: Loop through the dataset obs.
        
    for obz in time_model:
        
        # Step 2: Check the time component in time_obs and in each record if it is contained in time_obs then.
        if obz in time_obs:            
        # Step 3: If contained then replace that time value in with a value in time_model in the same index.
            index = time_obs.index(obz)
            data_obs_model_timeaxis[index] = data_obs[index]
        # Step 4: Return updated data_obs_model_timeaxis

    return data_obs_model_timeaxis

   

    

"""
       The next Function is for retrieving all datasets from the previous functions and package them into a netCDF file: 
"""
    
def get_model_obs_ts(fname,fname_obs,fname_out,obs,conversionType='D',var='temp',depth=-1,ref_date=None,time_threshold=timedelta(hours=12)):
    print("6. Im in get_model_obs_ts")
    # the output of this function is a netcdf file 'fname_out'
    # which will have the model and observations on the same time axis
    
    obs =  hourly_2_frequency(fname_obs,conversionType)
    
    # get the observations time-series
    time_obs, data_obs, long_obs, lat_obs = get_ts_obs(fname_obs,var,obs)   
    
    # get the model time-series
    time_model, data_model = get_ts(fname,var,long_obs,lat_obs,ref_date,depth=depth,time_lims=[time_obs[0],time_obs[300]]) # Change the 10 back to -1
        
    # get the observations onto the model time axis
    data_obs_model_timeaxis = obs_2_new_timeaxis(time_model, time_obs, data_obs, time_threshold=time_threshold)
    
    
    # Create a NetCDF File
    with nc.Dataset(output_path, 'w', format='NETCDF4') as nc_file:
        # Create dimensions
        nc_file.createDimension('time', len(time_model))
        nc_file.createDimension('latitude', len(lat_obs))
        nc_file.createDimension('longitude', len(long_obs))
        
        print(f"6.1 NetCDF file created at: {output_path}")
        # Create variables
        time_var = nc_file.createVariable('time', 'f8', ('time'))
        lat_var = nc_file.createVariable('latitude', 'f4', ('latitude'))
        lon_var = nc_file.createVariable('longitude', 'f4', ('longitude'))
        model_var = nc_file.createVariable('data_model', 'f4', ('time', 'latitude', 'longitude'))
        obs_model_var = nc_file.createVariable('data_obs_model_timeaxis', 'f4', ('time', 'latitude', 'longitude'))
    
        # Convert datetime objects to Unix timestamps (floats)
        float_time_model = np.array([dt.timestamp() for dt in time_model], dtype=float)
        
        print(float_time_model)
    
        # Assign data to variables
        time_var[:] = float_time_model
        lat_var[:] = lat_obs
        lon_var[:] = long_obs
        model_var[:, :, :] = data_model
        obs_model_var[:, :, :] = data_obs_model_timeaxis
    
        # Add attributes if needed
        time_var.units = 'day'
        lat_var.units = 'latitude'
        lon_var.units = 'longitude'
        model_var.units = 'degrees Celsius'
        obs_model_var.units = 'degrees Celsius'

    # Use the output_path variable as needed
    print(f"NetCDF file created at: {output_path}")
    
    # return data_obs_model_timeaxis

if __name__ == "__main__":
    
    # define what we're validating...
    
    dir_model = '/mnt/d/Run_False_Bay_2008_2018_SANHO/croco_avg_Y2013M*.nc.1'
    fname_obs = '/mnt/d/DATA-20231010T133411Z-003/DATA/ATAP/Processed/Data_Validation/FalseBaydata_FB001.nc'
    fname_out = 'OutPut_04_Jan_24.nc'
    
    # Output file name and directory
    output_directory = "/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/scripts/"
    output_path = os.path.join(output_directory, fname_out)
    
    # SelectFiles = "croco_avg_Y2013M*.nc.1"
    var = 'temp'
    depth=-35
    ref_date = datetime(1990, 1, 1, 0, 0, 0)
    # model_suffix='.1'
    time_threshold=timedelta(hours=12) # used getting observations onto model time axis
    
    get_model_obs_ts(dir_model,fname_obs,
                     fname_out,
                     var,
                     ref_date=ref_date,
                     depth=depth,
                     time_threshold=time_threshold                     
                     )
    # time_obs, data_obs, long_obs, lat_obs = get_ts_obs(fname_obs, 'temperature')
    # j,i = find_nearest_point(dir_model+SelectFiles, long_obs, lat_obs)
    # j,i = find_nearest_point(dir_model+SelectFiles, 19, -35)
    # time_model, data_model = get_ts(dir_model,var,depth=depth,time_lims=[])

    