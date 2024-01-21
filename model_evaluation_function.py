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
from datetime import datetime
import sys
directory_path = '/home/nkululeko/somisana/toolkit/cli/applications/croco'
sys.path.append(directory_path)
import postprocess as post
import netCDF4 as nc
# from netCDF4 import Dataset
# import pandas as pd
import os
# from datetime import datetime,timedelta
# import your libraries here


# %%
"""
       The next Function is for loading IN SITU time series:
           1. obs       : uses the given filename to open the file using xarray funtion
           2. data_obs  : uses obs[var] extract from postprocess stored in somisana toolkit; 
                          it retrieves obs data array
           3. time_obs  : uses the traditional dot method to retrieve time array from obs
                          it is then changed to a datetime64 datatype to ensure compliance with model time 
           4. lat_obs   : uses the dot method to extract latitude values. the same applieas to long_obs
           
"""
def get_ts_obs(fname_obs, var):
    print("2. Im in get_ts_obs")
    # A generic function like this will work if we process all the observations
    # from different sources into the same format
    # Cleaning out duplicates from the dataset
    obs = xr.open_dataset(fname_obs)
    data_obs = np.squeeze(obs[var].values)
    time_obs = obs.time.values
    time_obs = time_obs.astype('datetime64[s]').astype(datetime)
    long_obs = obs.longitude.values
    lat_obs = obs.latitude.values

    return time_obs, data_obs, long_obs, lat_obs

# %%
"""
        The next METHOD is for finding the nearest point to insitu data in the model:
            1. The if statement in this function is intended to apply in cases where there are 
                multiple files to read in the model. It instructs to only read the first one
            2. with post.get_ds(fname) as ds: Calculate the distance between model and insitu lats and lons  
                at all grid points. If you pay special attention you will see that it is indeed a 
                distance formular in the form of d = sqrt(x^2+y^2) expanded to d = sqrt((x1-x2)^2+(y1-y2)^2)
            3. min_index findes indices j,i which represents the minimum distance between model and insitu points
"""

def find_nearest_point(fname, Longi, Latit):
    print("3. Im in find_nearest_point")

    # for effeciency we shouldn't use open_mfdataset for this function
    # only use the first file
    if ('*' in fname) or ('?' in fname) or ('[' in fname):
        fname = glob(fname)[0]

    with post.get_ds(fname) as ds:
        # Calculate the distance between (Longi, Latit) and all grid points
        distance = ((ds['lon_rho'].values - Longi) ** 2 +
                    (ds['lat_rho'].values - Latit) ** 2) ** 0.5

    # Find the indices of the minimum distance
    min_index = np.unravel_index(distance.argmin(), distance.shape)
    # min_index is a tuple containing the row and column indices of the minimum value
    j, i = min_index

    return j, i



# %%
"""
       The next Function is for getting MODEL time series:
           1. time_model is computed using the get_time function from somisana toolkit called postprocess
               this model time is extracted based on time limits set by observations time-span that overlaps 
               the model time span. fname taken in by time model is the model name.
           2. j, i = find_nearest_point(fname, lon, lat) : finds the nearest point in the model to the insitu
               lon, lat extracted from the insitu file.
           3. the i_shifted or j_shifted represents the possible shift a user can physically input in order to 
               offset as the new supposed point of observation so that the model will find a point nearest to 
               it for model evaluation. The reason for this shift is a possible bathymetry mismatch between 
               model and station observation which results in nans if the model is shallower than the insitu
           4. the data_model function call extracts data in the location of the model that is nearest to the 
               insitu dataset based on the j and i indices. In the case where the bathymetry becomes a factor
               as explained above; then the data_model extracts at the shifted indices.
           5. lat_mod and lon_mod are extracted with the aim of using them for plotting model location closest
               to the insitu data with availability of corrasponding z-level to the insitu data later
"""
def get_ts(fname, var, lon, lat, ref_date, depth=-1,i_shifted=0,j_shifted=0, time_lims=[]):
    print("4. Im in get_ts")
    # this function will eventually get moved into the postprocess.py file of the somisana repo
    # which is why I'd call it just 'get_ts', as it will be obvious that it relates to croco model output
    # you can use xr.open_mfdataset(dir_model+*nc+model_suffix) as you've been doing or rather loop through the correct months
    # time_lims=[datetime(2012,11,1),datetime(2018,11,1)]
    # Convert datetime64 array to datetime.datetime objects

    time_model = post.get_time(fname, ref_date, time_lims=time_lims)
    
    # if lat_extract is not None:
    #     j, i = find_nearest_point(fname, lon, lat_extract)
    # else:
    j, i = find_nearest_point(fname, lon, lat)
        
    i = i+i_shifted
    j = j+j_shifted
        
    data_model = post.get_var(fname, var,
                              tstep=time_lims,
                              level=depth,
                              eta=j,
                              xi=i,
                              ref_date=ref_date)
    
    lat_mod =  post.get_var(fname,"lat_rho",eta=j,xi=i)
    lon_mod =  post.get_var(fname,"lon_rho",eta=j,xi=i)
        
    return time_model, data_model,lat_mod,lon_mod

# %%
"""
       The next Function is for matching time axis of both the insitu and model datasets:
"""

def obs_2_new_timeaxis(fname_obs,time_model,model_frequency,var):
    print("5. Im in obs_2_new_timeaxis")
    # My Approach:
        # Steps
        
        #         return 
    # obs = fname_obs.resample(time=model_frequency).mean()
    obs = xr.open_dataset(fname_obs)
    obs_formatted = obs.resample(time=model_frequency, base=12).mean()
    data_obs = np.squeeze(obs_formatted[var].values)
    time_obs = obs_formatted.time.values
    time_obs = time_obs.astype('datetime64[s]').astype(datetime)
        # Step 1: Convert the hourly data of time_obs to daily data. Or more specifically to the model freq.
        # Step 2: Load observation time series after making it daily.
        # Step 3: Create data_obs_model_timeaxis and set it to be data_obs
    data_obs_model_timeaxis = [None for i in range(len(time_model))]
        # Step 4: Loop through the dataset of obs and add thrashold to make it match time_model data. 
        
    # Convert the NumPy array to a list of datetime objects
    formatted_time_obs = time_obs.tolist()
    
    # print(formatted_time_obs)

    for index_mod, time_model_now in enumerate(time_model):
        # Step 5: Check the time component in time_obs and in each record if it is contained in time_obs then.
        if time_model_now in formatted_time_obs:            
        # Step 6: If contained then replace that time value in with a value in time_model in the same index.
            # index_mod = time_model.index(time_model_now)
            # index_obs =  np.where(time_obs==time_model_now)
            index_obs = formatted_time_obs.index(time_model_now)
            data_obs_model_timeaxis[index_mod] = data_obs[index_obs]
       

    return data_obs_model_timeaxis
  
    
# %% Statistical analysis section
decimal = 3
def calculate_rmse(obs_data, model_data):
    # Calculate RMSE, ignoring NaN values
    rmse = round(np.sqrt(np.nanmean((obs_data - model_data)**2)),decimal)
    return rmse

def calculate_correlation(obs_data, model_data):
    # Calculate correlation, ignoring NaN values. This function is not really working but I am on it.
    correlation = round(np.corrcoef(obs_data, model_data)[0, 1],decimal)
    return correlation

def calculate_std_dev(data):
    # Calculate standard deviation, ignoring NaN values
    std_dev = round(np.nanstd(data),decimal)
    return std_dev

def calculate_total_bias(obs_data, model_data):
    # Calculate total bias, ignoring NaN values
    bias = round(np.nanmean(obs_data - model_data),decimal)
    return bias

def calculate_min_mean_max(obs_data, model_data):
    # Calculate model_mean, ignoring NaN values
    model_mean = round(np.nanmean(model_data),decimal)
    # Calculate obs_mean, ignoring NaN values
    obs_mean = round(np.nanmean(obs_data),decimal)
    # Calculate model_min, ignoring NaN values
    model_min = round(np.min(model_data),decimal)
    # Calculate obs_min, ignoring NaN values
    obs_min = round(np.nanmin(obs_data),decimal)
    # Calculate model_max, ignoring NaN values
    model_max = round(np.max(model_data),decimal)
    # Calculate obs_max, ignoring NaN values
    obs_max = round(np.nanmax(obs_data),decimal)
    return model_mean,obs_mean,model_min,obs_min,model_max,obs_max

# def calculate_seasonal_means(obs_data, model_data, time_model):
#     """
#     Calculate seasonal means for JFM, AMJ, JAS, and OND.

#     Parameters:
#     - obs_data: List or NumPy array with observed data and a datetime-like structure
#     - model_data: List or NumPy array with model data and a datetime-like structure
#     - time_model: List or NumPy array indicating the time model for resampling (e.g., [1, 2, 3, 4])

#     Returns:
#     - JFM_mean, AMJ_mean, JAS_mean, OND_mean: Mean values for each season
#     """

#     # Convert lists to NumPy arrays
#     obs_data = np.array(obs_data)
#     model_data = np.array(model_data)
#     time_model = np.array(time_model)

#     # Resample data to custom seasonal frequency and calculate mean for each season
#     obs_means = np.mean(obs_data.reshape(-1, len(time_model)), axis=1)
#     model_means = np.mean(model_data.reshape(-1, len(time_model)), axis=1)

#     # Map numerical indices to season labels
#     season_labels = {1: 'JFM', 2: 'AMJ', 3: 'JAS', 4: 'OND'}

#     # Get mean values for each season
#     JFM_mean = np.mean(obs_means[time_model == 1]), np.mean(model_means[time_model == 1])
#     AMJ_mean = np.mean(obs_means[time_model == 2]), np.mean(model_means[time_model == 2])
#     JAS_mean = np.mean(obs_means[time_model == 3]), np.mean(model_means[time_model == 3])
#     OND_mean = np.mean(obs_means[time_model == 4]), np.mean(model_means[time_model == 4])

#     return JFM_mean, AMJ_mean, JAS_mean, OND_mean


# %%
"""
       The next Function is for retrieving all datasets from the previous functions and package them into a netCDF file: 
"""
    
def get_model_obs_ts(fname,fname_obs,output_path,model_frequency,var,depth=-1,i_shifted=0,j_shifted=0,ref_date=None,lon_extract=None,lat_extract=None):
    print("6. Im in get_model_obs_ts")
    # the output of this function is a netcdf file 'output_path'
    # which will have the model and observations on the same time axis
    
    # get the observations time-series
    time_obs, data_obs, long_obs, lat_obs = get_ts_obs(fname_obs,var)   
    if lon_extract is not None:
        long_obs = lon_extract
    
    if lat_extract is not None:
        # lat_obs = lat_extract
        # Get the data type of 'a'
        # type_of_lat_obs = type(lat_obs)
        # lat_extract = type_of_lat_obs
        
        lat_extract = np.array([lat_extract], dtype=np.float32)


        # # Convert the float to an array of float32
        # lat_extract = np.array(lat_extract, dtype=np.float32)
    # get the model time-series
    time_model, data_model,lat_mod,lon_mod = get_ts(fname,var,long_obs,lat_obs,ref_date,depth=depth,i_shifted=i_shifted,j_shifted=j_shifted,time_lims=[time_obs[0],time_obs[-1]]) # Change the 10 back to -1

    # get the observations onto the model time axis
    data_obs_model_timeaxis = obs_2_new_timeaxis(fname_obs,time_model, model_frequency,var)
    # print(data_obs_model_timeaxis)

    # Create a NetCDF file
    with nc.Dataset(output_path, 'w', format='NETCDF4') as nc_file:
        # Create dimensions
        nc_file.createDimension('time', len(time_model))

        # if lat_extract is not None:
        #     nc_file.createDimension('latitude', len(lat_extract))
        if j_shifted != 0:
            lat_mod = np.array([lat_mod], dtype=np.float32)
            nc_file.createDimension('latitude', len(lat_mod))
        else:
            nc_file.createDimension('latitude', len(lat_obs))

        if i_shifted != 0:
            nc_file.createDimension('longitude', len(lon_mod))
        else:
            nc_file.createDimension('longitude', len(long_obs))
        
        print(f"6.1 NetCDF file created at: {output_path}")
        # Create variables
        time_var = nc_file.createVariable('time', 'f8', ('time'))
        lat_var = nc_file.createVariable('latitude_insitu', 'f4', ('latitude'))
        lon_var = nc_file.createVariable('longitude_insitu', 'f4', ('longitude'))        
        lat_mod_var = nc_file.createVariable('latitude_on_model', 'f4', ('latitude'))
        lon_mod_var = nc_file.createVariable('longitude_on_model', 'f4', ('longitude'))        
        model_var = nc_file.createVariable('data_model', 'f4', ('time', 'latitude', 'longitude'))
        obs_model_var = nc_file.createVariable('data_obs_model_timeaxis', 'f4', ('time', 'latitude', 'longitude'))
    
        # Convert datetime objects to Unix timestamps (floats)
        float_time_model = np.array([dt.timestamp() for dt in time_model], dtype=int)
        # print(float_time_model)
        # Set attributes for time variable
        
        # Convert each float timestamp to datetime
        # time_model = [datetime.fromtimestamp(float_time) for float_time in float_time_model]
        for dt in time_model:
            print(dt)
    
    
        # Assign data to variables
        time_var[:] = float_time_model
        lat_var[:] = lat_obs
        lon_var[:] = long_obs        
        lat_mod_var[:] = lat_mod
        lon_mod_var[:] = lon_mod        
        model_var[:, :, :] = data_model
        obs_model_var[:, :, :] = data_obs_model_timeaxis
 
        # Add attributes if needed
        time_var.units = 'seconds since 1970-01-01 00:00:00'    
        time_var.calendar = 'standard'        
        time_var.long_name = 'time'
        # time_var.units = 'days'
        lat_var.units = 'latitude'
        lon_var.units = 'longitude'
        lat_mod_var.units = 'latitude'
        lon_mod_var.units= 'longitude'
        model_var.units = 'degrees Celsius'
        obs_model_var.units = 'degrees Celsius'
        
        # data_model_nonan=data_model[not np.isnan(data_obs_model_timeaxis)]
        # data_obs_model_timeaxis_nonan=data_obs_model_timeaxis[not np.isnan(data_obs_model_timeaxis)]
        
        # if lat_extract is not None:
        #     nc_file.setncattr('User_defined_obs_latitude',lat_extract)
        # if lon_extract is not None:
        #     nc_file.setncattr('User_defined_obs_longitude',lon_extract)
            
        nc_file.setncattr('depth',depth)
        nc_file.setncattr('i-shift',i_shifted)
        nc_file.setncattr('j-shift',j_shifted)
        
        # Calculate and add correlations as attributes
        correlation_model_obs = calculate_correlation(data_obs_model_timeaxis, data_model)
        nc_file.setncattr('correlation_model_obs', correlation_model_obs)
        
        # Calculate and add standard deviations as attributes
        std_dev_model = calculate_std_dev(data_model)
        std_dev_obs_model = calculate_std_dev(data_obs_model_timeaxis)
        nc_file.setncattr('std_dev_model', std_dev_model)
        nc_file.setncattr('std_dev_obs_model', std_dev_obs_model)
        
        # Calculate RMSE and add it as an attribute
        rmse_model_obs = calculate_rmse(data_obs_model_timeaxis, data_model)
        nc_file.setncattr('rmse_model_obs', rmse_model_obs)
        
        # Calculate and add total bias as an attribute
        total_bias = calculate_total_bias(data_obs_model_timeaxis, data_model)
        nc_file.setncattr('total_bias', total_bias)
               
        # Calculate and add total bias as an attribute
        model_mean,obs_mean,model_min,obs_min,model_max,obs_max = calculate_min_mean_max(data_obs_model_timeaxis, data_model)
        nc_file.setncattr('model_mean', model_mean)
        nc_file.setncattr('obs_mean', obs_mean)
        nc_file.setncattr('model_min', model_min)
        nc_file.setncattr('obs_min', obs_min)
        nc_file.setncattr('model_max', model_max)
        nc_file.setncattr('obs_max', obs_max)
        
        # seasonal_means = calculate_seasonal_means(data_obs_model_timeaxis,data_model,time_model)
        # nc_file.setncattr('seasonal_means', seasonal_means)
                
        # Calculate anomaly and climatology
        # model_ano = calculate_anomaly(data_obs,time_obs)
        # model_clim = calculate_anomaly(data_obs,time_obs)
        # nc_file.setncattr('anomaly', model_ano)
        # nc_file.setncattr('climatology', model_clim)
    

    
# %%
if __name__ == "__main__":
    # Define the input parameters
    dir_model = '/mnt/d/Run_False_Bay_2008_2018_SANHO/croco_avg_Y2016*.nc.1'
    fname_obs = '/mnt/d/DATA-20231010T133411Z-003/DATA/ATAP/Processed/Processed_Station_Files/CapePoint_CP002.nc'
    fname_out = 'Validation_'+'CapePoint_CP002.nc' #'CapePoint_CP002.nc'  'FalseBay_FB001.nc'

    # Output file name and directory
    output_directory = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/model_validation/'
    fname_out = os.path.join(output_directory, fname_out)

    # Other parameters
    model_frequency='24H'
    var = 'temp'
    depth=-40
    ref_date = datetime(1990, 1, 1, 0, 0, 0)
    
    get_model_obs_ts(dir_model,fname_obs,
                      fname_out,model_frequency=model_frequency,
                      var=var,
                      ref_date=ref_date,
                      depth=depth, 
                      i_shifted=0,j_shifted=-2      
                      # ,lat_extract = -34.4        
                      )
    