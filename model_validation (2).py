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
from datetime import datetime,timedelta


# %%

"""
       The next Function is for loading IN SITU time series: 
"""

def get_ts_obs(fname_obs,var):
    # A generic function like this will work if we process all the observations 
    # from different sources into the same format
    # Cleaning out duplicates from the dataset
    # obs = post.get_ds(fname_obs) 
    obs = xr.open_dataset(fname_obs)      
    data_obs = np.squeeze(obs[var].values)
    
    #Temperature is the data observation
    time_obs = obs.time.values
    # time_obs = pd.to_datetime(time_obs)
    
    time_obs = time_obs.astype('datetime64[s]').astype(datetime)
    
    long_obs = obs.longitude.values
    lat_obs  = obs.latitude.values
    
    return time_obs, data_obs,long_obs, lat_obs


"""
       The next METHOD is for finding the nearest point to insitu data in the model: 
"""

def find_nearest_point(fname, Longi, Latit):  
    
    # for effeciency we shouldn't use open_mfdataset for this function 
    # only use the first file
    if ('*' in fname) or ('?' in fname) or ('[' in fname):
        fname=glob(fname)[0]
    
    with post.get_ds(fname) as ds:
        # Calculate the distance between (Longi, lat0) and all grid points
        distance = ((ds['lon_rho'].values - Longi) ** 2 + (ds['lat_rho'].values - Latit) ** 2) ** 0.5
        
    # Find the indices of the minimum distance  
    min_index = np.unravel_index(distance.argmin(), distance.shape)
    # min_index is a tuple containing the row and column indices of the minimum value
    j, i = min_index
    
    return j,i

"""
       The next Function is for getting MODEL time series: 
"""

# def get_ts(dir_model,var,depth=-1,model_suffix='',time_lims=[]): 
def get_ts(fname,var,lon,lat,ref_date,depth=-1,time_lims=[]): 
    # this function will eventually get moved into the postprocess.py file of the somisana repo
    # which is why I'd call it just 'get_ts', as it will be obvious that it relates to croco model output   
    # you can use xr.open_mfdataset(dir_model+*nc+model_suffix) as you've been doing or rather loop through the correct months
    # time_lims=[datetime(2012,11,1),datetime(2018,11,1)]
    # Convert datetime64 array to datetime.datetime objects
    
    time_model = post.get_time(fname,ref_date,time_lims=time_lims)
   
    j,i = find_nearest_point(fname,lon,lat)

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
def custom_interp(values, time_diff, time_threshold):
    mask = time_diff <= time_threshold.total_seconds() / 60
    weights = mask * (1 - time_diff / (time_threshold.total_seconds() / 60))
    weights = weights / weights.sum()
    return np.sum(weights * values)

def obs_2_new_timeaxis(time_model, time_obs, data_obs, time_threshold=timedelta(hours=12)):
    # Convert time arrays to pandas datetime for easier handling
    time_model = pd.to_datetime(time_model)
    time_obs = pd.to_datetime(time_obs)

    # Create xarray DataArrays for time_obs and data_obs
    da_obs = xr.DataArray(data_obs, coords={'time': time_obs}, dims=['time'])
    da_model = xr.DataArray(np.nan, coords={'time': time_model}, dims=['time'])

    # Calculate time differences between time_model points and time_obs
    time_diff = (np.array(time_model)[:, np.newaxis] - time_obs).astype('timedelta64[m]')

    # Use custom interpolation function
    da_interpolated = da_obs.interp(time=da_model.time, method=custom_interp, kwargs={'time_diff': time_diff, 'time_threshold': time_threshold})

    # Convert the result to a numpy array
    data_obs_model_timeaxis = da_interpolated.values

    return data_obs_model_timeaxis

"""
       The next Function is for retrieving all datasets from the previous functions and package them into a netCDF file: 
"""
    
def get_model_obs_ts(fname,fname_obs,fname_out,var='temp',depth=-1,ref_date=None,time_threshold=timedelta(hours=12)):
    # the output of this function is a netcdf file 'fname_out'
    # which will have the model and observations on the same time axis
    # get the observations time-series
    time_obs, data_obs, long_obs, lat_obs = get_ts_obs(fname_obs,var)
    
    # get the model time-series
    time_model, data_model = get_ts(fname,var,long_obs,lat_obs,ref_date,depth=depth,time_lims=[time_obs[0],time_obs[10]]) # Change the 10 back to -1
    
    # get the observations onto the model time axis
    data_obs_model_timeaxis = obs_2_new_timeaxis(time_model, time_obs, data_obs, time_threshold=time_threshold)
    
    # j, i = find_nearest_point(dir_model + SelectFiles, long_obs, lat_obs)   
    # now write out the netcdf file
    # Create a DataArray or Dataset
    ds = xr.Dataset(
        {
            # 'time_obs': xr.DataArray(time_obs, dims='time'),
            'data_obs': xr.DataArray(data_obs, dims=('time')),
            # 'long_obs': xr.DataArray(long_obs, dims='lon'),
            # 'lat_obs': xr.DataArray(lat_obs, dims='lat'),
            'time_model': xr.DataArray(time_model, dims='time_model'),
            'data_model': xr.DataArray(data_model, dims=('time_model', 'depth_model', 'lat_model', 'lon_model')),
            # 'data_obs_model_timeaxis': xr.DataArray(data_obs_model_timeaxis, dims=('new_time', 'lat_obs_model', 'lon_obs_model')),
            # 'nearest_index_j': xr.DataArray(j, dims='index'),
            # 'nearest_index_i': xr.DataArray(i, dims='index'),
        }
    )

    # Save to NetCDF file
    output_file =  '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/scripts/Output_model_obs_ts.nc'
    ds.to_netcdf(output_file)

if __name__ == "__main__":
    
    # define what we're validating...
    
    dir_model = '/mnt/d/Run_False_Bay_2008_2018_SANHO/croco_avg_Y2013M*.nc.1'
    fname_obs = '/mnt/d/DATA-20231010T133411Z-003/DATA/ATAP/Processed/Data_Validation/FalseBaydata_FB001.nc'
    fname_out = ''''''
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

    