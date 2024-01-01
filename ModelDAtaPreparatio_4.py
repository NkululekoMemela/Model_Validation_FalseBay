
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 06:54:12 2023
@author: nkululeko
"""
# %% Importing neccassary packages 

import numpy as np
import xarray as xr 
import netCDF4 as nc
from glob import glob
from natsort import natsorted
import pandas as pd 
from datetime import datetime
import pylab as plt
import numpy as np
import xarray as xr 
import netCDF4 as nc
import pylab as plt
import cartopy.crs as ccrs
import cartopy
from glob import glob
from natsort import natsorted
import pandas as pd 
from sklearn.metrics import mean_squared_error
from math import sqrt

# %%Add postprocess library

#'/home/nkululeko/somisana/toolkit/cli/applications/croco'
# import postpocess as post
import sys
import os
# Add the directory to the Python import search path
directory_path = '/home/nkululeko/somisana/toolkit/cli/applications/croco'
sys.path.append(directory_path)
# Now you can import modules from the added directory
import postprocess as post

# %% Reading netcdf
DataDirModel = '/home/nkululeko/ATAP_FalseBayInsituData/croco_falsebay/'

# Open and read .nc file
with nc.Dataset(DataDirModel +'croco_avg_Y2012M11.nc.1', 'r') as dataset:
    # You now have access to the dataset's variables, dimensions, and attributes.
    # list of all variable names:
    variable_names = dataset.variables.keys()
    print("Variable names:", variable_names)

# %% Loading netcdf files using xarray 
DataDirModel = '/home/nkululeko/ATAP_FalseBayInsituData/croco_falsebay/'

depth_level = -35

# Start and end of the simulation 
start_date = '2012-11-10T12:00:00.000000000'
end_date = '2012-12-31T12:00:00.000000000'

# Start and end of the simulation 
start_date_in_situ = '2012-11-10T12:00:00.000000000'
end_date_in_situ = '2012-12-31T12:00:00.000000000'

# %% start reading in the data

#Getting a directory of file names natsorted is include to ensure files are in the correct order 
modelFiles = natsorted(glob(DataDirModel+"croco_avg_Y2012M1*.nc.1"))
#Variable name 
vname = 'temp'
#Concatenating mutliple dataset to single dataset with a single variable and cooridates  
model_dataset = xr.open_mfdataset(DataDirModel+"croco_avg_Y2012M1*.nc.1")
# %% start reading in variables

# Converting the time of dataset from seconds to datetime
a=model_dataset.coords['time'].values
local_time = nc.num2date(a, units='seconds since 1990-01-01 00:00:00') 
## then copy the decoded back into the dataset
lt=xr.DataArray(local_time,dims='time').data
model_dataset = model_dataset.assign_coords(time=('time',lt))

#Converting time from cftime.DatetimeGregorian to datetime
datetimeindex = model_dataset.indexes['time'].to_datetimeindex()
model_dataset['time'] = datetimeindex

#Slice time
model_dataset = model_dataset.sel(time=slice(start_date,end_date))

#Loading lon and lat 
lon_croco= model_dataset.lon_rho[0,:]
lat_croco = model_dataset.lat_rho[:,0]

#SST
model_sst = model_dataset.values
model_time = model_dataset.time

#Loading grid 
ds_grid = xr.open_dataset(DataDirModel+'croco_grd.nc.1')

# Load temp
temp = model_dataset.temp.values

# %% # %% Calculate slices 

#getting 1d height or water depth
h = ds_grid.h.values 

#getting 1d zeta (sea surface height)
ssh = model_dataset.zeta.values

#Variables from dataset 
theta_s = model_dataset.theta_s
theta_b = model_dataset.theta_b
hc = 200
N = np.shape(model_dataset.s_rho)[0]
type_coordinate = 'rho'
vtransform = 2
# %%
#%%
def csf(sc, theta_s, theta_b):
            '''
            Allows use of theta_b > 0 (July 2009)
            is required in zlevs.py
            '''
            one64 = np.float64(1)

            if theta_s > 0.:
                csrf = ((one64 - np.cosh(theta_s * sc)) /
                        (np.cosh(theta_s) - one64))
            else:
                csrf = -sc ** 2
            sc1 = csrf + one64
            if theta_b > 0.:
                Cs = ((np.exp(theta_b * sc1) - one64) /
                      (np.exp(theta_b) - one64) - one64)
            else:
                Cs = csrf
                
            return Cs

def zlevs(h,zeta,theta_s,theta_b,hc,N,type,vtransform):
    """
    this provides a 3D grid of the depths of the sigma levels
    h = 2D bathymetry of your grid
    zeta = zeta at particular timestep that you are interested in
    theta_s = surface stretching paramter
    theta_b = bottom stretching parameter
    hc = critical depth
    N = number of sigma levels
    type = 'w' or 'rho'
    vtransform = 1 (OLD) or 2 (NEW)
    
    this is adapted (by J.Veitch - Feb 2022) from zlevs.m in roms_tools (by P. Penven)

    """


    [M,L]=np.shape(h)

    sc_r=np.zeros((N,1))
    Cs_r=np.zeros((N,1))
    sc_w=np.zeros((N+1,1))
    Cs_w=np.zeros((N+1,1))

    if vtransform==2:
        ds=1/N

        if type=='w':
            sc_r[0,0]=-1.0
            sc_w[N,0]=0
            Cs_w[0,0]=-1.0
            Cs_w[N,0]=0

            sc_w[1:-1,0]=ds*(np.arange(1,N,1)-N)

            Cs_w=csf(sc_w,theta_s,theta_b)
            N=N+1
        else:
            sc=ds*(np.arange(1,N+1,1)-N-0.5)
            Cs_r=csf(sc, theta_s,theta_b)
            sc_r=sc

    else:
        cff1 = 1. / np.sinh(theta_s)
        cff2 = 0.5 / np.tanh(0.5 * theta_s)

        if type=='w':
            sc=(np.arange(0,N+1,1)-N)/N
            N=N+1
        else:
            sc=(np.arange(1,N+1,1)-N-0.5)/N

        Cs = (1. - theta_b) * cff1 * np.sinh(theta_s * sc) +\
                    theta_b * (cff2 * np.tanh(theta_s * (sc + 0.5)) - 0.5)

    z = np.empty((int(N),) + h.shape, dtype=np.float64)

    h[h==0]=1e-2
    Dcrit=0.01
    zeta[zeta<(Dcrit-h)]=Dcrit-h[zeta<(Dcrit-h)]
    hinv=1/h

    z=np.zeros((N,M,L))

    if vtransform == 2:
        if type=='w':
            cff1=Cs_w
            cff2=sc_w+1
            sc=sc_w
        else:
            cff1=Cs_r
            cff2=sc_r+1
            sc=sc_r
        h2=(h+hc)
        cff=hc*sc
        h2inv=1/h2

        for k in np.arange(N, dtype=int):
            z0=cff[k]+cff1[k]*h
            z[k,:,:]=z0*h/(h2) + zeta*(1.+z0*h2inv)
    else:
        cff1=Cs
        cff2=sc+1
        cff=hc*(sc-Cs)
        cff2=sc+1
        for k in np.arange(N, dtype=int):
            z0=cff[k]+cff1[k]*h
            z[k,:,:]=z0 + zeta*(1.+z0*hinv)

    return z.squeeze()


def hlev(var,z,depth):
    """
    this extracts a horizontal slice 
    
    var = 3D extracted variable of interest
    z = depths (in m) of sigma levels, also 3D array (this is done using zlevs)
    depth = the horizontal depth you want (should be negative)
    
    Adapted (by J.Veitch) from vinterp.m in roms_tools (by P.Penven)
    
    """
    [N,Mp,Lp]=np.shape(z)

    a=z.copy()
    a[a<depth]=1
    a[a!=1]=0
    levs=np.sum(a,axis=0)
    levs[levs==N]=N-1
    mask=levs/levs

    vnew = np.zeros((Mp,Lp))

    for m in np.arange(Mp):
        for l in np.arange(Lp):
            ind1=levs[m,l]
            ind2=levs[m,l]+1

            v1=var[int(ind1),m,l]
            v2=var[int(ind2),m,l]

            z1=z[int(ind1),m,l]
            z2=z[int(ind2),m,l]

            vnew[m,l]=(((v1-v2)*depth+v2*z1-v1*z2)/(z1-z2))

        vnew=vnew*mask
        
    return vnew

# %% converting sigma levels to depth 

depth_levels = np.zeros(np.shape(model_dataset.temp))

for x in np.arange(np.size(model_dataset.temp,0)):
    
    depth_temp = zlevs(h,ssh[x,:,:],theta_s,theta_b,hc,N,type_coordinate,vtransform)
    depth_levels[x,:] = depth_temp

# %% Find depth level

temp_depth = np.zeros(np.shape(model_dataset.zeta))

for z in np.arange(np.size(model_dataset.temp,0)):
    
    temp_slice=hlev(temp[z,:,:,:],depth_levels[z,:,:,:],depth_level)

    temp_depth[z,:,:] = temp_slice
    
# %% remove variables 

del temp
del depth_levels
del ssh 

# %%  closest model point to SWC-CROCO

#Station Points at dep

# This is an x-array conversion method
ds_model = xr.DataArray(temp_depth, coords=[model_time, lat_croco, lon_croco], dims=["time", "lat","lon"])
# This is an output of the above conversion and it captures the closest [point between model and observation
model_pp = ds_model.sel(lon=18.8,lat=-34.4, method= 'nearest')


# %% now I'm bringing in the In situ dataset  

dataset_path = '/home/nkululeko/croco/FalseBaydata_FB001.nc'
ds = xr.open_dataset(dataset_path)

# Cleaning out duplicates from the dataset
ds_clean = ds.drop_duplicates(dim="time")

# Confirming the success of this cleaning effort
if len(set(ds_clean.time.values)) == len(ds_clean.time.values):
    print("success")
else:
    print("duplicate found")
    
ds_sst = ds_clean.temperature.sel(time=slice(start_date_in_situ,end_date_in_situ))
ds_sst_daily = ds_sst.resample(time="1D").mean()
#Remove the following line as it is redundent. It converts to an xarray data that is already in xarray.
#ds_InSitu = xr.DataArray(ds.temperature, coords=[ds.time, ds.longitude, ds.latitude], dims=["time", "longitude","latitude"])

#SST
temp = np.squeeze(ds_sst.values)
temp_daily = np.squeeze(ds_sst_daily.values)
time = ds_sst.time
time_daily = ds_sst_daily.time


# %% Anomalies and clim

model_pp = model_pp

model_ano = model_pp - model_pp.mean("time")
model_ano = model_ano
model_clim =  model_pp.mean("time")

sst_ano =  ds_sst_daily - ds_sst_daily.mean("time")
sst_clim =  ds_sst_daily.mean("time")

# %% Correltation and rmse 
#loading values for dask

temp_corr = temp_daily[~np.isnan(temp_daily)]
model_corr = model_pp[~np.isnan(temp_daily)]

model_in_situ_corr = np.corrcoef(np.array(temp_corr),np.array(model_corr.values))[1][0]
model_in_situ_rmse = sqrt(mean_squared_error(temp_corr, model_corr)) 

# %% Correlation and rmse ano

sst_ano_reshaped = sst_ano.values.reshape(model_ano.shape)

temp_ano_corr = sst_ano.values[~np.isnan(sst_ano.values)]
model_ano_corr = model_ano.values[~np.isnan(sst_ano_reshaped)]

model_in_situ_ano_corr = np.corrcoef(temp_ano_corr,model_ano_corr)[1][0]
model_in_situ_ano_rmse = sqrt(mean_squared_error(temp_ano_corr, model_ano_corr)) 


#%%

fig = plt.figure(figsize=(10, 5),facecolor='white')

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

#plt.contourf(model_dataset.lon_rho,model_dataset.lat_rho,model_dataset.temp[0,0,:,:],transform=ccrs.PlateCarree());
plt.plot(18.8006,-34.35667,'or',label='In situ')
plt.plot(model_pp.lon,model_pp.lat,'oy',label='CROCO model')
#plt.plot(model_glorys_pp.lonT,model_glorys_pp.latT,'ob',label='GLORYS model')
plt.legend()

plt.title('Position of in situ data') 
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

#plt.savefig(savepath+'Timeseries_location.png')
       
# %%

fig, ax = plt.subplots(figsize=(12, 15),nrows=3, ncols=1)
#plt.savefig(savepath+'Postion_of_timeseries_HYCOM.png')

ax[0].plot(time_daily ,temp_daily,label='ATAP in situ');
ax[0].plot(model_pp.time,model_pp,'--',label='SCC_model', color='red', linewidth=2.5, linestyle='-');
ax[0].text(0.01, 0.95, '\n SCC_model r='+str(np.round(model_in_situ_corr,2)) +'\n RMSE='+str(np.round(model_in_situ_rmse,2))
            , fontsize=16,
            horizontalalignment='left', verticalalignment='center', transform=ax[0].transAxes)
ax[0].set_title('SCC_model and '+str(dataset_path.split("/")[-1].split("_")[0])+'\n ATAP in situ  SST ',fontsize=18)
ax[0].legend(loc="upper right")
ax[0].set_ylabel('SST') 

ax[1].plot(time_daily,sst_ano_reshaped,label='ATAP in situ Anomaly', color='blue', linewidth=2.5, linestyle='--');
ax[1].plot(model_pp.time,model_ano,'--',label='SCC_model Anomaly', color='red', linewidth=2.5, linestyle='--');
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
#plt.savefig(savepath+savename)


