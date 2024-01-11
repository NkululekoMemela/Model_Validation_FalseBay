#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 05:13:58 2024
@author: nkululeko
"""
import os
from datetime import datetime,timedelta
from model_evaluation_function import hourly_2_frequency
from model_evaluation_function import get_model_obs_ts

if __name__ == "__main__":
    # Define the input parameters
    dir_model = '/mnt/d/Run_False_Bay_2008_2018_SANHO/croco_avg_Y2013M1.nc.1'
    fname_obs = '/mnt/d/DATA-20231010T133411Z-003/DATA/ATAP/Processed/Processed_Station_Files/FalseBay_FB001.nc'
    fname_out = 'GILES2_'+'FalseBay_FB001.nc'

    # Output file name and directory
    output_directory = '/mnt/d/DATA-20231010T133411Z-003/DATA/ATAP/Processed/Data_Validation/'
    output_path = os.path.join(output_directory, fname_out)

    # Other parameters
    var = 'temp'
    depth = -35
    ref_date = datetime(1990, 1, 1, 0, 0, 0)
    time_threshold = timedelta(hours=12)
    
    # Model frequency parameter. This takes into account the model frequency; in our SWC case it is daily.     
    """    
        D/12- denotes 2 HOURLY  conversion
        D/6 - denotes 4 HOURLY  conversion
        D/4 - denotes 6 HOURLY conversion
        D/2 - denotes half-day conversion
        D - denotes daily conversion
        W - denotes weekly conversion
        M - denotes monthly conversion
    """
    conversionType='D'
    obs =  hourly_2_frequency(fname_obs,conversionType)
    
    # Call the function
    get_model_obs_ts(dir_model, 
                     fname_obs, 
                     fname_out, output_path,obs,conversionType=conversionType,
                     var=var, 
                     ref_date=ref_date, 
                     depth=depth, 
                     time_threshold=time_threshold)