#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 05:13:58 2024
@author: nkululeko
"""
import os
from datetime import datetime,timedelta
from model_evaluation_function_copy  import get_model_obs_ts

if __name__ == "__main__":
    # Define the input parameters
    dir_model = '/mnt/d/Run_False_Bay_2008_2018_SANHO/croco_avg_Y201*.nc.1'
    fname_obs = '/mnt/d/DATA-20231010T133411Z-003/DATA/ATAP/Processed/Processed_Station_Files/FalseBay_FB001.nc'
    fname_out = 'Validation_'+'FalseBay_FB001.nc'

    # Output file name and directory
    output_directory = '/mnt/d/DATA-20231010T133411Z-003/DATA/ATAP/Processed/Data_Validation/'
    output_path = os.path.join(output_directory, fname_out)

    # Other parameters

    model_frequency='24H'
    var = 'temp'
    depth=-35
    ref_date = datetime(1990, 1, 1, 0, 0, 0)
    
    get_model_obs_ts(dir_model,fname_obs,
                     fname_out,output_path,model_frequency=model_frequency,
                     var=var,
                     ref_date=ref_date,
                     depth=depth                    
                     )