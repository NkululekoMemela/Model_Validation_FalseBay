#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 05:13:58 2024
@author: nkululeko
"""
"""
    The following initialization script executes the model_evaluation_function.py remotely.
      - The get_model_obs_ts function is initialized by adding the following inputs.
           
        Parameters:
        - fname             :filename of the model
        - fname_obs         :filename of observations
        - output_path       :filename of fname_out
        - var               :variable input by the user. should be the same in both the modeland obs netCDFs.
        - lat               :lat read from the insitu file
        - lon               :lon read from the insitu file 
        - ref_date          :static user input
        - depth             :user input based on insitu sensor depth
        - model_frequency   :user input, it is the model frequency if the model is an average model output
        - i_shifted         :optional user inputs if the z level of the model does not match the insitu depth
        - j_shifted         :optional user inputs if the z level of the model does not match the insitu depth
        - time_lims         :limits are computed based on the length of the insitu data that matches model span    
"""
import os
from datetime import datetime
from model_evaluation_function  import get_model_obs_ts

if __name__ == "__main__":
    # Define the input parameters
    dir_model = '/mnt/d/Run_False_Bay_2008_2018_SANHO/croco_avg_Y201*.nc.1'
    fname_obs = '/mnt/d/DATA-20231010T133411Z-003/DATA/ATAP/Processed/Processed_Station_Files/CapePoint_CP001.nc'
    fname_out = 'Validation_'+'CapePoint_CP001.nc' #'CapePoint_CP002.nc'  'FalseBay_FB001.nc'

    # Output file name and directory
    output_directory = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/model_validation/'
    fname_out = os.path.join(output_directory, fname_out)

    # Other parameters
    model_frequency='24H'
    var = 'temp'
    depth=-32
    ref_date = datetime(1990, 1, 1, 0, 0, 0)
    
    get_model_obs_ts(dir_model,fname_obs,
                      fname_out,model_frequency=model_frequency,
                      var=var,
                      ref_date=ref_date,
                      depth=depth, 
                      i_shifted=0,j_shifted=0        
                      )
#%% CapePoint_CP003.nc
   
    # fname_obs = '/mnt/d/DATA-20231010T133411Z-003/DATA/ATAP/Processed/Processed_Station_Files/CapePoint_CP003.nc'
    # fname_out = 'Validation_'+'CapePoint_CP003.nc' #'CapePoint_CP002.nc'  'FalseBay_FB001.nc'

    # # Output file name and directory
    # output_directory = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/model_validation/'
    # fname_out = os.path.join(output_directory, fname_out)
    # depth=-55
    # ref_date = datetime(1990, 1, 1, 0, 0, 0)
    
    # get_model_obs_ts(dir_model,fname_obs,
    #                   fname_out,model_frequency=model_frequency,
    #                   var=var,
    #                   ref_date=ref_date,
    #                   depth=depth, 
    #                   x=0,y=-4     
    #                   # ,lat_extract = -34.4        
    #                   )
    
#%% CapePoint_CP002.nc
       
        # fname_obs = '/mnt/d/DATA-20231010T133411Z-003/DATA/ATAP/Processed/Processed_Station_Files/CapePoint_CP002.nc'
        # fname_out = 'Validation_'+'CapePoint_CP002.nc' #'CapePoint_CP002.nc'  'FalseBay_FB001.nc'

        # # Output file name and directory
        # output_directory = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/model_validation/'
        # fname_out = os.path.join(output_directory, fname_out)
        # depth=-50
        # ref_date = datetime(1990, 1, 1, 0, 0, 0)
        
        # get_model_obs_ts(dir_model,fname_obs,
        #                   fname_out,model_frequency=model_frequency,
        #                   var=var,
        #                   ref_date=ref_date,
        #                   depth=depth, 
        #                   x=0,y=-3     
        #                   # ,lat_extract = -34.4        
        #                   )
        
 #%% CapePoint_CP001.nc
           
            # fname_obs = '/mnt/d/DATA-20231010T133411Z-003/DATA/ATAP/Processed/Processed_Station_Files/CapePoint_CP001.nc'
            # fname_out = 'Validation_'+'CapePoint_CP001.nc' #'CapePoint_CP002.nc'  'FalseBay_FB001.nc'

            # # Output file name and directory
            # output_directory = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/model_validation/'
            # fname_out = os.path.join(output_directory, fname_out)
            # depth=-32
            # ref_date = datetime(1990, 1, 1, 0, 0, 0)
            
            # get_model_obs_ts(dir_model,fname_obs,
            #                   fname_out,model_frequency=model_frequency,
            #                   var=var,
            #                   ref_date=ref_date,
            #                   depth=depth, 
            #                   x=0,y=-2     
            #                   # ,lat_extract = -34.4        
            #                   )
            
 #%% FalseBay_FB001.nc
           
            # fname_obs = '/mnt/d/DATA-20231010T133411Z-003/DATA/ATAP/Processed/Processed_Station_Files/FalseBay_FB001.nc'
            # fname_out = 'Validation_'+'FalseBay_FB001.nc' #'CapePoint_CP002.nc'  'FalseBay_FB001.nc'

            # # Output file name and directory
            # output_directory = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/model_validation/'
            # fname_out = os.path.join(output_directory, fname_out)
            # depth=-40
            # ref_date = datetime(1990, 1, 1, 0, 0, 0)
            
            # get_model_obs_ts(dir_model,fname_obs,
            #                   fname_out,model_frequency=model_frequency,
            #                   var=var,
            #                   ref_date=ref_date,
            #                   depth=depth, 
            #                   x=0,y=0     
            #                   # ,lat_extract = -34.4        
            #                   )
            
#%% FalseBay_FB002.nc
          
           # fname_obs = '/mnt/d/DATA-20231010T133411Z-003/DATA/ATAP/Processed/Processed_Station_Files/FalseBay_FB002.nc'
           # fname_out = 'Validation_'+'FalseBay_FB002.nc' #'CapePoint_CP002.nc'  'FalseBay_FB001.nc'

           # # Output file name and directory
           # output_directory = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/model_validation/'
           # fname_out = os.path.join(output_directory, fname_out)
           # depth=-50
           # ref_date = datetime(1990, 1, 1, 0, 0, 0)
           
           # get_model_obs_ts(dir_model,fname_obs,
           #                   fname_out,model_frequency=model_frequency,
           #                   var=var,
           #                   ref_date=ref_date,
           #                   depth=depth, 
           #                   x=0,y=-2     
           #                   # ,lat_extract = -34.4        
           #                   )
           
#%% FalseBay_FB003.nc
          
           # fname_obs = '/mnt/d/DATA-20231010T133411Z-003/DATA/ATAP/Processed/Processed_Station_Files/FalseBay_FB003.nc'
           # fname_out = 'Validation_'+'FalseBay_FB003.nc' #'CapePoint_CP003.nc'  'FalseBay_FB001.nc'

           # # Output file name and directory
           # output_directory = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/model_validation/'
           # fname_out = os.path.join(output_directory, fname_out)
           # depth=-58
           # ref_date = datetime(1990, 1, 1, 0, 0, 0)
           
           # get_model_obs_ts(dir_model,fname_obs,
           #                   fname_out,model_frequency=model_frequency,
           #                   var=var,
           #                   ref_date=ref_date,
           #                   depth=depth, 
           #                   x=0,y=-3     
           #                   # ,lat_extract = -34.4        
           #                   )
           
#%% Gansbaai_GSB003.nc
          
           # fname_obs = '/mnt/d/DATA-20231010T133411Z-003/DATA/ATAP/Processed/Processed_Station_Files/Gansbaai_GSB003.nc'
           # fname_out = 'Validation_'+'Gansbaai_GSB003.nc' #'CapePoint_CP003.nc'  'FalseBay_FB001.nc'

           # # Output file name and directory
           # output_directory = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/model_validation/'
           # fname_out = os.path.join(output_directory, fname_out)
           # depth=-30
           # ref_date = datetime(1990, 1, 1, 0, 0, 0)
           
           # get_model_obs_ts(dir_model,fname_obs,
           #                   fname_out,model_frequency=model_frequency,
           #                   var=var,
           #                   ref_date=ref_date,
           #                   depth=depth, 
           #                   x=0,y=0     
           #                   # ,lat_extract = -34.4        
           #                   )
           
#%% WalkerBay_WB003.nc
          
           # fname_obs = '/mnt/d/DATA-20231010T133411Z-003/DATA/ATAP/Processed/Processed_Station_Files/WalkerBay_WB003.nc'
           # fname_out = 'Validation_'+'WalkerBay_WB003.nc' #'CapePoint_CP003.nc'  'FalseBay_FB001.nc'

           # # Output file name and directory
           # output_directory = '/mnt/d/Run_False_Bay_2008_2018_SANHO/Validation/ATAP/model_validation/'
           # fname_out = os.path.join(output_directory, fname_out)
           # depth=-23
           # ref_date = datetime(1990, 1, 1, 0, 0, 0)
           
           # get_model_obs_ts(dir_model,fname_obs,
           #                   fname_out,model_frequency=model_frequency,
           #                   var=var,
           #                   ref_date=ref_date,
           #                   depth=depth, 
           #                   x=0,y=0     
           #                   # ,lat_extract = -34.4        
           #                   )