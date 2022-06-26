# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
##################################################################################################################
# IMPORT MODULES
##################################################################################################################

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import glob
import pandas as pd
import xarray as xr

from pyhdf.SD import SD, SDC, SDAttr, HDF4Error
from pyhdf import HDF, VS, V
from pyhdf.HDF import *
from pyhdf.VS import *

import os
import os.path
import sys 

import cartopy.crs as ccrs

import datetime

import basepath
data_path_base = basepath.data_path_base

# %%
##################################################################################################################
# READING DATASETS FROM FILES
##################################################################################################################

# Define month for averaging
year = int(sys.argv[1])
month_start = int(sys.argv[2])
month_stop = int(sys.argv[3])

files = sorted(glob.glob(data_path_base + '/data/cloudsat/2B-CLDCLASS-LIDAR.P1_R05/'+str(year)+'/*/*.hdf'))

for month in range(month_start,month_stop):

    dimension_failure = 0
    granules = 0

    output = 0

    print('Month: ', month)

    for f in files:
        print('Reading datasets from files...')
        day_of_year = int(f[-63:-60])
        granule = int(f[-53:-48])
        print('Day of year: ', day_of_year)
        print('Granule: ', str(granule).zfill(5))
        t = datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year - 1)
        print(t)
        print(t.month, month)
        if t.month < month:
            continue
        if t.month == month:
            output = 1      # at least one file has been used --> output file in the end
            
            file_cld = glob.glob(data_path_base + '/data/cloudsat/2B-CLDCLASS-LIDAR.P1_R05/'+str(year)+'/*/*_*'+str(granule).zfill(5)+'*.hdf')
            file_aux = glob.glob(data_path_base + '/data/cloudsat/ECMWF-AUX.P_R05/'+str(year)+'/*/*_*'+str(granule).zfill(5)+'*.hdf')
        
            # Check if all files exist
            if file_cld and file_aux:
                file_cld = file_cld[0]
                file_aux = file_aux[0]
            else:
                print('Skipping granule (file(s) missing)...')
                continue
        
            print(file_cld)
            print(file_aux)
##################################################################################################################
# Reading 2B-CLDCLASS-LIDAR dataset
            try:
                # Geolocation metadata
                f = HDF(file_cld) 
                vs = f.vstart() 
                Latitude = vs.attach('Latitude')
                Longitude = vs.attach('Longitude')
                Flag = vs.attach('Data_quality')
                TAI_start = vs.attach('TAI_start')
                Profile_time = vs.attach('Profile_time')
                N_Layer = vs.attach('Cloudlayer')
                lat = np.array(Latitude[:])
                lon = np.array(Longitude[:])
                time = np.array(Profile_time[:]) + np.array(TAI_start[:])
                flag_cld = np.array(Flag[:])
                n_layer = np.array(N_Layer[:], dtype='float')
                n_layer = n_layer.flatten()
                Flag.detach() # "close" the vdata
                Latitude.detach() # "close" the vdata
                Longitude.detach() # "close" the vdata
                TAI_start.detach()
                Profile_time.detach() # "close" the vdata
                N_Layer.detach()
                vs.end() # terminate the vdata interface
                f.close()
    
                # 2B-CLDCLASS-LIDAR Cloud phase data
                hdf_cld = SD(file_cld, SDC.READ)
        
                #print(file_cld)
                #print(hdf_cld.datasets())
                
        
                sds_obj = hdf_cld.select('Height') # select sds
                height = sds_obj.get() # get sds data
                sds_obj = hdf_cld.select('CloudLayerTop') # select sds
                layer_top = sds_obj.get() # get sds data
                sds_obj = hdf_cld.select('LayerTopFlag') # select sds
                layer_top_flag = (np.array(sds_obj.get())).astype(float) # get sds data
                sds_obj = hdf_cld.select('CloudLayerBase') # select sds
                layer_base = sds_obj.get() # get sds data
                sds_obj = hdf_cld.select('LayerBaseFlag') # select sds
                layer_base_flag = (np.array(sds_obj.get())).astype(float) # get sds data
                sds_obj = hdf_cld.select('CloudPhase') # select sds
                phase = sds_obj.get() # get sds data
                sds_obj = hdf_cld.select('CloudPhaseConfidenceLevel') # select sds
                phase_flag = (np.array(sds_obj.get())).astype(float) # get sds data
                sds_obj = hdf_cld.select('Phase_log') # select sds
                phase_log = (np.array(sds_obj.get())).astype(float) # get sds data
                sds_obj = hdf_cld.select('Water_layer_top') # select sds
                water_layer_top = sds_obj.get() # get sds data
                sds_obj = hdf_cld.select('CloudLayerType') # select sds
                cloud_layer_type = (np.array(sds_obj.get())).astype(float) # get sds data
                sds_obj = hdf_cld.select('CloudTypeQuality') # select sds
                cloud_type_quality = (np.array(sds_obj.get())).astype(float) # get sds data
                
                

                
            except HDF4Error as msg:
                print("HDF4Error 2B-CLDCLASS-LIDAR", msg)
                print("Skipping granule ...")
                continue


##################################################################################################################
# Auxiliary data from ECMWF-AUX
            try:
                hdf_aux = SD(file_aux, SDC.READ)

                sds_obj = hdf_aux.select('Pressure') # select sds
                pres = sds_obj.get() # get sds data
                sds_obj = hdf_aux.select('Temperature') # select sds
                temp = sds_obj.get() # get sds data
    
            except HDF4Error as msg:
                print("HDF4Error ECMWF-AUX", msg)
                print("Skipping granule ...")
                continue
        
        
################################################################################################################## 
# PROCESS DATA: fill values, unit conversion, valid range, scale factors, offset
##################################################################################################################         
# Data from 2B-CLDCLASS-LIDAR

            n_layer[np.where(n_layer == -9)] = np.nan
            layer_top[np.where(layer_top == -99)] = np.nan
            layer_top_flag[np.where(layer_top_flag == -9)] = np.nan
            layer_base[np.where(layer_base == -99)] = np.nan
            layer_base_flag[np.where(layer_base_flag == -9)] = np.nan
            
            phase_flag[np.where(phase_flag == -9)] = np.nan
            phase_log[np.where(phase_log == -9)] = np.nan

            water_layer_top[np.where(water_layer_top == -9)] = np.nan
            cloud_layer_type[np.where(cloud_layer_type == -9)] = np.nan
            cloud_type_quality[np.where(cloud_type_quality == -99)] = np.nan

            layer_top = layer_top * 1000.    # in m
            layer_base = layer_base * 1000.    # in m

            water_layer_top = water_layer_top * 1000.  # in m
            
            
            height = height.astype(float)
            height[np.where(height == -9999.)] = np.nan
    
################################################################################################################## 
# Auxiliary data from ECMWF-AUX.P_R05

            pres[np.where(pres == -999.0)] = np.nan
            pres = pres / 100. # in hPa

            temp[temp == -999.0] = np.nan
        
##################################################################################################################
# processing
##################################################################################################################
        
            # Define domain: Southern Ocean, good data quality, single-layer clouds
            #index_domain=np.where((lat.flatten() < -20) & (lat.flatten() > -70) & (flag_cld.flatten() == 0) & (n_layer == 1.))
            index_domain=np.where((flag_cld.flatten() == 0) & (n_layer == 1.))

            index_domain = np.array(index_domain)
            index_domain = index_domain.flatten()
            
            
            temp = temp[index_domain,:]
            height = height[index_domain,:]
            n_layer = n_layer[index_domain]
            layer_top = layer_top[index_domain,0].flatten()
            layer_top_flag = layer_top_flag[index_domain,0].flatten()
            layer_base = layer_base[index_domain,0].flatten()
            layer_base_flag = layer_base_flag[index_domain,0].flatten()

            phase = phase[index_domain,0].flatten()
            phase_flag = phase_flag[index_domain,0].flatten()
            phase_log = phase_log[index_domain,0].flatten()

            water_layer_top = water_layer_top[index_domain,0].flatten()
            cloud_layer_type = cloud_layer_type[index_domain,0].flatten()
            cloud_type_quality = cloud_type_quality[index_domain,0].flatten()
                
            
            # find height layer of cloud top height and get cloud top temperature
            CTT = np.empty([len(index_domain)])
            CTT[:] = np.nan
                        
            
            for tt in np.arange(0,len(index_domain),1):
                
                if np.isfinite(water_layer_top[tt]):
                    height_diff = np.abs(water_layer_top[tt]-height[tt,:])
                else:
                    height_diff = np.abs(layer_top[tt]-height[tt,:])
        
                dmin = np.where(height_diff == np.nanmin(height_diff))[0]
        
                CTT[tt] = temp[tt,dmin[0]]
            
            
            # save parameters from this granule
            if granules == 0:
                time_all = time[index_domain].flatten()
                lon_all = lon[index_domain].flatten()
                lat_all = lat[index_domain].flatten()
                n_layer_all = n_layer
                phase_all = phase
                phase_flag_all = phase_flag
                phase_log_all = phase_log
                CTT_all = CTT
                CTH_all = layer_top         # cloud-top height
                CTH_flag_all = layer_top_flag
                CBH_all = layer_base        # cloud base height
                CBH_flag_all = layer_base_flag
                CTH_water_all = water_layer_top
                cloud_layer_type_all = cloud_layer_type
                cloud_type_quality_all = cloud_type_quality
                
            else:
                time_all = np.append(time_all, time[index_domain].flatten())
                lon_all = np.append(lon_all, lon[index_domain].flatten())
                lat_all = np.append(lat_all, lat[index_domain].flatten())
                n_layer_all = np.append(n_layer_all, n_layer)
                phase_all = np.append(phase_all, phase)
                phase_flag_all = np.append(phase_flag_all, phase_flag)
                phase_log_all = np.append(phase_log_all, phase_log)
                CTT_all = np.append(CTT_all, CTT)
                CTH_all = np.append(CTH_all, layer_top)    # cloud-top height
                CTH_flag_all = np.append(CTH_flag_all, layer_top_flag)    # cloud-top height flag
                CBH_all = np.append(CBH_all, layer_base)    # cloud base height
                CBH_flag_all = np.append(CBH_flag_all, layer_base_flag)    # cloud base height flag

                CTH_water_all = np.append(CTH_water_all, water_layer_top)
                cloud_layer_type_all = np.append(cloud_layer_type_all, cloud_layer_type)
                cloud_type_quality_all = np.append(cloud_type_quality_all, cloud_type_quality)
            
            
            granules += 1
            
    print('Month done.')
    
    
    # Create Dataset
    ds = xr.Dataset(data_vars={'lon': ('time', lon_all),
                               'lat': ('time', lat_all),
                               'n_layer': ('time', n_layer_all),
                               'phase': ('time', phase_all),
                               'phase_flag': ('time', phase_flag_all),
                               'phase_log': ('time', phase_log_all),
                               'CTT': ('time', CTT_all),
                               'CTH': ('time', CTH_all),
                               'CTH_flag': ('time', CTH_flag_all),
                               'CBH': ('time', CBH_all),
                               'CBH_flag': ('time', CBH_flag_all),
                               'CTH_water': ('time', CTH_water_all),
                               'cloud_layer_type': ('time', cloud_layer_type_all),
                               'cloud_type_quality': ('time', cloud_type_quality_all)}, 
                coords={'time': time_all})
    ds.to_netcdf('output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_'+str(month).zfill(2)+'.nc')

    
    
print('Done.')
