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
mpl.use('Agg')
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap

from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm

import xarray as xr

import numpy as np
import glob
import pandas as pd

from scipy.stats import norm
import scipy
import seaborn as sns

from pyhdf.SD import SD, SDC, SDAttr, HDF4Error
from pyhdf import HDF, VS, V
from pyhdf.HDF import *
from pyhdf.VS import *

import pprint
#from pyproj import Proj, transform

import os
import os.path
import sys 

import matplotlib as mpl
import cartopy.crs as ccrs
#import pyresample

from basepath import data_path_base

import netCDF4

import datetime

label_size=14
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

    
##############################################################################

def save_pixel_data(month, longitude, latitude, data_lo_pixel, data_mpc_pixel, data_ltmpc_pixel, data_io_pixel):
        data_lo_pixel.to_netcdf('output_files/pixel_data/monthly/pixel_'+str(year)+'_'+str(month).zfill(2)+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_lo.nc')
        data_mpc_pixel.to_netcdf('output_files/pixel_data/monthly/pixel_'+str(year)+'_'+str(month).zfill(2)+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_mpc.nc')
        data_ltmpc_pixel.to_netcdf('output_files/pixel_data/monthly/pixel_'+str(year)+'_'+str(month).zfill(2)+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_ltmpc.nc')
        data_io_pixel.to_netcdf('output_files/pixel_data/monthly/pixel_'+str(year)+'_'+str(month).zfill(2)+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_io.nc')



# %%
# Define season for T* calculation
year = int(sys.argv[1])

for month in np.arange(1,13,1):
        
    file = data_path_base+'/tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_'+str(month).zfill(2)+'.nc'
    print(file)
    
    try:
        ds = xr.open_dataset(file)
        print('file read...')
    except:
        print('+++ no data for '+str(year)+' '+str(month).zfill(2)+' +++')
        continue
    
    
    # Liquid clouds
    data_lo = ds.where((ds.phase == 3) & (ds.phase_flag >= 7) & (ds.CTT < 270.), drop = True)

    # Mixed phase clouds
    data_mpc = ds.where((ds.phase == 2) & (ds.phase_flag >= 7) & (ds.CTT < 270.), drop = True)

    # Liquid-top Mixed phase clouds
    data_ltmpc = data_mpc.where(data_mpc.CTH - data_mpc.CTH_water <= 90., drop = True)

    # Pure ice clouds
    data_io = ds.where((ds.phase == 1) & (ds.phase_flag >= 7), drop = True)
    
    print('data arrays constructed...')
    
    count_longi = 0
    count_lati = 0

    for longi in np.arange(-180,181,5):
    
        print('longi: ',longi)
    
        for lati in np.arange(-90,91,5):
            
            grid_index_lo = np.where((np.abs(data_lo.lon - longi) < 2.5) & (np.abs(data_lo.lat - lati) < 2.5))
            grid_index_lo = np.array(grid_index_lo).flatten()
            grid_index_mpc = np.where((np.abs(data_mpc.lon - longi) < 2.5) & (np.abs(data_mpc.lat - lati) < 2.5))
            grid_index_mpc = np.array(grid_index_mpc).flatten()
            grid_index_ltmpc = np.where((np.abs(data_ltmpc.lon - longi) < 2.5) & (np.abs(data_ltmpc.lat - lati) < 2.5))
            grid_index_ltmpc = np.array(grid_index_ltmpc).flatten()
            grid_index_io = np.where((np.abs(data_io.lon - longi) < 2.5) & (np.abs(data_io.lat - lati) < 2.5))
            grid_index_io = np.array(grid_index_io).flatten()
            
            if ((len(grid_index_lo) > 0) & (len(grid_index_ltmpc) > 0) & (len(grid_index_mpc) > 0) & (len(grid_index_io) > 0)):
                save_pixel_data(month, longi, lati, data_lo.isel(time = grid_index_lo), data_mpc.isel(time = grid_index_mpc), data_ltmpc.isel(time = grid_index_ltmpc), data_io.isel(time = grid_index_io))


            count_lati += 1                                    
    
        count_longi += 1
        count_lati = 0


