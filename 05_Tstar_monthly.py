# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
##################################################################################################################
# IMPORT MODULES
##################################################################################################################

import matplotlib as mpl
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap

from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker


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

    
##########################################
def save_pixel_data(month, longitude, latitude, data_lo_pixel, data_mpc_pixel, data_ltmpc_pixel, data_io_pixel):
        data_lo_pixel.to_netcdf('output_files/pixel_data/monthly_total/pixel_'+str(month).zfill(2)+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_lo.nc')
        data_mpc_pixel.to_netcdf('output_files/pixel_data/monthly_total/pixel_'+str(month).zfill(2)+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_mpc.nc')
        data_ltmpc_pixel.to_netcdf('output_files/pixel_data/monthly_total/pixel_'+str(month).zfill(2)+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_ltmpc.nc')
        data_io_pixel.to_netcdf('output_files/pixel_data/monthly_total/pixel_'+str(month).zfill(2)+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_io.nc')

##########################################      
def plot_pixel_histogram(month, longitude, latitude, freq_lo, freq_mpc, freq_ltmpc, freq_io, n_lo, n_mpc, n_ltmpc, n_io):
    fig, axs = plt.subplots(nrows=1,ncols=3,figsize =[18,5])
    
    bins = np.arange(235,271,2)


    if (n_lo > 0) or (n_mpc > 0) or (n_ltmpc > 0) or (n_io > 0): 
        axs[0].bar(bins[:-1],freq_lo,width=2, align="edge", ec="k", label = 'lo: '+str(n_lo))
        axs[1].bar(bins[:-1],freq_mpc,width=2, align="edge", ec="k", color = 'steelblue', alpha = 0.6, label = 'mpc: '+str(n_mpc))
        axs[0].bar(bins[:-1],freq_ltmpc,width=2, align="edge", ec="k", color = 'teal', alpha = 0.6, label = 'ltmpc: '+str(n_ltmpc))
        axs[1].bar(bins[:-1],freq_io,width=2, align="edge", ec="k", color = 'r', alpha = 0.6, label = 'io: '+str(n_io))
        axs[2].bar(bins[:-1],freq_mpc,width=2, align="edge", ec="k", color = 'steelblue', alpha = 0.6, label = 'mpc: '+str(n_mpc))
        axs[2].bar(bins[:-1],freq_ltmpc,width=2, align="edge", ec="k", color = 'teal', alpha = 0.6, label = 'ltmpc: '+str(n_ltmpc))

        if np.isnan(np.nanmax([freq_lo,freq_mpc,freq_ltmpc,freq_io])) is True:
            
            for k in np.arange(3):
                axs[k].set_xlim(275,235)
                axs[k].set_ylim(0,np.nanmax([freq_lo,freq_mpc,freq_ltmpc,freq_io])+0.01)
                axs[k].set_title(str(month).zfill(2)+' lat: '+str(latitude)+' lon: '+str(longitude))
                axs[k].legend()
                
        else:
            for k in np.arange(3):
                axs[k].set_xlim(275,235)
                axs[k].set_title(str(month).zfill(2)+' lat: '+str(latitude)+' lon: '+str(longitude))
                axs[k].legend()
    
        plt.savefig('Figures/pixel_histograms/monthly/pixel_histograms_'+str(month).zfill(2)+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'.png',bbox_inches='tight')

    plt.close()  
    
##########################################    
def calc_tstar_cloud_stats_monthly(month, latitude, longitude):
    
    files_pixel_lo = sorted(glob.glob('output_files/pixel_data/monthly/pixel_*_'+str(month).zfill(2)+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_lo.nc'))
    files_pixel_mpc = sorted(glob.glob('output_files/pixel_data/monthly/pixel_*_'+str(month).zfill(2)+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_mpc.nc'))
    files_pixel_ltmpc = sorted(glob.glob('output_files/pixel_data/monthly/pixel_*_'+str(month).zfill(2)+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_ltmpc.nc'))
    files_pixel_io = sorted(glob.glob('output_files/pixel_data/monthly/pixel_*_'+str(month).zfill(2)+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_io.nc'))
    
    # read data
    data_lo = xr.open_mfdataset(files_pixel_lo, combine = 'by_coords')
    data_mpc = xr.open_mfdataset(files_pixel_mpc, combine = 'by_coords')
    data_ltmpc = xr.open_mfdataset(files_pixel_ltmpc, combine = 'by_coords')
    data_io = xr.open_mfdataset(files_pixel_io, combine = 'by_coords')
    
    # get histograms
    bins = np.arange(235,271,2)

    hist_lo, edges_lo = np.histogram(data_lo.CTT, bins);
    hist_mpc, edges_mpc = np.histogram(data_mpc.CTT, bins);
    hist_ltmpc, edges_lo = np.histogram(data_ltmpc.CTT, bins);
    hist_io, edges_io = np.histogram(data_io.CTT, bins);

    freq_lo_total = hist_lo/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
    freq_mpc_total = hist_mpc/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
    freq_ltmpc_total = hist_ltmpc/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
    freq_io_total = hist_io/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
    
    # plotting histograms
    plot_pixel_histogram(month, longitude, latitude, freq_lo_total, freq_mpc_total, freq_ltmpc_total, freq_io_total, len(data_lo.CTT), len(data_mpc.CTT), len(data_ltmpc.CTT), len(data_io.CTT))
    save_pixel_data(month, longitude, latitude, data_lo, data_mpc, data_ltmpc, data_io)

         
    # Calculate T* again from data
    #index = np.array(np.where(freq_lo_total-freq_ltmpc_total < 0)).flatten();
    index = np.array(np.where((freq_lo_total-freq_ltmpc_total < 0) & (hist_ltmpc > 10))).flatten();    ####### NEW !!! ###########

                         
    if len(index) == 0:
        return([np.nan, len(data_lo.CTT), len(data_ltmpc.CTT), len(data_mpc.CTT), len(data_io.CTT), np.nanmean(data_lo.CTH), np.nanmean(data_ltmpc.CTH), np.nanmean(data_lo.CTT), np.nanmean(data_ltmpc.CTT)])

    else:
        return([np.max(bins[index]), len(data_lo.CTT), len(data_ltmpc.CTT), len(data_mpc.CTT), len(data_io.CTT), np.nanmean(data_lo.CTH), np.nanmean(data_ltmpc.CTH), np.nanmean(data_lo.CTT), np.nanmean(data_ltmpc.CTT)])


# +
month = int(sys.argv[1])

lat_grid = np.arange(-90,91,5)
lon_grid = np.arange(-180,181,5)
lon2d, lat2d = np.meshgrid(lon_grid,lat_grid,indexing='ij')


cld_lo = np.ones([len(lon_grid), len(lat_grid)],dtype=np.float64)
cld_lo[:,:] = np.nan
cld_ltmpc = np.ones([len(lon_grid), len(lat_grid)],dtype=np.float64)
cld_ltmpc[:,:] = np.nan
cld_mpc = np.ones([len(lon_grid), len(lat_grid)],dtype=np.float64)
cld_mpc[:,:] = np.nan
cld_io = np.ones([len(lon_grid), len(lat_grid)],dtype=np.float64)
cld_io[:,:] = np.nan
cth_lo = np.ones([len(lon_grid), len(lat_grid)],dtype=np.float64)
cth_lo[:,:] = np.nan
cth_ltmpc = np.ones([len(lon_grid), len(lat_grid)],dtype=np.float64)
cth_ltmpc[:,:] = np.nan
ctt_lo = np.ones([len(lon_grid), len(lat_grid)],dtype=np.float64)
ctt_lo[:,:] = np.nan
ctt_ltmpc = np.ones([len(lon_grid), len(lat_grid)],dtype=np.float64)
ctt_ltmpc[:,:] = np.nan

T_star = np.ones([len(lon_grid), len(lat_grid)],dtype=np.float64)
T_star[:,:] = np.nan


count = 0
count_longi = 0
count_lati = 0

for longi in np.arange(-180,181,5):
    
    print(longi)
        
    for lati in np.arange(-90,91,5):
        
        try:
            result = calc_tstar_cloud_stats_monthly(month, lati, longi)
            
            T_star[count_longi, count_lati] = result[0]

            # statistics
            cld_lo[count_longi, count_lati] = result[1]
            cld_ltmpc[count_longi, count_lati] = result[2]
            cld_mpc[count_longi, count_lati] = result[3]
            cld_io[count_longi, count_lati] = result[4]
            cth_lo[count_longi, count_lati] = result[5]
            cth_ltmpc[count_longi, count_lati] = result[6]
            ctt_lo[count_longi, count_lati] = result[7]
            ctt_ltmpc[count_longi, count_lati] = result[8]
            

        except:
            count_lati += 1
            continue
        
        
        count_lati += 1
        
        
    count_longi += 1
    count_lati = 0
    
# limit T* (exclude results too close to phase algorithm limits)   
T_star[T_star > 269] = np.nan
T_star[T_star < 244] = np.nan


# save data as xarray in netcdf files
T_star_xr = xr.DataArray(data = T_star, coords = [lon_grid,lat_grid], dims = ['lon','lat'], name = 'Tstar')

cld_lo_xr = xr.DataArray(data = cld_lo, coords = [lon_grid,lat_grid], dims = ['lon','lat'], name = 'n_lo')
cld_ltmpc_xr = xr.DataArray(data = cld_ltmpc, coords = [lon_grid,lat_grid], dims = ['lon','lat'], name = 'n_ltmpc')
cld_mpc_xr = xr.DataArray(data = cld_mpc, coords = [lon_grid,lat_grid], dims = ['lon','lat'], name = 'n_mpc')
cld_io_xr = xr.DataArray(data = cld_io, coords = [lon_grid,lat_grid], dims = ['lon','lat'], name = 'n_io')
cth_lo_xr = xr.DataArray(data = cth_lo, coords = [lon_grid,lat_grid], dims = ['lon','lat'], name = 'cth_lo')
cth_ltmpc_xr = xr.DataArray(data = cth_ltmpc, coords = [lon_grid,lat_grid], dims = ['lon','lat'], name = 'cth_ltmpc')
ctt_lo_xr = xr.DataArray(data = ctt_lo, coords = [lon_grid,lat_grid], dims = ['lon','lat'], name = 'ctt_lo')
ctt_ltmpc_xr = xr.DataArray(data = ctt_ltmpc, coords = [lon_grid,lat_grid], dims = ['lon','lat'], name = 'ctt_ltmpc')

T_star_xr.to_netcdf('output_files/T_star/monthly/Tstar_5x5_total_'+str(month).zfill(2)+'.nc')

cld_lo_xr.to_netcdf('output_files/T_star/monthly/n_obs_lo_5x5_total_'+str(month).zfill(2)+'.nc')
cld_ltmpc_xr.to_netcdf('output_files/T_star/monthly/n_obs_ltmpc_5x5_total_'+str(month).zfill(2)+'.nc')
cld_mpc_xr.to_netcdf('output_files/T_star/monthly/n_obs_mpc_5x5_total_'+str(month).zfill(2)+'.nc')
cld_io_xr.to_netcdf('output_files/T_star/monthly/n_obs_io_5x5_total_'+str(month).zfill(2)+'.nc')
cth_lo_xr.to_netcdf('output_files/T_star/monthly/cth_lo_5x5_total_'+str(month).zfill(2)+'.nc')
cth_ltmpc_xr.to_netcdf('output_files/T_star/monthly/cth_ltmpc_5x5_total_'+str(month).zfill(2)+'.nc')
ctt_lo_xr.to_netcdf('output_files/T_star/monthly/ctt_lo_5x5_total_'+str(month).zfill(2)+'.nc')
ctt_ltmpc_xr.to_netcdf('output_files/T_star/monthly/ctt_ltmpc_5x5_total_'+str(month).zfill(2)+'.nc')

sys.exit()
# +
lat_grid = np.arange(-90,91,5)
lon_grid = np.arange(-180,181,5)
lon2d, lat2d = np.meshgrid(lon_grid,lat_grid,indexing='ij')

month_1 = 9


file = data_path_base+'/tcarlsen/05_INP_SO/output_files/T_star/monthly/Tstar_5x5_total_'+str(month_1).zfill(2)+'.nc'
ds = xr.open_dataset(file)
T_star_1 = ds.Tstar

file = data_path_base+'/tcarlsen/05_INP_SO/output_files/T_star/monthly/n_obs_lo_5x5_total_'+str(month_1).zfill(2)+'.nc'
ds = xr.open_dataset(file)
cld_lo_1 = ds.n_lo

file = data_path_base+'/tcarlsen/05_INP_SO/output_files/T_star/monthly/n_obs_ltmpc_5x5_total_'+str(month_1).zfill(2)+'.nc'
ds = xr.open_dataset(file)
cld_ltmpc_1 = ds.n_ltmpc

########################################
month_2 = 3


file = data_path_base+'/tcarlsen/05_INP_SO/output_files/T_star/monthly/Tstar_5x5_total_'+str(month_2).zfill(2)+'.nc'
ds = xr.open_dataset(file)
T_star_2 = ds.Tstar

file = data_path_base+'/tcarlsen/05_INP_SO/output_files/T_star/monthly/n_obs_lo_5x5_total_'+str(month_2).zfill(2)+'.nc'
ds = xr.open_dataset(file)
cld_lo_2 = ds.n_lo

file = data_path_base+'/tcarlsen/05_INP_SO/output_files/T_star/monthly/n_obs_ltmpc_5x5_total_'+str(month_2).zfill(2)+'.nc'
ds = xr.open_dataset(file)
cld_ltmpc_2 = ds.n_ltmpc
# -

print(np.nanmin(T_star_1), np.nanmax(T_star_1), np.nanmean(T_star_1))
print(np.nanmin(cld_lo_1), np.nanmax(cld_lo_1), np.nanmean(cld_lo_1))
print(np.nanmin(cld_ltmpc_1), np.nanmax(cld_ltmpc_1), np.nanmean(cld_ltmpc_1))
print('####################################')
print(np.nanmin(T_star_2), np.nanmax(T_star_2), np.nanmean(T_star_2))
print(np.nanmin(cld_lo_2), np.nanmax(cld_lo_2), np.nanmean(cld_lo_2))
print(np.nanmin(cld_ltmpc_2), np.nanmax(cld_ltmpc_2), np.nanmean(cld_ltmpc_2))

# +
# Plotting
fig, axs = plt.subplots(nrows=3,ncols=2,figsize =[16,16], subplot_kw={'projection': ccrs.PlateCarree()})

custom_cmap = cm.get_cmap('Spectral_r', 40)
cs = axs[0,0].pcolormesh(lon2d, lat2d, cld_lo_1, cmap=custom_cmap, vmin = 0, vmax = 10000)
gl = axs[0,0].gridlines(crs = ccrs.PlateCarree(), draw_labels=True, linewidth = 1, color = 'k', linestyle = ':')
gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
gl.ylocator = mticker.FixedLocator([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90])

cbar = fig.colorbar(cs, ax=axs[0,0], orientation='horizontal',pad=0.05)
cbar.set_label('Liquid-only '+str(month_1).zfill(2),fontsize=16)
    
extent = [-180, 180, -90, 90]
axs[0,0].set_extent(extent)
axs[0,0].coastlines()

cs = axs[1,0].pcolormesh(lon2d, lat2d, cld_ltmpc_1, cmap=custom_cmap, vmin = 0, vmax = 10000)
gl = axs[1,0].gridlines(crs = ccrs.PlateCarree(), draw_labels=True, linewidth = 1, color = 'k', linestyle = ':')
gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
gl.ylocator = mticker.FixedLocator([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90])

cbar = fig.colorbar(cs, ax=axs[1,0], orientation='horizontal',pad=0.05)
cbar.set_label('LTMPC '+str(month_1).zfill(2),fontsize=16)
    
extent = [-180, 180, -90, 90]
axs[1,0].set_extent(extent)
axs[1,0].coastlines()

cs = axs[2,0].pcolormesh(lon2d, lat2d, T_star_1, cmap=custom_cmap, vmin = 240, vmax = 270)
gl = axs[2,0].gridlines(crs = ccrs.PlateCarree(), draw_labels=True, linewidth = 1, color = 'k', linestyle = ':')
gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
gl.ylocator = mticker.FixedLocator([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90])

cbar = fig.colorbar(cs, ax=axs[2,0], orientation='horizontal',pad=0.05)
cbar.set_label('Tstar (K) ' +str(month_1).zfill(2),fontsize=16)
    
extent = [-180, 180, -90, 90]
axs[2,0].set_extent(extent)
axs[2,0].coastlines()

###############################################################
custom_cmap = cm.get_cmap('Spectral_r', 40)
cs = axs[0,1].pcolormesh(lon2d, lat2d, cld_lo_2, cmap=custom_cmap, vmin = 0, vmax = 10000)
gl = axs[0,1].gridlines(crs = ccrs.PlateCarree(), draw_labels=True, linewidth = 1, color = 'k', linestyle = ':')
gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
gl.ylocator = mticker.FixedLocator([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90])

cbar = fig.colorbar(cs, ax=axs[0,1], orientation='horizontal',pad=0.05)
cbar.set_label('Liquid-only '+str(month_2).zfill(2),fontsize=16)
    
extent = [-180, 180, -90, 90]
axs[0,1].set_extent(extent)
axs[0,1].coastlines()

cs = axs[1,1].pcolormesh(lon2d, lat2d, cld_ltmpc_2, cmap=custom_cmap, vmin = 0, vmax = 10000)
gl = axs[1,1].gridlines(crs = ccrs.PlateCarree(), draw_labels=True, linewidth = 1, color = 'k', linestyle = ':')
gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
gl.ylocator = mticker.FixedLocator([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90])

cbar = fig.colorbar(cs, ax=axs[1,1], orientation='horizontal',pad=0.05)
cbar.set_label('LTMPC '+str(month_2).zfill(2),fontsize=16)
    
extent = [-180, 180, -90, 90]
axs[1,1].set_extent(extent)
axs[1,1].coastlines()

cs = axs[2,1].pcolormesh(lon2d, lat2d, T_star_2, cmap=custom_cmap, vmin = 240, vmax = 270)
gl = axs[2,1].gridlines(crs = ccrs.PlateCarree(), draw_labels=True, linewidth = 1, color = 'k', linestyle = ':')
gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
gl.ylocator = mticker.FixedLocator([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90])

cbar = fig.colorbar(cs, ax=axs[2,1], orientation='horizontal',pad=0.05)
cbar.set_label('Tstar (K) '+str(month_2).zfill(2),fontsize=16)
    
extent = [-180, 180, -90, 90]
axs[2,1].set_extent(extent)
axs[2,1].coastlines()







# +
fig, axs = plt.subplots(nrows=1,ncols=1,figsize =[16,16], subplot_kw={'projection': ccrs.SouthPolarStereo(central_longitude=0.0, true_scale_latitude=None, globe=None)})

custom_cmap = cm.get_cmap('Spectral_r', 40)
cs = axs.pcolormesh(lon2d, lat2d, cld_lo_1, cmap=custom_cmap, vmin = 0, vmax = 10000)
#gl = axs.gridlines(crs = ccrs.SouthPolarStereo(), draw_labels=True, linewidth = 1, color = 'k', linestyle = ':')
#gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
#gl.ylocator = mticker.FixedLocator([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90])

cbar = fig.colorbar(cs, ax=axs, orientation='horizontal',pad=0.05)
cbar.set_label('Liquid-only '+str(month_1).zfill(2),fontsize=16)
    
extent = [-180, 180, -90, 90]
#axs.set_extent(extent)
#axs.coastlines()

# +
import matplotlib.path as mpath

plt.figure(figsize=(6, 6))
ax = plt.axes(projection=ccrs.SouthPolarStereo(central_longitude=0.0, true_scale_latitude=None, globe=None))
ax.coastlines(resolution='110m')
ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
ax.gridlines()
 
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

gl = ax.gridlines(draw_labels=False, xlocs=None, ylocs=None)
gl.n_steps = 50

ax.set_boundary(circle, transform=ax.transAxes)
# -

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# setup north polar stereographic basemap.
# The longitude lon_0 is at 6-o'clock, and the
# latitude circle boundinglat is tangent to the edge
# of the map at lon_0. Default value of lat_ts
# (latitude of true scale) is pole.
m = Basemap(projection='npstere',boundinglat=10,lon_0=270,resolution='l')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(-180.,181.,20.))
m.drawmapboundary(fill_color='aqua')
# draw tissot's indicatrix to show distortion.
ax = plt.gca()
for y in np.linspace(m.ymax/20,19*m.ymax/20,10):
    for x in np.linspace(m.xmax/20,19*m.xmax/20,10):
        lon, lat = m(x,y,inverse=True)
        poly = m.tissot(lon,lat,2.5,100,\
                        facecolor='green',zorder=10,alpha=0.5)
plt.title("North Polar Stereographic Projection")
plt.show()










check_tstar(9, 80, 0)

check_tstar(3, 80, 0)















def check_tstar(month, latitude, longitude):
    print(month)
    print('longitude: ', longitude)
    print('latitude: ', latitude)
    print('===========================================')
    
    file_pixel_lo = 'output_files/pixel_data/monthly_total/pixel_'+str(month).zfill(2)+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_lo.nc'
    file_pixel_mpc = 'output_files/pixel_data/monthly_total/pixel_'+str(month).zfill(2)+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_mpc.nc'
    file_pixel_ltmpc = 'output_files/pixel_data/monthly_total/pixel_'+str(month).zfill(2)+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_ltmpc.nc'
    file_pixel_io = 'output_files/pixel_data/monthly_total/pixel_'+str(month).zfill(2)+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_io.nc'
    
    # read data
    data_lo = xr.open_dataset(file_pixel_lo)
    data_mpc = xr.open_dataset(file_pixel_mpc)
    data_ltmpc = xr.open_dataset(file_pixel_ltmpc)
    data_io = xr.open_dataset(file_pixel_io)
    
    # read T*
    file_tstar = data_path_base+'/tcarlsen/05_INP_SO/output_files/T_star/monthly/Tstar_5x5_total_'+str(month).zfill(2)+'.nc'
    ds = xr.open_dataset(file_tstar)
    tstar_file = ds.Tstar
    
    # get T* from data file
    x = np.where((tstar_file.lon == longitude) & (tstar_file.lat == latitude))
    
    print('T*: ', tstar_file.isel(lon = x[0][0], lat = x[1][0]))
    print('T*: ', tstar_file.sel(lon = longitude, lat = latitude))

    
    print('===========================================')

    # get histograms
    bins = np.arange(235,271,2)

    hist_lo, edges_lo = np.histogram(data_lo.CTT, bins);
    hist_mpc, edges_mpc = np.histogram(data_mpc.CTT, bins);
    hist_ltmpc, edges_lo = np.histogram(data_ltmpc.CTT, bins);
    hist_io, edges_io = np.histogram(data_io.CTT, bins);

    freq_lo_total = hist_lo/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
    freq_mpc_total = hist_mpc/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
    freq_ltmpc_total = hist_ltmpc/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
    freq_io_total = hist_io/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
    
    # plotting histograms
    plot_pixel_histogram(month, longitude, latitude, freq_lo_total, freq_mpc_total, freq_ltmpc_total, freq_io_total, len(data_lo.CTT), len(data_mpc.CTT), len(data_ltmpc.CTT), len(data_io.CTT))

                 
    # Calculate T* again from data
    #index = np.array(np.where(freq_lo_total-freq_ltmpc_total < 0)).flatten();
    index = np.array(np.where((freq_lo_total-freq_ltmpc_total < 0) & (hist_ltmpc > 10))).flatten();    ####### NEW !!! ###########

                         
    if len(index) == 0:
        print('T* (re-calculated): nan')
    else:
        print('T* (re-calculated): ', np.max(bins[index]))
        
    fig, axs = plt.subplots(nrows=1,ncols=1,figsize =[18,5])

    axs.scatter(bins[:-1], freq_lo_total-freq_ltmpc_total)
    axs.plot([0,300],[0,0],'k')
    axs.set_xlim(234,270)


def plot_pixel_histogram(month, longitude, latitude, freq_lo, freq_mpc, freq_ltmpc, freq_io, n_lo, n_mpc, n_ltmpc, n_io):
    fig, axs = plt.subplots(nrows=1,ncols=3,figsize =[18,5])
    
    bins = np.arange(235,271,2)


    if (n_lo > 0) or (n_mpc > 0) or (n_ltmpc > 0) or (n_io > 0): 
        axs[0].bar(bins[:-1],freq_lo,width=2, align="edge", ec="k", label = 'lo: '+str(n_lo))
        axs[1].bar(bins[:-1],freq_mpc,width=2, align="edge", ec="k", color = 'steelblue', alpha = 0.6, label = 'mpc: '+str(n_mpc))
        axs[0].bar(bins[:-1],freq_ltmpc,width=2, align="edge", ec="k", color = 'teal', alpha = 0.6, label = 'ltmpc: '+str(n_ltmpc))
        axs[1].bar(bins[:-1],freq_io,width=2, align="edge", ec="k", color = 'r', alpha = 0.6, label = 'io: '+str(n_io))
        axs[2].bar(bins[:-1],freq_mpc,width=2, align="edge", ec="k", color = 'steelblue', alpha = 0.6, label = 'mpc: '+str(n_mpc))
        axs[2].bar(bins[:-1],freq_ltmpc,width=2, align="edge", ec="k", color = 'teal', alpha = 0.6, label = 'ltmpc: '+str(n_ltmpc))

        if np.isnan(np.nanmax([freq_lo,freq_mpc,freq_ltmpc,freq_io])) is True:
            
            for k in np.arange(3):
                axs[k].set_xlim(275,235)
                axs[k].set_ylim(0,np.nanmax([freq_lo,freq_mpc,freq_ltmpc,freq_io])+0.01)
                axs[k].set_title(str(month).zfill(2)+' lat: '+str(latitude)+' lon: '+str(longitude))
                axs[k].legend()
                
        else:
            for k in np.arange(3):
                axs[k].set_xlim(275,235)
                axs[k].set_title(str(month).zfill(2)+' lat: '+str(latitude)+' lon: '+str(longitude))
                axs[k].legend()
    
        #plt.savefig('Figures/pixel_histograms/monthly/pixel_histograms_'+str(month).zfill(2)+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'.png',bbox_inches='tight')

    #plt.close()  


