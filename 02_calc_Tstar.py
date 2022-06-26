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


##########################################
def plt_Tstar(fig, ax, lon2d, lat2d, Tstar, label, vmin, vmax, n_cmap):
    custom_cmap = cm.get_cmap('Spectral_r', n_cmap)
    cs = ax.pcolormesh(lon2d, lat2d, Tstar, cmap=custom_cmap, vmin = vmin, vmax = vmax)
    ax.gridlines(draw_labels=True, linewidth = 0)
    cbar = fig.colorbar(cs, ax=ax, orientation='horizontal',pad=0.05)
    cbar.set_label(label,fontsize=16)
    
    
    extent = [-180, 180, -90, 90]
    ax.set_extent(extent)
    ax.coastlines()
    
    plt.savefig('Figures/T_star/Tstar_5x5_'+str(year)+'_'+season+'.png',bbox_inches='tight')
    plt.close()

##########################################
def plt_obs(fig, ax, lon2d, lat2d, CTT, label, vmin, vmax, n_cmap):
    custom_cmap = cm.get_cmap('RdYlGn', n_cmap)
    cs = ax.pcolormesh(lon2d, lat2d, CTT, cmap=custom_cmap, vmin = vmin, vmax = vmax, norm = LogNorm())
    ax.gridlines(draw_labels=True, linewidth = 0)
    cbar = fig.colorbar(cs, ax=ax, orientation='horizontal',pad=0.1, norm = LogNorm())
    cbar.set_label(label,fontsize=16)
    
    
    extent = [-180, 180, -90, 90]
    ax.set_extent(extent)
    ax.coastlines()
    
##########################################
def plot_pixel_histogram(season, longitude, latitude, freq_lo, freq_mpc, freq_ltmpc, freq_io, n_lo, n_mpc, n_ltmpc, n_io):
    fig, axs = plt.subplots(nrows=1,ncols=3,figsize =[18,5])

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
                axs[k].set_title(season+' lat: '+str(latitude)+' lon: '+str(longitude))
                axs[k].legend()
                
        else:
            for k in np.arange(3):
                axs[k].set_xlim(275,235)
                axs[k].set_title(season+' lat: '+str(latitude)+' lon: '+str(longitude))
                axs[k].legend()
    
        plt.savefig('Figures/pixel_histograms/pixel_histograms_'+str(year)+'_'+season+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'.png',bbox_inches='tight')

    plt.close()

def save_pixel_data(season, longitude, latitude, data_lo_pixel, data_mpc_pixel, data_ltmpc_pixel, data_io_pixel):
        data_lo_pixel.to_netcdf('output_files/pixel_data/pixel_'+str(year)+'_'+season+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_lo.nc')
        data_mpc_pixel.to_netcdf('output_files/pixel_data/pixel_'+str(year)+'_'+season+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_mpc.nc')
        data_ltmpc_pixel.to_netcdf('output_files/pixel_data/pixel_'+str(year)+'_'+season+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_ltmpc.nc')
        data_io_pixel.to_netcdf('output_files/pixel_data/pixel_'+str(year)+'_'+season+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_io.nc')



# %%
# define grid
lat_grid = np.arange(-90,91,5)
lon_grid = np.arange(-180,181,5)

lon2d, lat2d = np.meshgrid(lon_grid,lat_grid,indexing='ij')


# %%
# Define season for T* calculation
year = int(sys.argv[1])
season = str(sys.argv[2])
#year = 2013
#season = 'JJA'

if season == 'summer':
    filelist = [data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_10.nc', data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_11.nc', data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_12.nc',data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year+1)+'_01.nc',data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year+1)+'_02.nc',data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year+1)+'_03.nc']

if season == 'winter':
    filelist = [data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_04.nc', data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_05.nc', data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_06.nc',data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_07.nc',data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_08.nc',data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_09.nc']

if season == 'DJF':
    filelist = [data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_12.nc', data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year+1)+'_01.nc', data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year+1)+'_02.nc']

if season == 'MAM':
    filelist = [data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_03.nc', data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_04.nc', data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_05.nc']

if season == 'JJA':
    filelist = [data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_06.nc', data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_07.nc', data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_08.nc']

if season == 'SON':
    filelist = [data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_09.nc', data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_10.nc', data_path_base+'tcarlsen/05_INP_SO/output_files/CTT-monthly-mean/radar_lidar_ctt_phase_'+str(year)+'_11.nc']

print(*filelist, sep='\n')
    
ds = xr.open_mfdataset(filelist, combine = 'by_coords')


# %%
# Liquid clouds
data_lo = ds.where((ds.phase == 3) & (ds.phase_flag >= 7) & (ds.CTT < 270.), drop = True)

# Mixed phase clouds
data_mpc = ds.where((ds.phase == 2) & (ds.phase_flag >= 7) & (ds.CTT < 270.), drop = True)

# Liquid-top Mixed phase clouds
data_ltmpc = data_mpc.where(data_mpc.CTH - data_mpc.CTH_water <= 90., drop = True)

# Pure ice clouds
data_io = ds.where((ds.phase == 1) & (ds.phase_flag >= 7), drop = True)

# %%
#print(ds)
#print(data_lo)
#print(data_ltmpc)
#print(data_mpc)
#print(data_io)

# %%
# calculate T_star and T_median
T_star_total = np.ones([len(lon_grid), len(lat_grid)],dtype=np.float64)
T_star_total[:,:] = np.nan

n_lo_all = np.zeros([len(lon_grid), len(lat_grid)],dtype=np.float64)
n_ltmpc_all = np.zeros([len(lon_grid), len(lat_grid)],dtype=np.float64)
n_mpc_all = np.zeros([len(lon_grid), len(lat_grid)],dtype=np.float64)
n_io_all = np.zeros([len(lon_grid), len(lat_grid)],dtype=np.float64)


T_star_freq = np.copy(T_star_total)

# %%
count = 0
count_longi = 0
count_lati = 0

for longi in np.arange(-180,181,5):
    
    print('longi: ',longi)
    
    for lati in np.arange(-90,91,5):
        
        if ((lati < 95.) & (lati > -95.)):
        #if ((lati == -60) & (longi == 0)):
        
            grid_index_lo = np.where((np.abs(data_lo.lon - longi) < 2.5) & (np.abs(data_lo.lat - lati) < 2.5))
            grid_index_lo = np.array(grid_index_lo).flatten()
            grid_index_mpc = np.where((np.abs(data_mpc.lon - longi) < 2.5) & (np.abs(data_mpc.lat - lati) < 2.5))
            grid_index_mpc = np.array(grid_index_mpc).flatten()
            grid_index_ltmpc = np.where((np.abs(data_ltmpc.lon - longi) < 2.5) & (np.abs(data_ltmpc.lat - lati) < 2.5))
            grid_index_ltmpc = np.array(grid_index_ltmpc).flatten()
            grid_index_io = np.where((np.abs(data_io.lon - longi) < 2.5) & (np.abs(data_io.lat - lati) < 2.5))
            grid_index_io = np.array(grid_index_io).flatten()
            
            
            if (len(grid_index_lo) > 0) & (len(grid_index_ltmpc) > 0):
        
                CTT_lo_grid = data_lo.CTT[grid_index_lo]
                CTT_mpc_grid = data_mpc.CTT[grid_index_mpc]
                CTT_ltmpc_grid = data_ltmpc.CTT[grid_index_ltmpc]
                CTT_io_grid = data_io.CTT[grid_index_io]
                
                
                # Get histograms
                bins = np.arange(235,271,2)

                hist_lo, edges_lo = np.histogram(CTT_lo_grid, bins);
                
                hist_mpc, edges_mpc = np.histogram(CTT_mpc_grid, bins);
                
                hist_ltmpc, edges_lo = np.histogram(CTT_ltmpc_grid, bins);
                
                hist_io, edges_io = np.histogram(CTT_io_grid, bins);
                
                freq_lo_total = hist_lo/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
                freq_mpc_total = hist_mpc/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
                freq_ltmpc_total = hist_ltmpc/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
                freq_io_total = hist_io/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
                
                plot_pixel_histogram(season, longi, lati, freq_lo_total, freq_mpc_total, freq_ltmpc_total, freq_io_total, len(grid_index_lo), len(grid_index_mpc), len(grid_index_ltmpc), len(grid_index_io))
                save_pixel_data(season, longi, lati, data_lo.isel(time = grid_index_lo), data_mpc.isel(time = grid_index_mpc), data_ltmpc.isel(time = grid_index_ltmpc), data_io.isel(time = grid_index_io))

                
                # generate domain histogram    
                if count == 0:
                    hist_lo_all = hist_lo
                    hist_mpc_all = hist_mpc
                    hist_ltmpc_all = hist_ltmpc
                    hist_io_all = hist_io
                else:
                    hist_lo_all += hist_lo
                    hist_mpc_all += hist_mpc
                    hist_ltmpc_all += hist_ltmpc
                    hist_io_all += hist_io
                    
                n_lo_all[count_longi, count_lati] += len(grid_index_lo)
                n_ltmpc_all[count_longi, count_lati] += len(grid_index_ltmpc)
                n_mpc_all[count_longi, count_lati] += len(grid_index_mpc)
                n_io_all[count_longi, count_lati] += len(grid_index_io)

            
                # Calculate T*
                index = np.array(np.where(freq_lo_total-freq_ltmpc_total < 0)).flatten();
                         
                if len(index) == 0:
                    T_star_total[count_longi, count_lati] = np.nan

                else:
                    T_star_total[count_longi, count_lati] = np.max(bins[index])

                
                
                # compare with Alexander and Protat (2018): https://doi.org/10.1002/2017JD026552
                # Cape Grim, Australia (40.7 °S, 144.7 °E), at the northern edge of the Southern Ocean
                # from July 2013 to February 2014
                if ((lati == -40.) and (longi == 145)):
                    print('Tstar Cape Grim, Australia: ', T_star_total[count_longi, count_lati], ' K')

                                
                        
                count_lati += 1
                count += 1
                                    
            else:
                count_lati += 1
    
    count_longi += 1
    count_lati = 0
    
    
# save domain histogram
outfile = 'output_files/histograms/domain_'+str(year)+'_'+season+'_data_lo.txt'
np.savetxt(outfile, list(zip(bins, hist_lo_all)),delimiter='   ',fmt='%11.8f')
outfile = 'output_files/histograms/domain_'+str(year)+'_'+season+'_data_mpc.txt'
np.savetxt(outfile, list(zip(bins, hist_mpc_all)),delimiter='   ',fmt='%11.8f')
outfile = 'output_files/histograms/domain_'+str(year)+'_'+season+'_data_ltmpc.txt'
np.savetxt(outfile, list(zip(bins, hist_ltmpc_all)),delimiter='   ',fmt='%11.8f')
outfile = 'output_files/histograms/domain_'+str(year)+'_'+season+'_data_io.txt'
np.savetxt(outfile, list(zip(bins, hist_io_all)),delimiter='   ',fmt='%11.8f')


# limit T_star so that the temperatures are not too close to the algorithm limits
T_star_total[T_star_total > 269] = np.nan
T_star_total[T_star_total < 244] = np.nan


# save data as xarray in netcdf files
T_star_xr = xr.DataArray(data = T_star_total, coords = [lon_grid,lat_grid], dims = ['lon','lat'], name = 'Tstar')

n_lo_xr = xr.DataArray(data = n_lo_all, coords = [lon_grid,lat_grid], dims = ['lon','lat'], name = 'n_lo')
n_ltmpc_xr = xr.DataArray(data = n_ltmpc_all, coords = [lon_grid,lat_grid], dims = ['lon','lat'], name = 'n_ltmpc')
n_mpc_xr = xr.DataArray(data = n_mpc_all, coords = [lon_grid,lat_grid], dims = ['lon','lat'], name = 'n_mpc')
n_io_xr = xr.DataArray(data = n_io_all, coords = [lon_grid,lat_grid], dims = ['lon','lat'], name = 'n_io')


T_star_xr.to_netcdf('output_files/T_star/Tstar_5x5_'+str(year)+'_'+season+'.nc')

n_lo_xr.to_netcdf('output_files/T_star/n_obs_lo_5x5_'+str(year)+'_'+season+'.nc')
n_ltmpc_xr.to_netcdf('output_files/T_star/n_obs_ltmpc_5x5_'+str(year)+'_'+season+'.nc')
n_mpc_xr.to_netcdf('output_files/T_star/n_obs_mpc_5x5_'+str(year)+'_'+season+'.nc')
n_io_xr.to_netcdf('output_files/T_star/n_obs_io_5x5_'+str(year)+'_'+season+'.nc')


# output T*
fig, axs = plt.subplots(nrows=2,ncols=1,figsize =[15,12], subplot_kw={'projection': ccrs.PlateCarree()})

plt_Tstar(fig, axs[0], lon2d, lat2d, T_star_total, 'T_star total (K)', vmin=240, vmax = 270, n_cmap=10)


# output N_obs
fig, axs = plt.subplots(nrows=2,ncols=2,figsize =[20,8], subplot_kw={'projection': ccrs.PlateCarree()})

plt_obs(fig, axs[0,0], lon2d, lat2d, n_lo_all, 'LO observations', n_cmap=30, vmin = 1, vmax = 3000)
plt_obs(fig, axs[0,1], lon2d, lat2d, n_ltmpc_all, 'LTMPC observations', n_cmap=30, vmin = 1, vmax = 3000)
plt_obs(fig, axs[1,0], lon2d, lat2d, n_mpc_all, 'MPC observations', n_cmap=30, vmin = 1, vmax = 3000)
plt_obs(fig, axs[1,1], lon2d, lat2d, n_io_all, 'IO observations', n_cmap=30, vmin = 1, vmax = 3000)


plt.savefig('Figures/observations/N_obs_5x5_'+str(year)+'_'+season+'.png',bbox_inches='tight')
plt.close()


sys.exit()

# %%
count = 0
count_longi = 0
count_lati = 0

for longi in np.arange(-180,181,5):
    
    print('longi: ',longi)
    
    for lati in np.arange(-90,91,5):
        
        if ((lati < 95.) & (lati > -95.)):
        
            grid_index = np.where((np.abs(data_lo.lon - longi) < 2.5) & (np.abs(data_lo.lat - lati) < 2.5))
            grid_index = np.array(grid_index).flatten()
            
            if (len(grid_index) > 0):
        
                CTT_grid = data_lo.CTT[grid_index]
               
               

                                
                        
                count_lati += 1
                count += 1
                                    
            else:
                count_lati += 1
    
    count_longi += 1
    count_lati = 0
    

# %%
fig, axs = plt.subplots(nrows=4,ncols=1,figsize =[15,17], subplot_kw={'projection': ccrs.PlateCarree()})
plt_obs(fig, axs[0], lon2d, lat2d, n_lo_all, 'LO observations', n_cmap=50, vmin = 1, vmax = 5000)
plt_obs(fig, axs[1], lon2d, lat2d, n_ltmpc_all, 'LTMPC observations', n_cmap=50, vmin = 1, vmax = 5000)
plt_obs(fig, axs[2], lon2d, lat2d, n_mpc_all, 'MPC observations', n_cmap=50, vmin = 1, vmax = 5000)
plt_obs(fig, axs[3], lon2d, lat2d, n_io_all, 'IO observations', n_cmap=50, vmin = 1, vmax = 5000)
plt.show()


# %%

# %%
def plt_obs(fig, ax, lon2d, lat2d, CTT, label, vmin, vmax, n_cmap):
    custom_cmap = cm.get_cmap('RdYlGn', n_cmap)
    cs = ax.pcolormesh(lon2d, lat2d, CTT, cmap=custom_cmap, vmin = vmin, vmax = vmax, norm = LogNorm())
    ax.gridlines(draw_labels=True, linewidth = 0)
    cbar = fig.colorbar(cs, ax=ax, orientation='horizontal',pad=0.1, norm = LogNorm())
    cbar.set_label(label,fontsize=16)
    
    
    extent = [-180, 180, -80, 0]
    ax.set_extent(extent)
    ax.coastlines()

# %%
print(np.nanmax(n_lo_all),np.nanmax(n_ltmpc_all),np.nanmax(n_mpc_all),np.nanmax(n_io_all))

# %%
fig, axs = plt.subplots(nrows=2,ncols=2,figsize =[20,8], subplot_kw={'projection': ccrs.PlateCarree()})
plt_obs(fig, axs[0,0], lon2d, lat2d, n_lo_all, 'LO observations', n_cmap=30, vmin = 1, vmax = 3000)
plt_obs(fig, axs[0,1], lon2d, lat2d, n_ltmpc_all, 'LTMPC observations', n_cmap=30, vmin = 1, vmax = 3000)
plt_obs(fig, axs[1,0], lon2d, lat2d, n_mpc_all, 'MPC observations', n_cmap=30, vmin = 1, vmax = 3000)
plt_obs(fig, axs[1,1], lon2d, lat2d, n_io_all, 'IO observations', n_cmap=30, vmin = 1, vmax = 3000)
plt.show()

# %%

# %%
print('lo: ',np.shape(grid_index_lo))
print('mpc: ',np.shape(grid_index_mpc))
print('ltmpc: ',np.shape(grid_index_ltmpc))
print('io: ',np.shape(grid_index_io))

# %%
data_io

# %%
freq_lo = freq_lo_total
freq_mpc = freq_mpc_total
freq_ltmpc = freq_ltmpc_total
freq_io = freq_io_total

n_lo = len(grid_index_lo)
n_mpc = len(grid_index_mpc)
n_ltmpc = len(grid_index_ltmpc)
n_io = len(grid_index_io)

longitude = 0
latitude = -60




fig, axs = plt.subplots(nrows=1,ncols=3,figsize =[18,5])

axs[0].bar(bins[:-1],freq_lo,width=2, align="edge", ec="k", label = 'lo: '+str(n_lo))
axs[1].bar(bins[:-1],freq_mpc,width=2, align="edge", ec="k", color = 'steelblue', alpha = 0.6, label = 'mpc: '+str(n_mpc))
axs[0].bar(bins[:-1],freq_ltmpc,width=2, align="edge", ec="k", color = 'teal', alpha = 0.6, label = 'ltmpc: '+str(n_ltmpc))
axs[1].bar(bins[:-1],freq_io,width=2, align="edge", ec="k", color = 'r', alpha = 0.6, label = 'io: '+str(n_io))
axs[2].bar(bins[:-1],freq_mpc,width=2, align="edge", ec="k", color = 'steelblue', alpha = 0.6, label = 'mpc: '+str(n_mpc))
axs[2].bar(bins[:-1],freq_ltmpc,width=2, align="edge", ec="k", color = 'teal', alpha = 0.6, label = 'ltmpc: '+str(n_ltmpc))

            
for k in np.arange(3):
    axs[k].set_xlim(275,235)
    axs[k].set_ylim(0,np.nanmax([freq_lo,freq_mpc,freq_ltmpc,freq_io])+0.01)
    axs[k].set_title(season+' lat: '+str(latitude)+' lon: '+str(longitude))
    axs[k].legend()
    axs[k].set_xlim(275,235)
    axs[k].set_title(season+' lat: '+str(latitude)+' lon: '+str(longitude))
    axs[k].legend()

plt.show()

# %%
bins_ctt = np.arange(235,271,2)
hist_io_ctt,edges = np.histogram(data_io.CTT[grid_index_io], bins_ctt);

bins_flag = np.arange(0,10,1)
hist_io_phase_flag,edges = np.histogram(data_io.phase_flag[grid_index_io], bins_flag);

bins_cth = np.arange(0,5000,250)
hist_io_cth,edges = np.histogram(data_io.CTH[grid_index_io], bins_cth);



# %%
print(data_io.CTT[data_io.phase_flag > 8])

# %%
bins_ctt = np.arange(235,271,2)
hist_io_ctt,edges = np.histogram(data_io.CTT[data_io.phase_flag >8], bins_ctt);
fig, axs = plt.subplots(nrows=1,ncols=3,figsize =[18,5])

axs[0].bar(bins_ctt[:-1],hist_io_ctt,width=2, align="edge", ec="k", label = 'io: '+str(n_io),color='r')

# %%
print(hist_io_ctt)
print(hist_io_phase_flag)
print(np.shape(hist_io_cth))
print(np.shape(bins_cth[:-1]))

# %%
fig, axs = plt.subplots(nrows=1,ncols=3,figsize =[18,5])

axs[0].bar(bins_ctt[:-1],hist_io_ctt,width=2, align="edge", ec="k", label = 'io: '+str(n_io),color='r')
axs[0].set_xlim(275,235)


axs[1].bar(bins_flag[:-1],hist_io_phase_flag,width=1, align="edge", ec="k", label = 'io: '+str(n_io),color='r')

axs[2].bar(bins_cth[:-1],hist_io_cth,width=250, align="edge", ec="k", label = 'io: '+str(n_io),color='r')


# %%

# %%

# %%
print(np.nanmax(freq_lo_total))

# %%
print(freq_lo_total)

# %%
print('lo: ',len(grid_index_lo))
print('mpc: ',np.shape(grid_index_mpc))
print('ltmpc: ',np.shape(grid_index_ltmpc))
print('io: ',np.shape(grid_index_io))

# %%
if season == 'DJF':
    freq_lo_summer = freq_lo_total
    freq_mpc_summer = freq_mpc_total
    freq_ltmpc_summer = freq_ltmpc_total
    freq_io_summer = freq_io_total
    n_lo_summer = len(grid_index_lo)
    n_mpc_summer = len(grid_index_mpc)
    n_ltmpc_summer = len(grid_index_ltmpc)
    n_io_summer = len(grid_index_io)
else:
    freq_lo_winter = freq_lo_total
    freq_mpc_winter = freq_mpc_total
    freq_ltmpc_winter = freq_ltmpc_total
    freq_io_winter = freq_io_total
    n_lo_winter = len(grid_index_lo)
    n_mpc_winter = len(grid_index_mpc)
    n_ltmpc_winter = len(grid_index_ltmpc)
    n_io_winter = len(grid_index_io)

# %%
fig, axs = plt.subplots(nrows=2,ncols=3,figsize =[18,8])

axs[0,0].bar(bins[:-1],freq_lo_summer,width=2, align="edge", ec="k", label = 'lo: '+str(n_lo_summer))
axs[0,1].bar(bins[:-1],freq_mpc_summer,width=2, align="edge", ec="k", color = 'steelblue', alpha = 0.6, label = 'mpc: '+str(n_mpc_summer))
axs[0,0].bar(bins[:-1],freq_ltmpc_summer,width=2, align="edge", ec="k", color = 'teal', alpha = 0.6, label = 'ltmpc: '+str(n_ltmpc_summer))
axs[0,1].bar(bins[:-1],freq_io_summer,width=2, align="edge", ec="k", color = 'r', alpha = 0.6, label = 'io: '+str(n_io_summer))
axs[0,2].bar(bins[:-1],freq_mpc_summer,width=2, align="edge", ec="k", color = 'steelblue', alpha = 0.6, label = 'mpc: '+str(n_mpc_summer))
axs[0,2].bar(bins[:-1],freq_ltmpc_summer,width=2, align="edge", ec="k", color = 'teal', alpha = 0.6, label = 'ltmpc: '+str(n_ltmpc_summer))


axs[1,0].bar(bins[:-1],freq_lo_winter,width=2, align="edge", ec="k", label = 'lo: '+str(n_lo_winter))
axs[1,1].bar(bins[:-1],freq_mpc_winter,width=2, align="edge", ec="k", color = 'steelblue', alpha = 0.6, label = 'mpc: '+str(n_mpc_winter))
axs[1,0].bar(bins[:-1],freq_ltmpc_winter,width=2, align="edge", ec="k", color = 'teal', alpha = 0.6, label = 'ltmpc: '+str(n_ltmpc_winter))
axs[1,1].bar(bins[:-1],freq_io_winter,width=2, align="edge", ec="k", color = 'r', alpha = 0.6, label = 'io: '+str(n_io_winter))
axs[1,2].bar(bins[:-1],freq_mpc_winter,width=2, align="edge", ec="k", color = 'steelblue', alpha = 0.6, label = 'mpc: '+str(n_mpc_winter))
axs[1,2].bar(bins[:-1],freq_ltmpc_winter,width=2, align="edge", ec="k", color = 'teal', alpha = 0.6, label = 'ltmpc: '+str(n_ltmpc_winter))


axs[0,0].set_xlim(275,235)
axs[0,1].set_xlim(275,235)
axs[0,2].set_xlim(275,235)
axs[1,0].set_xlim(275,235)
axs[1,1].set_xlim(275,235)
axs[1,2].set_xlim(275,235)

axs[0,0].set_ylim(0,0.12)
axs[0,1].set_ylim(0,0.12)
axs[0,2].set_ylim(0,0.12)
axs[1,0].set_ylim(0,0.12)
axs[1,1].set_ylim(0,0.12)
axs[1,2].set_ylim(0,0.12)

axs[0,0].set_title('Summer')
axs[0,1].set_title('Summer')
axs[0,2].set_title('Summer')
axs[1,0].set_title('Winter')
axs[1,1].set_title('Winter')
axs[1,2].set_title('Winter')


axs[0,0].legend()
axs[0,1].legend()
axs[0,2].legend()
axs[1,0].legend()
axs[1,1].legend()
axs[1,2].legend()


# %%

# %%
fig, axs = plt.subplots(nrows=2,ncols=2,figsize =[10,8], subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)

extent = [-3, 3, -63, -57]

axs[0,0].set_extent(extent)
cs = axs[0,0].scatter(data_lo.lon[grid_index_lo], data_lo.lat[grid_index_lo], c=data_lo.CTT[grid_index_lo], cmap='Spectral_r', s=3, marker='s', vmin=235, vmax=275)
cs2 = axs[0,0].plot([-2.5,2.5,2.5,-2.5,-2.5],[-62.5,-62.5,-57.5,-57.5,-62.5],linestyle='-',color='k',linewidth=3)
#cbar = plt.colorbar(cs, ax=axs[0,0],shrink=0.95)
#cbar.set_label('Cloud Top Temperature (K)',fontsize=16, fontweight='bold')

axs[0,1].set_extent(extent)
cs3 = axs[0,1].scatter(data_io.lon[grid_index_io], data_io.lat[grid_index_io], c=data_io.CTT[grid_index_io], cmap='Spectral_r', s=3, marker='s', vmin=235, vmax=275)
cs4 = axs[0,1].plot([-2.5,2.5,2.5,-2.5,-2.5],[-62.5,-62.5,-57.5,-57.5,-62.5],linestyle='-',color='k',linewidth=3)
#cbar2 = plt.colorbar(cs, ax=axs[0,1],shrink=0.95)
#cbar2.set_label('Cloud Top Temperature (K)',fontsize=16, fontweight='bold')

axs[1,0].set_extent(extent)
cs5 = axs[1,0].scatter(data_ltmpc.lon[grid_index_ltmpc], data_ltmpc.lat[grid_index_ltmpc], c=data_ltmpc.CTT[grid_index_ltmpc], cmap='Spectral_r', s=3, marker='s', vmin=235, vmax=275)
cs6 = axs[1,0].plot([-2.5,2.5,2.5,-2.5,-2.5],[-62.5,-62.5,-57.5,-57.5,-62.5],linestyle='-',color='k',linewidth=3)
#cbar3 = plt.colorbar(cs, ax=axs[1,0],shrink=0.95)
#cbar3.set_label('Cloud Top Temperature (K)',fontsize=16, fontweight='bold')

axs[1,1].set_extent(extent)
cs7 = axs[1,1].scatter(data_mpc.lon[grid_index_mpc], data_mpc.lat[grid_index_mpc], c=data_mpc.CTT[grid_index_mpc], cmap='Spectral_r', s=3, marker='s', vmin=235, vmax=275)
cs8 = axs[1,1].plot([-2.5,2.5,2.5,-2.5,-2.5],[-62.5,-62.5,-57.5,-57.5,-62.5],linestyle='-',color='k',linewidth=3)
#cbar4 = plt.colorbar(cs, ax=axs[1,1],shrink=0.95)
#cbar4.set_label('Cloud Top Temperature (K)',fontsize=16, fontweight='bold')


axs[0,0].set_title('lo')
axs[0,1].set_title('io')
axs[1,0].set_title('ltmpc')
axs[1,1].set_title('mpc')

cbar = fig.colorbar(cs, ax=axs.ravel().tolist())
cbar.set_label('Cloud Top Temperature (K)',fontsize=16, fontweight='bold')

plt.show()

# %%
fig, axs = plt.subplots(nrows=1,ncols=1,figsize =[15,12], subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)

extent = [-3, 3, -63, -57]

axs.set_extent(extent)
cs = axs.scatter(data_lo.lon[grid_index_lo], data_lo.lat[grid_index_lo], c=data_lo.time[grid_index_lo], cmap='Spectral_r', s=3, marker='s')
cs2 = axs.plot([-2.5,2.5,2.5,-2.5,-2.5],[-62.5,-62.5,-57.5,-57.5,-62.5],linestyle='-',color='k',linewidth=3)
cbar = plt.colorbar(cs, ax=axs,shrink=0.95)
cbar.set_label('Time',fontsize=16, fontweight='bold')

# %%

# %%

# %%

# %%
if index[0]:
    print('a')
else:
    print('nope')

# %%
# save data as xarray in netcdf files
#T_star_xr = xr.DataArray(data = T_star, coords = [lon_grid,lat_grid], dims = ['lon','lat'], name = 'Tstar')
#T_star_freq_xr = xr.DataArray(data = T_star_freq, coords = [lon_grid,lat_grid], dims = ['lon','lat'])
#T_median_lo_xr = xr.DataArray(data = T_median_lo, coords = [lon_grid,lat_grid], dims = ['lon','lat'])
#T_median_ltmpc_xr = xr.DataArray(data = T_median_ltmpc, coords = [lon_grid,lat_grid], dims = ['lon','lat'])

#T_star_xr.to_netcdf('output_files/Tstar/Tstar_5x5_'+str(year)+'_'+season+'.nc')
#T_star_freq_xr.to_netcdf('output_files/Tstar/Tstar_freq_5x5_'+str(year)+'_'+season+'.nc')
#T_median_lo_xr.to_netcdf('output_files/Tstar/T_median_lo_5x5_'+str(year)+'_'+season+'.nc')
#T_median_ltmpc_xr.to_netcdf('output_files/T_median_ltmpc_5x5_'+str(year)+'_'+season+'.nc')

# %%
fig, axs = plt.subplots(nrows=2,ncols=1,figsize =[15,12], subplot_kw={'projection': ccrs.PlateCarree()})

plt_CTT(fig, axs[0], lon2d, lat2d, T_star_total, 'T_star total (K)', vmin=240, vmax = 270, n_cmap=10)
plt_CTT(fig, axs[1], lon2d, lat2d, T_star_lo_ltmpc, 'T_star (K)', vmin=240, vmax = 270, n_cmap=10)

plt.savefig('Figures/Tstar/Tstar_total_5x5_'+str(year)+'_'+season+'.png',bbox_inches='tight')


# %%

# %%
# define grid
lat_grid = np.arange(-90,91,5)
lon_grid = np.arange(-180,181,5)

lon2d, lat2d = np.meshgrid(lon_grid,lat_grid,indexing='ij')

# %%
ds_summer = xr.open_dataset('T_star_5x5_summer.nc')
ds_winter = xr.open_dataset('T_star_5x5_winter.nc')

# %%
fig, axs = plt.subplots(nrows=3,ncols=1,figsize =[12,13], subplot_kw={'projection': ccrs.PlateCarree()})

plt_CTT(fig, axs[0], lon2d, lat2d, ds_summer['Tstar'], 'Summer T_star (K)', vmin=240, vmax = 270, n_cmap=30)
plt_CTT(fig, axs[1], lon2d, lat2d, ds_winter['Tstar'], 'Winter T_star (K)', vmin=240, vmax = 270, n_cmap=30)
plt_CTT(fig, axs[2], lon2d, lat2d, ds_summer['Tstar']-ds_winter['Tstar'], 'Summer minus Winter T_star (K)', vmin=-8, vmax = 8, n_cmap=30)

plt.savefig('Figures/T_star_5x5_2017_seasonal_diff.png',bbox_inches='tight')

# %%

# %%

# %%
print(longi, lati)
#print(idx_star)
#print(T_star[longi+180, lati+90])
print(np.shape(CTT_lo_grid),np.shape(CTT_ltmpc_grid))

# %%
fig = plt.figure(figsize = (8, 6))

bins = np.arange(235,271,1)

ax = sns.distplot(CTT_lo_grid,
                  bins=bins,
                  kde=True,
                  color='k',
                  label='Liquid-only',
                  hist_kws={"linewidth": 15,'alpha':1,"color": "steelblue","alpha":0.5})
ax2 = sns.distplot(CTT_ltmpc_grid,
                  bins=bins,
                  kde=True,
                  kde_kws={"color": "k", "lw": 3},
                  label='Liquid-top MPC',
                  hist_kws={"linewidth": 15,'alpha':1,"color": "teal","alpha":0.5})
ax.set_xlabel('Cloud Top Temperature (K)',fontsize=14,fontweight='bold')
ax.set_ylabel('Frequency',fontsize=14,fontweight='bold')
#ax.set_ylim(0,0.2)
ax.set_xlim(275,235)
plt.xticks(np.arange(235, 275+1, 5))

ax.legend(loc='upper right',fontsize=14)

plt.savefig('Figures/Histogram_year_gridbox_5_summer_3000.png',bbox_inches='tight')

x_lo,y_lo = ax.get_lines()[0].get_data()
x_ltmpc,y_ltmpc = ax2.get_lines()[1].get_data()

y_new = scipy.interpolate.griddata(x_ltmpc, y_ltmpc, x_lo)


idx = np.argwhere(np.diff(np.sign(y_new - y_lo))).flatten()

plt.plot(x_lo[idx[0]], y_new[idx[0]], 'ro')
plt.plot([x_lo[idx[0]],x_lo[idx[0]]],[0,y_new[idx[0]]],'--',c='r')

plt.savefig('Figures/T_star_year_gridbox_5_summer_3000.png',bbox_inches='tight')

#plt.show()

sys.exit()
