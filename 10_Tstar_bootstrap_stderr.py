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

from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
import matplotlib.path as mpath


import xarray as xr

import numpy as np
import numpy.ma as ma

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

    


# +
def plotting_distribution(freq_lo, freq_ltmpc, i, T_star):

    # plotting histograms
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize =[10,5])

    bins = np.arange(235,271,2)-273.

    axs[0].bar(bins[:-1],freq_lo*100.,width=2, align="edge", ec="k", label = 'LO: N = '+str(len(data_lo.CTT)))
    axs[0].bar(bins[:-1],freq_ltmpc*100.,width=2, align="edge", ec="k", color = 'r', alpha = 0.6, label = 'LTMPC: N = '+str(len(data_ltmpc.CTT)))
        
    axs[0].set_xticks(np.arange(-40,1,10))
    axs[0].set_xticks(np.arange(-40,1,5), minor = True)
    axs[0].set_xlim(-40, 0)
    axs[0].set_ylim(0,30)    
    axs[0].legend(loc='upper left', bbox_to_anchor=(0.025, 0.95), fontsize = 12, frameon = True)
    axs[0].set_xlabel('Cloud top temperature (°C)', fontsize = 14)
    axs[0].set_ylabel('Frequency of cloud type (%)', fontsize = 14)
    
    hist_tstar, edges_tstar = np.histogram(T_star-273., bins);
    axs[1].bar(bins[:-1],hist_tstar,width=2, align="edge", ec="k", color = 'teal')
    axs[1].set_ylim(0,100)
    axs[1].set_title('Bootstrapping: i = '+str(i), fontsize = 14)
    axs[1].set_xlabel('T* (°C)', fontsize = 14)
    axs[1].set_ylabel('Occurence of T* (#)', fontsize = 14)

    sns.despine()
    
    plt.savefig('Figures/bootstrap/bs_4_'+str(i).zfill(4)+'.png', dpi = 200)
    plt.close()

    
def bootstrap(population_lo, population_ltmpc, N):
    
    T_star = np.ones(N)
    T_star[:] = np.nan

    for i in np.arange(0, N, 1):
    
        # bootstrapping (random sample with replacement)
        n_lo = len(population_lo)
        n_ltmpc = len(population_ltmpc)
        sample_lo = np.random.choice(population_lo, n_lo)
        sample_ltmpc = np.random.choice(population_ltmpc, n_ltmpc)
        
        bins = np.arange(235,271,2)

        hist_lo, edges_lo = np.histogram(sample_lo, bins);
        hist_ltmpc, edges_lo = np.histogram(sample_ltmpc, bins);

        # new normalization!!!!!!!!!!!
        freq_lo_total = hist_lo/float(hist_lo.sum() + hist_ltmpc.sum())
        freq_ltmpc_total = hist_ltmpc/float(hist_lo.sum() + hist_ltmpc.sum())
    
        
        #calculate T* of new sample 
        index = np.array(np.where((freq_lo_total-freq_ltmpc_total < 0) & (hist_ltmpc > 10))).flatten();    ####### NEW !!! ###########
            
        if len(index) == 0:
            T_star[i] = np.nan                 
        else:
            T_star[i] = np.max(bins[index])
            
        
        #plotting_distribution(freq_lo_total, freq_ltmpc_total, i, T_star)

            
    return(T_star)

# +
lat_grid = np.arange(-90,91,5)
lon_grid = np.arange(-180,181,5)
lon2d, lat2d = np.meshgrid(lon_grid,lat_grid,indexing='ij')


T_star_err = np.ones([len(lon_grid), len(lat_grid)],dtype=np.float64)
T_star_err[:,:] = np.nan

season = str(sys.argv[1])
#season = 'DJF'

# perform bootstrapping for each pixel and save the spread of bootstrapped T* calculations (max - min)
count_longi = 0
count_lati = 0

for longi in np.arange(-180,181,5):
    
    print(longi)
        
    for lati in np.arange(-90,91,5):
            
        try:
            file_pixel_lo = 'output_files/pixel_data/seasonal_total/pixel_'+season+'_'+str(lati).zfill(3)+'_'+str(longi).zfill(3)+'_data_lo.nc'
            file_pixel_ltmpc = 'output_files/pixel_data/seasonal_total/pixel_'+season+'_'+str(lati).zfill(3)+'_'+str(longi).zfill(3)+'_data_ltmpc.nc'
    
            # read data
            data_lo = xr.open_dataset(file_pixel_lo)
            data_ltmpc = xr.open_dataset(file_pixel_ltmpc)
    
            # get histograms
            bins = np.arange(235,271,2)
            tstar_bs = bootstrap(data_lo.CTT, data_ltmpc.CTT, 100)
            T_star_err[count_longi, count_lati] = np.std(tstar_bs)/np.sqrt(100)
            
        except:
            T_star_err[count_longi, count_lati] = np.nan
                
            count_lati += 1
            continue
        
        
        count_lati += 1
        
        
    count_longi += 1
    count_lati = 0
        

        
# save data as xarray in netcdf files    
T_star_err_xr = xr.DataArray(data = T_star_err, coords = [lon_grid,lat_grid], dims = ['lon','lat'], name = 'tstar_err')

T_star_err_xr.to_netcdf('output_files/bootstrap_spread/tstar_bootstrap_stderr_'+season+'_100.nc')
# -

sys.exit()











# +
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.path as mpath
import matplotlib.ticker as mticker


import xarray as xr

import numpy as np
import numpy.ma as ma

import cartopy.crs as ccrs

import netCDF4

# define grid and read data
lat_grid = np.arange(-90,91,5)
lon_grid = np.arange(-180,181,5)
lon2d, lat2d = np.meshgrid(lon_grid,lat_grid,indexing='ij')

ds = xr.open_dataset('Figures/tstar_bootstrap_spread_DJF.nc')
data = ds.tstar_spread

# plot setup
fig, ax = plt.subplots(nrows=1,ncols=1,figsize =[8,8], subplot_kw={'projection': ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)})

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)
ax.coastlines(resolution='110m', linewidth = 2.0)
ax.set_extent([-180, 180, -90, -27.5], ccrs.PlateCarree())
    
gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels=False, linewidth = 1.4, color = 'k', linestyle = '--')
gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
gl.ylocator = mticker.FixedLocator([-90, -80, -70, -60, -50, -40, -30])
gl.n_steps = 50

ax.set_axis_off()

custom_cmap = cm.get_cmap('Spectral_r', 10)


# plotting
cs = ax.pcolormesh(lon2d-2.5, lat2d-2.5, data, cmap = custom_cmap, vmin = 0, vmax = 20, alpha = 1.0, transform = ccrs.PlateCarree())

# data with spread greater than 6 is insignificant
mask = np.ma.masked_less(data, 6.0)
significant = np.ma.masked_greater(data, 6.0)


cs = ax.pcolormesh(lon2d-2.5, lat2d-2.5, significant, cmap = custom_cmap, vmin = 0, vmax = 20, transform = ccrs.PlateCarree())

 
# hatch insignificant data
#hatch = plt.fill_between([xmin,xmax],y1,y2,hatch='///////',color="none",edgecolor='black')
#cs2 = ax.pcolormesh(lon2d-2.5, lat2d-2.5, mask, transform = ccrs.PlateCarree())
#index = np.where(data > 6.)
cs2 = ax.pcolor(lon2d-2.5, lat2d-2.5, mask, hatch='xxx', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())
#hatch = plt.plot(lon2d[index],lat2d[index],'x',color='k',markersize=5.5, transform = ccrs.PlateCarree())

cbar = fig.colorbar(cs, ticks = np.arange(0,21,2), orientation = 'vertical', shrink = 0.8)
cbar.set_label('Bootstrapping (N = 1000) T* spread (°C)', fontsize = 16)

#cbar.set_ticklabels(['-35', '-30', '-25', '-20','-15','-10','-5']) 

plt.savefig('Figures/tstar_bootstrap_spread_DJF.png', dpi = 200)


# +
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.path as mpath
import matplotlib.ticker as mticker


import xarray as xr

import numpy as np
import numpy.ma as ma

import cartopy.crs as ccrs

import netCDF4

# define grid and read data
lat_grid = np.arange(-90,91,5)
lon_grid = np.arange(-180,181,5)
lon2d, lat2d = np.meshgrid(lon_grid,lat_grid,indexing='ij')

ds = xr.open_dataset('Figures/tstar_bootstrap_spread_JJA.nc')
data = ds.tstar_spread

# plot setup
fig, ax = plt.subplots(nrows=1,ncols=1,figsize =[8,8], subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)})

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)
ax.coastlines(resolution='110m', linewidth = 2.0)
ax.set_extent([-180, 180, 90, 42.5], ccrs.PlateCarree())
    
gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels=False, linewidth = 1.4, color = 'k', linestyle = '--')
gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
gl.ylocator = mticker.FixedLocator([90, 80, 70, 60, 50, 40])
gl.n_steps = 50

ax.set_axis_off()

custom_cmap = cm.get_cmap('Spectral_r', 10)


# plotting
cs = ax.pcolormesh(lon2d-2.5, lat2d-2.5, data, cmap = custom_cmap, vmin = 0, vmax = 20, alpha = 0.2, transform = ccrs.PlateCarree())

# data with spread greater than 6 is insignificant
mask = np.ma.masked_less(data, 6.0)
significant = np.ma.masked_greater(data, 6.0)


cs = ax.pcolormesh(lon2d-2.5, lat2d-2.5, significant, cmap = custom_cmap, vmin = 0, vmax = 20, transform = ccrs.PlateCarree())

 
# hatch insignificant data
#hatch = plt.fill_between([xmin,xmax],y1,y2,hatch='///////',color="none",edgecolor='black')
#cs2 = ax.pcolormesh(lon2d-2.5, lat2d-2.5, mask, transform = ccrs.PlateCarree())
#index = np.where(data > 6.)
cs2 = ax.pcolor(lon2d-2.5, lat2d-2.5, mask, hatch='xxx', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())
#hatch = plt.plot(lon2d[index],lat2d[index],'x',color='k',markersize=5.5, transform = ccrs.PlateCarree())

cbar = fig.colorbar(cs, ticks = np.arange(0,21,2), orientation = 'vertical', shrink = 0.8)
cbar.set_label('Bootstrapping (N = 1000) T* spread (°C)', fontsize = 16)

#cbar.set_ticklabels(['-35', '-30', '-25', '-20','-15','-10','-5']) 


plt.savefig('Figures/tstar_bootstrap_spread_JJA_Arctic.png', dpi = 200)
# -





print(np.shape(mask), np.nanmax(mask), np.nanmin(mask))

np.shape(index)

lat2d[index[1]]

np.nanmin(data[index[0],index[1]])

np.shape(index)

np.shape(data)

np.shape(data)

ds





##########################################    
def plt_var_polar(fig, ax, lon2d, lat2d, var, label, hemisphere, custom_cmap, vmin, vmax, n_cmap):
    
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.coastlines(resolution='110m', linewidth = 2.0)
    
    if hemisphere == 'S':
        ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
    if hemisphere == 'N':
        ax.set_extent([-180, 180, 90, 40], ccrs.PlateCarree())

    gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels=False, linewidth = 1.4, color = 'k', linestyle = '--')
    gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
    
    if hemisphere == 'S':
        gl.ylocator = mticker.FixedLocator([-90, -80, -70, -60, -50, -40, -30])
    if hemisphere == 'N':    
        gl.ylocator = mticker.FixedLocator([90, 80, 70, 60, 50, 40])

    gl.n_steps = 50
    
    ax.set_axis_off()
    
   

    cs = ax.pcolormesh(lon2d, lat2d, var, cmap = custom_cmap, vmin = vmin, vmax = vmax, transform = ccrs.PlateCarree())
        
        
    return(cs)

# +
custom_cmap = cm.get_cmap('Spectral_r', 10)

T_star_spread = data


#fig, axs = plt.subplots(nrows=1,ncols=1,figsize =[8,8], subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)})

#cs = plt_var_polar(fig, axs, lon2d-2.5, lat2d-2.5, T_star_spread, 'T_star spread', 'N', custom_cmap, vmin = 0, vmax = 20, n_cmap = 10)

#cb_ax = fig.add_axes([0.92, 0.12, 0.05, 0.75])
#cbar = fig.colorbar(cs, cax=cb_ax,ticks = np.arange(0,21,2))
#cbar.set_label('Bootstrapping (N = 1000) spread T* (°C)', fontsize = 16)




fig, axs = plt.subplots(nrows=1,ncols=1,figsize =[8,8], subplot_kw={'projection': ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)})

cs2 = plt_var_polar(fig, axs, lon2d, lat2d, T_star_spread, 'T_star spread', 'S', custom_cmap, vmin = 0, vmax = 20, n_cmap = 10)
#cs2 = axs.pcolormesh(lon2d-2.5, lat2d-2.5, T_star_spread, cmap = custom_cmap, vmin = 0, vmax = 20, hatch = '.',transform = ccrs.PlateCarree())

zm = np.ma.masked_greater(T_star_spread, 6.0)

#cm = plt.pcolormesh(x, y, z)
#axs.pcolormesh(lon2d-2.5, lat2d-2.5, zm, hatch='x', zorder = 50, transform = ccrs.PlateCarree())


#cs = axs[0].pcolormesh(lon2d-2.5, lat2d-2.5, T_star_spread, cmap = custom_cmap, vmin = 0, vmax = 20, transform = ccrs.PlateCarree())
#cs = axs[1].pcolormesh(lon2d-2.5, lat2d-2.5, T_star_spread, cmap = custom_cmap, vmin = 0, vmax = 20, transform = ccrs.PlateCarree())

cb_ax = fig.add_axes([0.92, 0.12, 0.05, 0.75])
cbar = fig.colorbar(cs, cax=cb_ax,ticks = np.arange(0,21,2))
cbar.set_label('Bootstrapping (N = 1000) spread T* (°C)', fontsize = 16)
# -



np.nanmax(zm)

print(lon2d[36,6], lat2d[36,6], T_star_spread[36,6])
print(lon2d[36,12], lat2d[36,12], T_star_spread[36,12])





# +
season = 'DJF'
latitude = -30
longitude = 0

file_pixel_lo = 'output_files/pixel_data/seasonal_total/pixel_'+season+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_lo.nc'
file_pixel_mpc = 'output_files/pixel_data/seasonal_total/pixel_'+season+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_mpc.nc'
file_pixel_ltmpc = 'output_files/pixel_data/seasonal_total/pixel_'+season+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_ltmpc.nc'
file_pixel_io = 'output_files/pixel_data/seasonal_total/pixel_'+season+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_io.nc'
    
# read data
data_lo = xr.open_dataset(file_pixel_lo)
data_mpc = xr.open_dataset(file_pixel_mpc)
data_ltmpc = xr.open_dataset(file_pixel_ltmpc)
data_io = xr.open_dataset(file_pixel_io)
    
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

# new normalization
freq_lo_total = hist_lo/float(hist_lo.sum() + hist_ltmpc.sum())
freq_ltmpc_total = hist_ltmpc/float(hist_lo.sum() + hist_ltmpc.sum())

plotting_distribution(freq_lo_total, freq_ltmpc_total, freq_mpc_total, freq_io_total)

tstar = bootstrap(data_lo.CTT, data_ltmpc.CTT, data_mpc.CTT, data_io.CTT, 1000)
# -

print(np.shape(tstar), np.nanmax(tstar), np.nanmin(tstar), np.nanmean(tstar))

# +
bins = np.arange(235,271,2)
hist_tstar, edges_tstar = np.histogram(tstar, bins);
#freq_tstar = hist_tstar/float(hist_tstar.sum())


# plotting histograms
fig, axs = plt.subplots(nrows=1,ncols=1,figsize =[8,5])

axs.bar(bins[:-1], hist_tstar,width=2, align="edge", ec="k", label = 'tstar')
        
axs.set_xlim(275,235)
axs.set_ylim(0,1100.0)
axs.set_title(season+' lat: '+str(latitude)+' lon: '+str(longitude))
axs.legend()
# -

print(np.nanmax(tstar)-np.nanmin(tstar))













