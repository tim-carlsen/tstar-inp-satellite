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

import netCDF4

import xarray as xr

import math

import datetime

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from cmcrameri import cm as cm_crameri


label_size=14
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


# -

##########################################    
def plt_var_polar(fig, ax, lon2d, lat2d, var, label, hemisphere, custom_cmap, vmin, vmax, n_cmap):
    
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.coastlines(resolution='110m', linewidth = 2.0)
    
    if hemisphere == 'S':
        ax.set_extent([-180, 180, -90, -42.5], ccrs.PlateCarree())
    if hemisphere == 'N':
        ax.set_extent([-180, 180, 90, 42.5], ccrs.PlateCarree())

    gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels=False, linewidth = 1.4, color = 'k', linestyle = '--')
    gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
    
    if hemisphere == 'S':
        gl.ylocator = mticker.FixedLocator([-90, -80, -70, -60, -50, -40])
    if hemisphere == 'N':    
        gl.ylocator = mticker.FixedLocator([90, 80, 70, 60, 50, 40])

    gl.n_steps = 50
    
    #ax.set_axis_off()
    
   

    cs = ax.pcolormesh(lon2d, lat2d, var, cmap = custom_cmap, vmin = vmin, vmax = vmax, alpha = 1.0, transform = ccrs.PlateCarree())
        
        
    return(cs)

lat_grid = np.arange(-90,91,5)
lon_grid = np.arange(-180,181,5)
lon2d, lat2d = np.meshgrid(lon_grid,lat_grid,indexing='ij')

ds = xr.open_dataset('output_files/n_obs_seasonal/n_obs_seasonal.nc')

ds

n_lo_total = ds.n_lo[0,:,:]+ds.n_lo[1,:,:]+ds.n_lo[2,:,:]+ds.n_lo[3,:,:]
n_mpc_total = ds.n_mpc[0,:,:]+ds.n_mpc[1,:,:]+ds.n_mpc[2,:,:]+ds.n_mpc[3,:,:]
n_ltmpc_total = ds.n_ltmpc[0,:,:]+ds.n_ltmpc[1,:,:]+ds.n_ltmpc[2,:,:]+ds.n_ltmpc[3,:,:]
n_io_total = ds.n_io[0,:,:]+ds.n_io[1,:,:]+ds.n_io[2,:,:]+ds.n_io[3,:,:]
n_total = n_lo_total + n_mpc_total + n_io_total    #LTMPC are part of MPC


# +
fig, axs = plt.subplots(nrows=1,ncols=2,figsize =[12,6], subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)})

cs = plt_var_polar(fig, axs[0], lon2d, lat2d, n_ltmpc_total/n_mpc_total * 100., 'LT ratio', 'N', discrete_cmap(10, cm_crameri.lajolla), vmin = 0, vmax = 100, n_cmap = 10)

axs[1].remove()
axs[1] = fig.add_subplot(1, 2, 2, projection=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)) 

cs = plt_var_polar(fig, axs[1], lon2d, lat2d, n_ltmpc_total/n_mpc_total * 100., 'LT ratio', 'S', discrete_cmap(10, cm_crameri.lajolla), vmin = 0, vmax = 100, n_cmap = 10)

cb_ax = fig.add_axes([0.3, 0.08, 0.4, 0.03])
cbar = fig.colorbar(cs, cax=cb_ax, ticks = np.arange(0,101,10), orientation = 'horizontal')
cbar.set_ticklabels(['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']) 
cbar.set_label('Fraction of MPCs with liquid top (%)', fontsize = 16)

axs[0].set_title('Arctic', fontsize = 16)
axs[1].set_title('Antarctica', fontsize = 16)

axs[0].text(0.0,1.01,'A',fontsize = 22,transform=axs[0].transAxes)
axs[1].text(0.0,1.01,'B',fontsize = 22,transform=axs[1].transAxes)

mask_nan = np.ma.masked_greater(n_ltmpc_total, 1.0)

cs5 = axs[0].pcolor(lon2d-2.5, lat2d-2.5, mask_nan, hatch='...', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())
cs5 = axs[1].pcolor(lon2d-2.5, lat2d-2.5, mask_nan, hatch='...', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())

legend_elements = [ Patch(facecolor='white', edgecolor='k', hatch = '...',
                         label='No data')]


axs[1].legend(ncol = 1, bbox_to_anchor=(0.8, 0.0), loc='upper center', handles = [legend_elements[0]], labels = ['No data'], handlelength=1.5, handleheight=1.5, fontsize = 16)




# +
plt.savefig('Figures/final_paper/FigureS4_LTratio.pdf',format = 'pdf', bbox_inches='tight')

sys.exit()
# -





# +
season = 'DJF'
latitude = -60
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
# -

data_lo.phase

# +
fig, axs = plt.subplots(nrows=1,ncols=2,figsize =[10,5])


axs[0].bar([1], [len(data_lo.CTT)], width=0.5, bottom=None, align='center', ec="k", label = 'LO:       N = '+str(len(data_lo.CTT)))
axs[0].bar([2], [len(data_ltmpc.CTT)], color = 'r', alpha = 0.6, width=0.5, bottom=None, align='center', ec="k", label = 'LTMPC: N = '+str(len(data_ltmpc.CTT)))

#axs[0] = sns.distplot(data_lo.phase,
#                  bins=[1,2,3],
#                  kde=False,
#                  norm_hist=False,
#                  color='k',
#                  ax = axs[0],
#                  hist_kws={"linewidth": 15,'alpha':1,"color": "teal","alpha":1.0,"align":'left',"rwidth":0.7,"log":False})

axs[0].set_xlim(0,3)
axs[0].set_ylim(0, 25000)
axs[0].set_xticks([1,2])
axs[0].set_xticklabels(['liquid-only', 'LTMPC'])
axs[0].set_ylabel('Occurence (#)',fontsize=14)
axs[0].set_xlabel('',fontsize=14)


axs[0].legend(loc='upper right', fontsize = 12, frameon = True)


axs[1].remove()
axs[1] = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())  

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
#axs[1].set_boundary(circle, transform=axs[1].transAxes)

axs[1].coastlines(resolution='110m', linewidth = 2.0)
    
axs[1].set_extent([-10, 10, -75, -55], ccrs.PlateCarree())

gl = axs[1].gridlines(crs = ccrs.PlateCarree(), draw_labels=True, linewidth = 1., color = 'grey', linestyle = '--')
gl.xlocator = mticker.FixedLocator([-20, -15, -10, -5, 0, 5, 10, 15, 20])
gl.ylocator = mticker.FixedLocator([-80, -75, -70, -65, -60, -55, -50])
gl.n_steps = 50
    



cs3 = axs[1].plot([-2.5,2.5,2.5,-2.5,-2.5],[-62.5,-62.5,-57.5,-57.5,-62.5],linestyle='-',color='k',linewidth=4.5)



for latitude in [-55, -60, -65, -70]:
    for longitude in [-10, -5, 0, 5, 10]:
        file_pixel_lo = 'output_files/pixel_data/seasonal_total/pixel_'+season+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_lo.nc'
        data_lo = xr.open_dataset(file_pixel_lo)
        cs = axs[1].scatter(data_lo.lon, data_lo.lat, c='grey', s = 0.01, marker = '.')



latitude = -60
longitude = 0
file_pixel_lo = 'output_files/pixel_data/seasonal_total/pixel_'+season+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_lo.nc'
data_lo = xr.open_dataset(file_pixel_lo)

cs = axs[1].scatter(data_ltmpc.lon, data_ltmpc.lat, c='r', alpha = 0.6, s=2, marker='.')
cs = axs[1].scatter(data_lo.lon, data_lo.lat, s=0.1, marker='.')




sns.despine()

plt.savefig('Figures/Illustration_overpasses.png',bbox_inches='tight')
plt.show()

# -

def count_clouds_seasonal(longitude, latitude, season):
    
    try:
        file_pixel_lo = 'output_files/pixel_data/seasonal_total/pixel_'+season+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_lo.nc'
        file_pixel_mpc = 'output_files/pixel_data/seasonal_total/pixel_'+season+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_mpc.nc'
        file_pixel_ltmpc = 'output_files/pixel_data/seasonal_total/pixel_'+season+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_ltmpc.nc'
        file_pixel_io = 'output_files/pixel_data/seasonal_total/pixel_'+season+'_'+str(latitude).zfill(3)+'_'+str(longitude).zfill(3)+'_data_io.nc'
    
        # read data
        data_lo = xr.open_dataset(file_pixel_lo)
        data_mpc = xr.open_dataset(file_pixel_mpc)
        data_ltmpc = xr.open_dataset(file_pixel_ltmpc)
        data_io = xr.open_dataset(file_pixel_io)
        
        return([len(data_lo.CTT), len(data_mpc.CTT), len(data_ltmpc.CTT), len(data_io.CTT)])
    
    except:
        return([0, 0, 0, 0])



lat_grid = np.arange(-90,91,5)
lon_grid = np.arange(-180,181,5)
lon2d, lat2d = np.meshgrid(lon_grid,lat_grid,indexing='ij')

# +
n_lo = np.zeros([4, len(lon_grid), len(lat_grid)],dtype=np.float64)
n_mpc = np.zeros([4, len(lon_grid), len(lat_grid)],dtype=np.float64)
n_ltmpc = np.zeros([4, len(lon_grid), len(lat_grid)],dtype=np.float64)
n_io = np.zeros([4, len(lon_grid), len(lat_grid)],dtype=np.float64)



# perform bootstrapping for each pixel and save the spread of bootstrapped T* calculations (max - min)
count_longi = 0
count_lati = 0

for longi in np.arange(-180,181,5):
    
    print(longi)
        
    for lati in np.arange(-90,91,5):
        
        try:
            # DJF
            result = count_clouds_seasonal(longi, lati, 'DJF')
            n_lo[0, count_longi, count_lati] += result[0]
            n_mpc[0, count_longi, count_lati] += result[1]
            n_ltmpc[0, count_longi, count_lati] += result[2]
            n_io[0, count_longi, count_lati] += result[3]
            
            # MAM
            result = count_clouds_seasonal(longi, lati, 'MAM')
            n_lo[1, count_longi, count_lati] += result[0]
            n_mpc[1, count_longi, count_lati] += result[1]
            n_ltmpc[1, count_longi, count_lati] += result[2]
            n_io[1, count_longi, count_lati] += result[3]
            
            # JJA
            result = count_clouds_seasonal(longi, lati, 'JJA')
            n_lo[2, count_longi, count_lati] += result[0]
            n_mpc[2, count_longi, count_lati] += result[1]
            n_ltmpc[2, count_longi, count_lati] += result[2]
            n_io[2, count_longi, count_lati] += result[3]
            
            # SON
            result = count_clouds_seasonal(longi, lati, 'SON')
            n_lo[3, count_longi, count_lati] += result[0]
            n_mpc[3, count_longi, count_lati] += result[1]
            n_ltmpc[3, count_longi, count_lati] += result[2]
            n_io[3, count_longi, count_lati] += result[3]

        except:
                            
            count_lati += 1
            continue
        
        count_lati += 1
        
        
    count_longi += 1
    count_lati = 0
# -

np.nanmean(n_lo)

# +
ds = xr.Dataset(

    data_vars=dict(

        n_lo=(["season", "x", "y"], n_lo),
        n_mpc=(["season", "x", "y"], n_mpc),
        n_ltmpc=(["season", "x", "y"], n_ltmpc),
        n_io=(["season", "x", "y"], n_io),

    ),

    coords=dict(
        season = ['DJF', 'MAM', 'JJA', 'SON'],
        
        lon=(["x", "y"], lon2d),

        lat=(["x", "y"], lat2d),

    ),

    attrs=dict(description="Number of registered clouds per cloud type (LO, MPC, LTMPC, IO) and season (DJF, MAM, JJA, SON) between 2006-2017."),

)
# -

ds.to_netcdf('output_files/n_obs_seasonal/n_obs_seasonal.nc')

ds = xr.open_dataset('output_files/n_obs_seasonal/n_obs_seasonal.nc')

np.nanmax(ds.n_lo)

ds.n_lo


##########################################    
def plt_var_global(fig, ax, lon2d, lat2d, var, label, custom_cmap, vmin, vmax):
    
    
    ax.coastlines(resolution='110m', linewidth = 2.0)

    gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels=False, linewidth = 1.4, color = 'k', linestyle = '--')
    gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
    gl.n_steps = 50    
   
    cs = ax.pcolormesh(lon2d, lat2d, var, cmap = custom_cmap, vmin = vmin, vmax = vmax, alpha = 1.0, transform = ccrs.PlateCarree())
    ax.set_title(label)    
        
    return(cs)

# +
n_lo_total = ds.n_lo[0,:,:]+ds.n_lo[1,:,:]+ds.n_lo[2,:,:]+ds.n_lo[3,:,:]
n_mpc_total = ds.n_mpc[0,:,:]+ds.n_mpc[1,:,:]+ds.n_mpc[2,:,:]+ds.n_mpc[3,:,:]
n_ltmpc_total = ds.n_ltmpc[0,:,:]+ds.n_ltmpc[1,:,:]+ds.n_ltmpc[2,:,:]+ds.n_ltmpc[3,:,:]
n_io_total = ds.n_io[0,:,:]+ds.n_io[1,:,:]+ds.n_io[2,:,:]+ds.n_io[3,:,:]
n_total = n_lo_total + n_mpc_total + n_io_total    #LTMPC are part of MPC



fig, axs = plt.subplots(nrows=3,ncols=5,figsize =[25, 8], subplot_kw={'projection': ccrs.PlateCarree()})

# Total
#axs[0,0].set_title('Total', fontsize = 16)
#axs[0,1].set_title('DJF', fontsize = 16)
#axs[0,2].set_title('MAM', fontsize = 16)
#axs[0,3].set_title('JJA', fontsize = 16)
#axs[0,4].set_title('SON', fontsize = 16)

cs = plt_var_global(fig, axs[0,0], lon2d, lat2d, n_mpc_total/n_total * 100., 'MPC total', discrete_cmap(20, 'Spectral_r'), 0, 100)
cs = plt_var_global(fig, axs[0,1], lon2d, lat2d, ds.n_mpc[0,:,:]/(ds.n_lo[0,:,:] + ds.n_mpc[0,:,:] + ds.n_io[0,:,:]) * 100., 'MPC DJF', discrete_cmap(20, 'Spectral_r'), 0, 100)
cs = plt_var_global(fig, axs[0,2], lon2d, lat2d, ds.n_mpc[1,:,:]/(ds.n_lo[1,:,:] + ds.n_mpc[1,:,:] + ds.n_io[1,:,:]) * 100., 'MPC MAM', discrete_cmap(20, 'Spectral_r'), 0, 100)
cs = plt_var_global(fig, axs[0,3], lon2d, lat2d, ds.n_mpc[2,:,:]/(ds.n_lo[2,:,:] + ds.n_mpc[2,:,:] + ds.n_io[2,:,:]) * 100., 'MPC JJA', discrete_cmap(20, 'Spectral_r'), 0, 100)
cs = plt_var_global(fig, axs[0,4], lon2d, lat2d, ds.n_mpc[3,:,:]/(ds.n_lo[3,:,:] + ds.n_mpc[3,:,:] + ds.n_io[3,:,:]) * 100., 'MPC SON', discrete_cmap(20, 'Spectral_r'), 0, 100)

cs = plt_var_global(fig, axs[1,0], lon2d, lat2d, n_ltmpc_total/n_total * 100., 'LTMPC total', discrete_cmap(20, 'Spectral_r'), 0, 100)
cs = plt_var_global(fig, axs[1,1], lon2d, lat2d, ds.n_ltmpc[0,:,:]/(ds.n_lo[0,:,:] + ds.n_mpc[0,:,:] + ds.n_io[0,:,:]) * 100., 'LTMPC DJF', discrete_cmap(20, 'Spectral_r'), 0, 100)
cs = plt_var_global(fig, axs[1,2], lon2d, lat2d, ds.n_ltmpc[1,:,:]/(ds.n_lo[1,:,:] + ds.n_mpc[1,:,:] + ds.n_io[1,:,:]) * 100., 'LTMPC MAM', discrete_cmap(20, 'Spectral_r'), 0, 100)
cs = plt_var_global(fig, axs[1,3], lon2d, lat2d, ds.n_ltmpc[2,:,:]/(ds.n_lo[2,:,:] + ds.n_mpc[2,:,:] + ds.n_io[2,:,:]) * 100., 'LTMPC JJA', discrete_cmap(20, 'Spectral_r'), 0, 100)
cs = plt_var_global(fig, axs[1,4], lon2d, lat2d, ds.n_ltmpc[3,:,:]/(ds.n_lo[3,:,:] + ds.n_mpc[3,:,:] + ds.n_io[3,:,:]) * 100., 'LTMPC SON', discrete_cmap(20, 'Spectral_r'), 0, 100)

cs = plt_var_global(fig, axs[2,0], lon2d, lat2d, n_ltmpc_total/n_mpc_total * 100., 'LT ratio total', discrete_cmap(20, 'Spectral_r'), 0, 100)
cs = plt_var_global(fig, axs[2,1], lon2d, lat2d, ds.n_ltmpc[0,:,:]/ds.n_mpc[0,:,:] * 100., 'LT ratio DJF', discrete_cmap(20, 'Spectral_r'), 0, 100)
cs = plt_var_global(fig, axs[2,2], lon2d, lat2d, ds.n_ltmpc[1,:,:]/ds.n_mpc[1,:,:] * 100., 'LT ratio MAM', discrete_cmap(20, 'Spectral_r'), 0, 100)
cs = plt_var_global(fig, axs[2,3], lon2d, lat2d, ds.n_ltmpc[2,:,:]/ds.n_mpc[2,:,:] * 100., 'LT ratio JJA', discrete_cmap(20, 'Spectral_r'), 0, 100)
cs = plt_var_global(fig, axs[2,4], lon2d, lat2d, ds.n_ltmpc[3,:,:]/ds.n_mpc[3,:,:] * 100., 'LT ratio SON', discrete_cmap(20, 'Spectral_r'), 0, 100)




cb_ax = fig.add_axes([0.3, 0.08, 0.4, 0.03])
cbar = fig.colorbar(cs, cax=cb_ax, ticks = np.arange(0,101,20), orientation = 'horizontal')
cbar.set_ticklabels(['0', '20', '40', '60','80','100']) 
cbar.set_label('Frequency (%)', fontsize = 16)


# -

##########################################    
def plt_var_polar(fig, ax, lon2d, lat2d, var, label, hemisphere, custom_cmap, vmin, vmax, n_cmap):
    
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.coastlines(resolution='110m', linewidth = 2.0)
    
    if hemisphere == 'S':
        ax.set_extent([-180, 180, -90, -42.5], ccrs.PlateCarree())
    if hemisphere == 'N':
        ax.set_extent([-180, 180, 90, 42.5], ccrs.PlateCarree())

    gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels=False, linewidth = 1.4, color = 'k', linestyle = '--')
    gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
    
    if hemisphere == 'S':
        gl.ylocator = mticker.FixedLocator([-90, -80, -70, -60, -50, -40])
    if hemisphere == 'N':    
        gl.ylocator = mticker.FixedLocator([90, 80, 70, 60, 50, 40])

    gl.n_steps = 50
    
    #ax.set_axis_off()
    
   

    cs = ax.pcolormesh(lon2d, lat2d, var, cmap = custom_cmap, vmin = vmin, vmax = vmax, alpha = 1.0, transform = ccrs.PlateCarree())
        
        
    return(cs)

# +
fig, axs = plt.subplots(nrows=2,ncols=1,figsize =[12,6], subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)})

cs = plt_var_polar(fig, axs[0], lon2d, lat2d, n_ltmpc_total/n_mpc_total * 100., 'LT ratio', 'N', discrete_cmap(10, cm_crameri.oslo), vmin = 0, vmax = 100, n_cmap = 10)

axs[1].remove()
axs[1] = fig.add_subplot(1, 2, 2, projection=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)) 

cs = plt_var_polar(fig, axs[1], lon2d, lat2d, n_ltmpc_total/n_mpc_total * 100., 'LT ratio', 'S', discrete_cmap(10, cm_crameri.oslo), vmin = 0, vmax = 100, n_cmap = 10)

# -





# +
# save data as xarray in netcdf files    
n_obs_xr = xr.DataArray(data = [n_lo], coords = [lon_grid,lat_grid], dims = ['lon','lat'], name = 'tstar_spread')

T_star_spread_xr.to_netcdf('output_files/n_obs_seasonal/n_obs_seasonal.nc')
# -





# +
season = 'DJF'
latitude = -60
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

freq_lo = hist_lo/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
freq_mpc = hist_mpc/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
freq_ltmpc = hist_ltmpc/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
freq_io = hist_io/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())

# +
fig, axs = plt.subplots(nrows=1,ncols=2,figsize =[11,5])
    
bins = np.arange(235,271,2)-273.

with plt.xkcd():

    #cs = axs[0].bar(bins[:-1],100.*freq_lo,width=2, align="edge", ec="k", label = 'LO:       N = '+str(len(data_lo.CTT)))
    #axs[1].bar(bins[:-1],100.*freq_ltmpc,width=2, align="edge", ec="k", color = 'r', alpha = 0.6, label = 'LTMPC: N = '+str(len(data_ltmpc.CTT)))
    
    #axs[0].plot(bins[:-1],100.*freq_lo)
    #axs[1].plot(bins[:-1],100.*freq_ltmpc, color = 'r', alpha = 0.6)
    from scipy.signal import savgol_filter
    yhat = savgol_filter(100.*np.append(freq_ltmpc,[0.002, 0]), 5, 1)
    axs[1].plot(np.append(bins[:-1],[-4, 2]),1.4*yhat, color = 'r', alpha = 0.6, label = 'LTMPC')

    
    temp = np.arange(-40, 0, 0.1);
    amplitude = 6*np.cos(0.08*temp)
    axs[0].plot(temp,amplitude, label = 'LO')

        
    for k in np.arange(2):
        axs[k].set_xlim(-3, 0)
        axs[k].set_ylim(0,9)
         
        axs[k].set_xticks(np.arange(-40,1,10))
        axs[k].set_xticks(np.arange(-40,1,5), minor = True)

            
        axs[k].set_xlabel('Cloud top temperature (°C)', fontsize = 14)
        axs[k].set_ylabel('Frequency of cloud type (%)', fontsize = 14)
            
        axs[k].legend(loc='upper left', bbox_to_anchor=(0.025, 0.9), fontsize = 12, frameon = True)
            
        axs[k].tick_params(left = True, right = False, bottom = True, top = False)
            
            

        
        
    plt.setp(axs[0].get_yticklabels(), visible=False)
    plt.setp(axs[1].get_yticklabels(), visible=False)

    
    import seaborn as sns
    sns.despine()
    
    plt.savefig('Figures/Illustration_theory.png',bbox_inches='tight')
    plt.show()
# -

np.append(bins[:-1],[-4,0])

freq_ltmpc

np.append(freq_ltmpc,[0.001, 0])








