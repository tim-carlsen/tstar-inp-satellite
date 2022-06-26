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
from scipy import stats

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

from cmcrameri import cm as cm_crameri
import xesmf as xe


import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D



def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
    
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
# Read in averaged and re-gridded sea ice data (from 10_sea_ice_regrid_final.py)

ds = xr.open_dataset('output_files/sea_ice/ice_s_djf_regrid.nc')
ice_s_djf = xr.where(ds.sic_djf >= 15.0, 100.0, ds.sic_djf)
ice_lon = ds.lon
ice_lat = ds.lat

ds = xr.open_dataset('output_files/sea_ice/ice_s_jja_regrid.nc')
ice_s_jja = xr.where(ds.sic_jja >= 15.0, 100.0, ds.sic_jja)
ice_lon = ds.lon
ice_lat = ds.lat


#################################
ds = xr.open_dataset('output_files/sea_ice/ice_n_jja_regrid.nc')
ice_n_jja = xr.where(ds.sic_jja >= 15.0, 100.0, ds.sic_jja)
ice_lon = ds.lon
ice_lat = ds.lat

ds = xr.open_dataset('output_files/sea_ice/ice_n_djf_regrid.nc')
ice_n_djf = xr.where(ds.sic_djf >= 15.0, 100.0, ds.sic_djf)
ice_lon = ds.lon
ice_lat = ds.lat



# +
# read T* data
file = data_path_base+'/tcarlsen/05_INP_SO/output_files/T_star/seasonal/Tstar_5x5_total_DJF.nc'
ds = xr.open_dataset(file)
T_star_djf = ds.Tstar

file = data_path_base+'/tcarlsen/05_INP_SO/output_files/T_star/seasonal/Tstar_5x5_total_JJA.nc'
ds = xr.open_dataset(file)
T_star_jja = ds.Tstar


# mask nan values
mask_nan_djf = np.ma.masked_greater(T_star_djf.fillna(0.), 1.0)
mask_nan_jja = np.ma.masked_greater(T_star_jja.fillna(0.), 1.0)

# +
# hatch out non-significant T*/DeltaT* pixels
ds = xr.open_dataset('output_files/bootstrap_spread/tstar_bootstrap_stderr_DJF_100.nc')
data_djf = ds.tstar_err * 2.58

ds = xr.open_dataset('output_files/bootstrap_spread/tstar_bootstrap_stderr_JJA_100.nc')
data_jja = ds.tstar_err * 2.58

# data with CI less than half the bin size (1°C)
mask_djf = np.ma.masked_less(data_djf, 0.5)
mask_jja = np.ma.masked_less(data_jja, 0.5)

# significance for summer - winter
mask_diff_n = np.ma.masked_less(data_djf + data_jja, np.abs(T_star_jja - T_star_djf))
mask_diff_s = np.ma.masked_less(data_djf + data_jja, np.abs(T_star_djf - T_star_jja))


# +
# Plotting
lat_grid = np.arange(-90,91,5)
lon_grid = np.arange(-180,181,5)
lon2d, lat2d = np.meshgrid(lon_grid,lat_grid,indexing='ij')


fig, axs = plt.subplots(nrows=2,ncols=3,figsize =[16,12], subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)})

fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                    wspace=0.02, hspace=0.03)


#######################################
# T* (K)
cs = plt_var_polar(fig, axs[0,0], lon2d, lat2d, T_star_jja, 'T_star JJA', 'N', discrete_cmap(16, cm_crameri.oslo), vmin = 238, vmax = 270, n_cmap = 16)

cs2 = plt_var_polar(fig, axs[0,1], lon2d, lat2d, T_star_djf, 'T_star DJF', 'N', discrete_cmap(16, cm_crameri.oslo), vmin = 238, vmax = 270, n_cmap = 16)

cs3 = plt_var_polar(fig, axs[0,2], lon2d, lat2d, T_star_jja - T_star_djf, 'T_star JJA-DJF', 'N', discrete_cmap(20, cm_crameri.vik), vmin = -20, vmax = 20, n_cmap = 20)


#######################################
# Sea ice edge contour line
cs = axs[0,0].contour(ice_lon, ice_lat, ice_n_jja, transform = ccrs.PlateCarree(), levels = [15.], colors = 'greenyellow')

cs = axs[0,1].contour(ice_lon, ice_lat, ice_n_djf, transform = ccrs.PlateCarree(), levels = [15.], colors = 'greenyellow')

########################################
# hatch insignificant data
# hatch insignificant data
cs5 = plt_var_polar(fig, axs[0,0], lon2d, lat2d, mask_jja, 'insignificant DJF', 'N', discrete_cmap(2, 'Greys'), vmin = 238, vmax = 270, n_cmap = 1)
cs5 = plt_var_polar(fig, axs[0,0], lon2d, lat2d, mask_nan_jja, 'insignificant DJF', 'N', discrete_cmap(2, 'Greys'), vmin = 238, vmax = 270, n_cmap = 1)

cs6 = plt_var_polar(fig, axs[0,1], lon2d, lat2d, mask_djf, 'insignificant JJA', 'N', discrete_cmap(2, 'Greys'), vmin = 238, vmax = 270, n_cmap = 1)
cs6 = plt_var_polar(fig, axs[0,1], lon2d, lat2d, mask_nan_djf, 'insignificant DJF', 'N', discrete_cmap(2, 'Greys'), vmin = 238, vmax = 270, n_cmap = 1)

cs7 = plt_var_polar(fig, axs[0,2], lon2d, lat2d, mask_diff_n, 'insignificant DJF', 'N', discrete_cmap(2, 'Greys'), vmin = 238, vmax = 270, n_cmap = 1)
cs7 = plt_var_polar(fig, axs[0,2], lon2d, lat2d, mask_djf, 'insignificant JJA', 'N', discrete_cmap(2, 'Greys'), vmin = 238, vmax = 270, n_cmap = 1)
cs7 = plt_var_polar(fig, axs[0,2], lon2d, lat2d, mask_jja, 'insignificant JJA', 'N', discrete_cmap(2, 'Greys'), vmin = 238, vmax = 270, n_cmap = 1)
cs7 = plt_var_polar(fig, axs[0,2], lon2d, lat2d, mask_nan_jja, 'insignificant DJF', 'N', discrete_cmap(2, 'Greys'), vmin = 238, vmax = 270, n_cmap = 1)
cs7 = plt_var_polar(fig, axs[0,2], lon2d, lat2d, mask_nan_djf, 'insignificant JJA', 'N', discrete_cmap(2, 'Greys'), vmin = 238, vmax = 270, n_cmap = 1)


cs5 = axs[0,0].pcolor(lon2d-2.5, lat2d-2.5, mask_jja, hatch='xxx', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())
cs5 = axs[0,0].pcolor(lon2d-2.5, lat2d-2.5, mask_nan_jja, hatch='...', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())

cs6 = axs[0,1].pcolor(lon2d-2.5, lat2d-2.5, mask_djf, hatch='xxx', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())
cs5 = axs[0,1].pcolor(lon2d-2.5, lat2d-2.5, mask_nan_djf, hatch='...', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())

cs7 = axs[0,2].pcolor(lon2d-2.5, lat2d-2.5, mask_jja, hatch='xxx', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())
cs7 = axs[0,2].pcolor(lon2d-2.5, lat2d-2.5, mask_diff_n, hatch='xxx', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())
cs7 = axs[0,2].pcolor(lon2d-2.5, lat2d-2.5, mask_djf, hatch='xxx', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())
cs5 = axs[0,2].pcolor(lon2d-2.5, lat2d-2.5, mask_nan_jja, hatch='...', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())
cs5 = axs[0,2].pcolor(lon2d-2.5, lat2d-2.5, mask_nan_djf, hatch='...', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())

#######################################
axs[1,0].remove()
axs[1,0] = fig.add_subplot(2, 3, 4, projection=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)) 
axs[1,1].remove()
axs[1,1] = fig.add_subplot(2, 3, 5, projection=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)) 
axs[1,2].remove()
axs[1,2] = fig.add_subplot(2, 3, 6, projection=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)) 

#######################################
# T* (K)
cs = plt_var_polar(fig, axs[1,0], lon2d, lat2d, T_star_djf, 'T_star DJF', 'S', discrete_cmap(16, cm_crameri.oslo), vmin = 238, vmax = 270, n_cmap = 16)

cs2 = plt_var_polar(fig, axs[1,1], lon2d, lat2d, T_star_jja, 'T_star JJA', 'S', discrete_cmap(16, cm_crameri.oslo), vmin = 238, vmax = 270, n_cmap = 16)


cs3 = plt_var_polar(fig, axs[1,2], lon2d, lat2d, T_star_djf - T_star_jja, 'T_star DJF-JJA', 'S', discrete_cmap(20, cm_crameri.vik), vmin = -20, vmax = 20, n_cmap = 20)


cb_ax = fig.add_axes([0.1, 0.08, 0.255, 0.03])
cbar = fig.colorbar(cs, cax=cb_ax, ticks = np.arange(238,272,5), orientation = 'horizontal')
cbar.set_ticklabels(['-35', '-30', '-25', '-20','-15','-10','-5']) 
cbar.set_label('T* (°C)', fontsize = 16)

cb_ax2 = fig.add_axes([0.37, 0.08, 0.255, 0.03])
cbar2 = fig.colorbar(cs2, cax=cb_ax2, ticks = np.arange(238,272,5), orientation = 'horizontal')
cbar2.set_ticklabels(['-35', '-30', '-25', '-20','-15','-10','-5']) 
cbar2.set_label('T* (°C)', fontsize = 16)

cb_ax3 = fig.add_axes([0.64, 0.08, 0.255, 0.03])
cbar3 = fig.colorbar(cs3, cax=cb_ax3, ticks = np.arange(-20,21,5), orientation = 'horizontal')
#cbar3.set_ticklabels(['-35', '-30', '-25', '-20','-15','-10','-5']) 
cbar3.set_label('$\Delta$T* (°C)',fontsize=16)


#######################################
# Sea ice edge contour line
cs = axs[1,0].contour(ice_lon, ice_lat, ice_s_djf, transform = ccrs.PlateCarree(), levels = [15.], colors = 'greenyellow')

cs = axs[1,1].contour(ice_lon, ice_lat, ice_s_jja, transform = ccrs.PlateCarree(), levels = [15.], colors = 'greenyellow')

########################################
# hatch insignificant data
cs5 = plt_var_polar(fig, axs[1,0], lon2d, lat2d, mask_djf, 'insignificant DJF', 'S', discrete_cmap(2, 'Greys'), vmin = 238, vmax = 270, n_cmap = 1)
cs5 = plt_var_polar(fig, axs[1,0], lon2d, lat2d, mask_nan_djf, 'insignificant DJF', 'S', discrete_cmap(2, 'Greys'), vmin = 238, vmax = 270, n_cmap = 1)

cs6 = plt_var_polar(fig, axs[1,1], lon2d, lat2d, mask_jja, 'insignificant JJA', 'S', discrete_cmap(2, 'Greys'), vmin = 238, vmax = 270, n_cmap = 1)
cs6 = plt_var_polar(fig, axs[1,1], lon2d, lat2d, mask_nan_jja, 'insignificant DJF', 'S', discrete_cmap(2, 'Greys'), vmin = 238, vmax = 270, n_cmap = 1)

cs7 = plt_var_polar(fig, axs[1,2], lon2d, lat2d, mask_djf, 'insignificant DJF', 'S', discrete_cmap(2, 'Greys'), vmin = 238, vmax = 270, n_cmap = 1)
cs7 = plt_var_polar(fig, axs[1,2], lon2d, lat2d, mask_jja, 'insignificant DJF', 'S', discrete_cmap(2, 'Greys'), vmin = 238, vmax = 270, n_cmap = 1)
cs7 = plt_var_polar(fig, axs[1,2], lon2d, lat2d, mask_diff_s, 'insignificant JJA', 'S', discrete_cmap(2, 'Greys'), vmin = 238, vmax = 270, n_cmap = 1)
cs7 = plt_var_polar(fig, axs[1,2], lon2d, lat2d, mask_nan_djf, 'insignificant DJF', 'S', discrete_cmap(2, 'Greys'), vmin = 238, vmax = 270, n_cmap = 1)
cs7 = plt_var_polar(fig, axs[1,2], lon2d, lat2d, mask_nan_jja, 'insignificant JJA', 'S', discrete_cmap(2, 'Greys'), vmin = 238, vmax = 270, n_cmap = 1)


cs5 = axs[1,0].pcolor(lon2d-2.5, lat2d-2.5, mask_djf, hatch='xxx', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())
cs5 = axs[1,0].pcolor(lon2d-2.5, lat2d-2.5, mask_nan_djf, hatch='...', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())

cs6 = axs[1,1].pcolor(lon2d-2.5, lat2d-2.5, mask_jja, hatch='xxx', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())
cs5 = axs[1,1].pcolor(lon2d-2.5, lat2d-2.5, mask_nan_jja, hatch='...', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())

cs7 = axs[1,2].pcolor(lon2d-2.5, lat2d-2.5, mask_djf, hatch='xxx', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())
cs7 = axs[1,2].pcolor(lon2d-2.5, lat2d-2.5, mask_jja, hatch='xxx', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())
cs7 = axs[1,2].pcolor(lon2d-2.5, lat2d-2.5, mask_diff_s, hatch='xxx', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())
cs5 = axs[1,2].pcolor(lon2d-2.5, lat2d-2.5, mask_nan_djf, hatch='...', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())
cs5 = axs[1,2].pcolor(lon2d-2.5, lat2d-2.5, mask_nan_jja, hatch='...', color = 'k', zorder = 100000, alpha=0.0 ,transform = ccrs.PlateCarree())


########################################


#legend_elements = [Patch(facecolor='white', edgecolor='k', hatch = 'xxx',
     #                    label='Insignificant'), 
     #             Patch(facecolor='white', edgecolor='k', hatch = '...',
    #                     label='Insignificant'),
    #              Patch(facecolor='greenyellow',edgecolor='tab:blue', lw=3)]
            
legend_elements = [Patch(facecolor='white', edgecolor='k', hatch = 'xxx',
                         label='Insignificant'), 
                  Patch(facecolor='white', edgecolor='k', hatch = '...',
                         label='Insignificant'),
                  Patch(facecolor='white',edgecolor='k', label = 'sea ice edge'),
                 Line2D([0], [0], color='greenyellow', lw=4, label = 'sea ice edge')]


axs[1,1].legend(ncol = 3, bbox_to_anchor=(0.5, -0.25), loc='upper center', handles = [legend_elements[0],legend_elements[1],legend_elements[3]], labels = ['Insignificant', 'No data', 'sea ice edge'], handlelength=1.5, handleheight=1.5, fontsize = 16)

legend_elements = [Patch(facecolor='white', edgecolor='k', hatch = 'xxx',
                         label='Insignificant'), 
                  Patch(facecolor='white', edgecolor='k', hatch = '...',
                         label='Insignificant'),
                  Line2D([0], [0], color='greenyellow', lw=4, label = 'sea ice edge', alpha = 0.5)]

#axs[1].legend(ncol = 3, bbox_to_anchor=(1.04, -0.25), loc='upper left', handles = legend_elements,  handlelength=1.5, handleheight=1.5)

########################################
axs[0,0].text(0.0,1.01,'A',fontsize = 22,transform=axs[0,0].transAxes)
axs[0,1].text(0.0,1.01,'B',fontsize = 22,transform=axs[0,1].transAxes)
axs[0,2].text(0.0,1.01,'C',fontsize = 22,transform=axs[0,2].transAxes);

axs[1,0].text(0.0,1.01,'D',fontsize = 22,transform=axs[1,0].transAxes)
axs[1,1].text(0.0,1.01,'E',fontsize = 22,transform=axs[1,1].transAxes)
axs[1,2].text(0.0,1.01,'F',fontsize = 22,transform=axs[1,2].transAxes);


axs[0,0].set_title('Summer (JJA)', fontsize = 16)
axs[0,1].set_title('Winter (DJF)', fontsize = 16)
axs[0,2].set_title('Summer - Winter (JJA-DJF)', fontsize = 16)
axs[1,0].set_title('Summer (DJF)', fontsize = 16)
axs[1,1].set_title('Winter (JJA)', fontsize = 16)
axs[1,2].set_title('Summer - Winter (DJF-JJA)', fontsize = 16)
########################################


plt.savefig('Figures/final_paper/Figure2.pdf',format = 'pdf', bbox_inches='tight')

sys.exit()

