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


#from cmcrameri import cm as cm_crameri

import seaborn as sns


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
        ax.set_extent([-180, 180, 90, 40], ccrs.PlateCarree())

    gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels=False, linewidth = 1.4, color = 'k', linestyle = '--')
    gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
    
    if hemisphere == 'S':
        gl.ylocator = mticker.FixedLocator([-90, -80, -70, -60, -50, -40])
    if hemisphere == 'N':    
        gl.ylocator = mticker.FixedLocator([90, 80, 70, 60, 50, 40])

    gl.n_steps = 50
    
    ax.set_axis_off()
    
   

    cs = ax.pcolormesh(lon2d, lat2d, var, cmap = custom_cmap, vmin = vmin, vmax = vmax, transform = ccrs.PlateCarree())
        
        
    return(cs)
 




# +
# read T* data

file = data_path_base+'/tcarlsen/05_INP_SO/output_files/T_star/seasonal/Tstar_5x5_total_DJF.nc'
ds = xr.open_dataset(file)
T_star_djf = ds.Tstar

file = data_path_base+'/tcarlsen/05_INP_SO/output_files/T_star/seasonal/Tstar_5x5_total_JJA.nc'
ds = xr.open_dataset(file)
T_star_jja = ds.Tstar

# +
# hatch out non-significant T*/DeltaT* pixels
ds = xr.open_dataset('output_files/bootstrap_spread/tstar_bootstrap_stderr_DJF_100.nc')
data_djf = ds.tstar_err * 2.58

ds = xr.open_dataset('output_files/bootstrap_spread/tstar_bootstrap_stderr_JJA_100.nc')
data_jja = ds.tstar_err * 2.58

# data with CI less than half the bin size (1°C)
mask_djf = np.ma.masked_less(data_djf, 0.5)
mask_jja = np.ma.masked_less(data_jja, 0.5)

# +
T_star_djf_mean = T_star_djf.where(data_djf <= 0.5).groupby('lat').mean(dim=xr.ALL_DIMS)
T_star_djf_std = T_star_djf.where(data_djf <= 0.5).groupby('lat').std(dim=xr.ALL_DIMS)

T_star_jja_mean = T_star_jja.where(data_jja <= 0.5).groupby('lat').mean(dim=xr.ALL_DIMS)
T_star_jja_std = T_star_jja.where(data_jja <= 0.5).groupby('lat').std(dim=xr.ALL_DIMS)

# -
lat_grid = np.arange(-90,91,5)
lon_grid = np.arange(-180,181,5)
lon2d, lat2d = np.meshgrid(lon_grid,lat_grid,indexing='ij')


# +
fig, axs = plt.subplots(nrows=1,ncols=1,figsize =[12,6])



# Summer (DJF)
axs.plot(lat_grid, T_star_djf_mean - 273., label = 'Summer (DJF)', color = 'r')
axs.scatter(lat_grid, T_star_djf_mean - 273., color = 'r')

y1 = T_star_djf_mean-T_star_djf_std-273.
y2 = T_star_djf_mean+T_star_djf_std-273.
axs.fill_between(lat_grid, y1, y2,color='r',alpha=.5)

# Winter (JJA)
axs.plot(lat_grid, T_star_jja_mean - 273., label = 'Winter (JJA)', color = 'blue')
axs.scatter(lat_grid, T_star_jja_mean - 273., color = 'blue')

y1 = T_star_jja_mean-T_star_jja_std-273.
y2 = T_star_jja_mean+T_star_jja_std-273.
axs.fill_between(lat_grid, y1, y2,color='blue',alpha=.5)



# sea ice edges
axs.plot([-70.,-70.],[-40.,0.],linestyle = '--',color = 'k')
axs.plot([-60.,-60.],[-40.,0.],linestyle = '--',color = 'k')

axs.text(-69.9,-11.,'Sea ice (summer)',fontsize = 14)
axs.text(-59.9,-11.,'Sea ice (winter)',fontsize = 14)



axs.set_xlim(-80, -50)
axs.set_ylim(-30, -10)

axs.set_xticks(np.arange(-80,-49,5))
axs.set_xticklabels(['80°S', '75°S', '70°S', '65°S','60°S','55°S','50°S'], fontsize = 14) 

axs.set_yticks(np.arange(-30,-9,5))

#axs.set_xlabel('Latitude', fontsize = 14)
axs.set_ylabel('T* average (°C)', fontsize = 16)



legend_elements = [Patch(facecolor='red', edgecolor='k', label='Insignificant', alpha=.5), 
                   Line2D([0], [0], color='red', lw=4, label = 'Summer (DJF)'),
                   Patch(facecolor='blue', label='Insignificant', alpha=.5),
                 Line2D([0], [0], color='blue', lw=4, label = 'Winter (JJA)')]


axs.legend(bbox_to_anchor=(0, 1.0), loc='upper left', handles = [(legend_elements[0],legend_elements[1]),(legend_elements[2],legend_elements[3])], labels = ['Summer (DJF)', 'Winter (JJA)'], handlelength=2.5, handleheight=1.5, fontsize = 14)

            
axs.tick_params(left = True, right = False, bottom = True, top = False)

axs.text(0.97,0.95,'A',fontsize = 22,transform=axs.transAxes)

            
sns.despine()
        

plt.savefig('Figures/final_paper/Figure3.pdf',format = 'pdf', bbox_inches='tight') 

    


# -






