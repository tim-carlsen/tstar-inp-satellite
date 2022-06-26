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


from cmcrameri import cm as cm_crameri


##########################################    
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
def plot_methods(season, latitude, longitude):
    print(season)
    print('longitude: ', longitude)
    print('latitude: ', latitude)
    print('===========================================')
    
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
    
    print(np.nanmax(data_lo.CTT - 273.))
    print(np.nanmax(data_ltmpc.CTT - 273.))
    print('CTT was filtered to be below 270K (not too close to algorithm limits), that is why we just use full bins starting at 269K (-4°C)')


    hist_lo, edges_lo = np.histogram(data_lo.CTT, bins);
    hist_mpc, edges_mpc = np.histogram(data_mpc.CTT, bins);
    hist_ltmpc, edges_lo = np.histogram(data_ltmpc.CTT, bins);
    hist_io, edges_io = np.histogram(data_io.CTT, bins);

    freq_lo_total = hist_lo/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
    freq_mpc_total = hist_mpc/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
    freq_ltmpc_total = hist_ltmpc/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
    freq_io_total = hist_io/float(hist_lo.sum() + hist_mpc.sum() + hist_io.sum())
    
    # plotting histograms
    plot_pixel_histogram(season, longitude, latitude, freq_lo_total, freq_mpc_total, freq_ltmpc_total, freq_io_total, len(data_lo.CTT), len(data_mpc.CTT), len(data_ltmpc.CTT), len(data_io.CTT))

                 
    # Calculate T* again from data
    #index = np.array(np.where(freq_lo_total-freq_ltmpc_total < 0)).flatten();
    index = np.array(np.where((freq_lo_total-freq_ltmpc_total < 0) & (hist_ltmpc > 10))).flatten();    ####### NEW !!! ###########

                         
    if len(index) == 0:
        print('T* (re-calculated): nan')
    else:
        print('T* (re-calculated): ', np.max(bins[index]))
        
        
        
##########################################      
def plot_pixel_histogram(season, longitude, latitude, freq_lo, freq_mpc, freq_ltmpc, freq_io, n_lo, n_mpc, n_ltmpc, n_io):
    fig, axs = plt.subplots(nrows=1,ncols=4,figsize =[22,5])
    
    bins = np.arange(235,271,2)-273.


    if (n_lo > 0) or (n_mpc > 0) or (n_ltmpc > 0) or (n_io > 0): 
        axs[0].bar(bins[:-1],100.*freq_lo,width=2, align="edge", ec="k", label = 'LO: N = '+str(n_lo))
        axs[1].bar(bins[:-1],100.*freq_ltmpc,width=2, align="edge", ec="k", color = 'r', alpha = 0.6, label = 'LTMPC: N = '+str(n_ltmpc))
        axs[2].bar(bins[:-1],100.*freq_lo,width=2, align="edge", ec="k", label = 'LO:       N = '+str(n_lo))
        axs[2].bar(bins[:-1],100.*freq_ltmpc,width=2, align="edge", ec="k", color = 'r', alpha = 0.6, label = 'LTMPC: N = '+str(n_ltmpc))
        
        print(bins[:-1])
        
        for k in np.arange(3):
            axs[k].set_xlim(-40, 0)
            axs[k].set_ylim(0,8)
         
            axs[k].set_xticks(np.arange(-40,1,10))
            axs[k].set_xticks(np.arange(-40,1,5), minor = True)

            
            axs[k].set_xlabel('Cloud top temperature (°C)', fontsize = 14)
            axs[k].set_ylabel('Frequency of cloud type (%)', fontsize = 14)
            
            axs[k].legend(loc='upper left', bbox_to_anchor=(0.025, 0.9), fontsize = 12, frameon = True)
            
            axs[k].tick_params(left = True, right = False, bottom = True, top = False)
            
            

        axs[2].arrow(-13., 4.5, 0, -1, head_length = 0.3, head_width = 1., fc = 'k', ec = 'k')
        axs[2].text(-25,4.2,'T* = -13°C', fontsize = 14)
        
        
        import seaborn as sns
        sns.despine()
        
        
    axs[3].remove()
    axs[3] = fig.add_subplot(1, 4, 4, projection=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None))   
    
    # show hemisphere view of pixel
    lat_grid = np.arange(-90,91,5)
    lon_grid = np.arange(-180,181,5)
    lon2d, lat2d = np.meshgrid(lon_grid,lat_grid,indexing='ij')
    
    #custom_cmap = cm.get_cmap('Spectral_r', 16)
    custom_cmap = discrete_cmap(16, cm_crameri.oslo)

    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    axs[3].set_boundary(circle, transform=axs[3].transAxes)

    axs[3].coastlines(resolution='110m', linewidth = 2.0)
    
    axs[3].set_extent([-180, 180, -90, -40], ccrs.PlateCarree())

    gl = axs[3].gridlines(crs = ccrs.PlateCarree(), draw_labels=False, linewidth = 1.4, color = 'k', linestyle = '--')
    gl.xlocator = mticker.FixedLocator([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
    gl.ylocator = mticker.FixedLocator([-90, -80, -70, -60, -50, -40])
    gl.n_steps = 50
    
    #axs[3].set_axis_off()
    

    import matplotlib.patches as mpatches
    axs[3].add_patch(mpatches.Rectangle(xy=[-2.5, -62.5], width=5, height=5,
                                    #facecolor=(1.0,1.0,1.0,0.0),
                                    edgecolor = 'k',
                                    lw = 4.5,
                                    alpha=1.0,
                                    transform=ccrs.PlateCarree(), zorder = 5)
                 )
    
    cs = axs[3].pcolormesh(lon2d, lat2d, T_star, cmap=custom_cmap, vmin = 238, vmax = 270, transform = ccrs.PlateCarree(), zorder = 10)


    cb_ax = fig.add_axes([0.73, 0.09, 0.17, 0.03])
    cbar = fig.colorbar(cs, cax=cb_ax, ticks = np.arange(238,272,5), orientation = 'horizontal')
    #cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    cbar.set_ticklabels(['-35', '-30', '-25', '-20','-15','-10','-5']) 
    #cbar.set_label('T* (K)',fontsize=14)
    
    
    axs[0].text(0.05,0.93,'A',fontsize = 22,transform=axs[0].transAxes)
    axs[1].text(0.05,0.93,'B',fontsize = 22,transform=axs[1].transAxes)
    axs[2].text(0.05,0.93,'C',fontsize = 22,transform=axs[2].transAxes)
    axs[3].text(0.05,0.93,'D',fontsize = 22,transform=axs[3].transAxes)
    axs[3].text(0.03,0.02,'T* (°C)',fontsize = 14,transform=axs[3].transAxes)

        
    plt.savefig('Figures/final_paper/Figure1.pdf',format = 'pdf', bbox_inches='tight')

    #plt.close()  
    
    
    #axs[2].bar(bins[:-1],100.*freq_ltmpc,width=2, align="edge", ec='red', color = 'red', alpha = 0.0, label = 'LTMPC: N = '+str(n_ltmpc))
    #axs[2].bar(bins[:-1],100.*freq_lo,width=2, align="edge", color = (0,0,1.0,0.0), ec='blue', label = 'LO:       N = '+str(n_lo), linewidth = 3.0)
    #axs[2].bar(bins[:-1],100.*freq_ltmpc,width=2, align="edge", color = (1.0,0,0,0.0), ec = 'r', label = 'LTMPC: N = '+str(n_ltmpc), linewidth = 3.0)

    
  
 




# +
file = data_path_base+'/tcarlsen/05_INP_SO/output_files/T_star/seasonal/Tstar_5x5_total_DJF.nc'

ds = xr.open_dataset(file)
T_star = ds.Tstar
T_star = np.array(T_star)
T_star[:,:] = np.nan
T_star[36,6] = 260.
# -

plot_methods('DJF',-60,0)














