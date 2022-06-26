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
 

def get_wghts(lat):

    latr = np.deg2rad(lat) # convert to radians

    weights = np.cos(latr) # calc weights

    return weights




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
# Read-in and regrid land mask
ds_grid = xr.open_dataset('LongitudeLatitudeGrid-n6250-Arctic.hdf')

lon = ds_grid.Longitudes
lat = ds_grid.Latitudes

ds_grid = xr.open_dataset('LongitudeLatitudeGrid-s6250-Antarctic.hdf')

lon_s = ds_grid.Longitudes
lat_s = ds_grid.Latitudes
# -

ds_n = xr.open_dataset(data_path_base+'/data/sea_ice/landmask_Arctic_6.250km.nc')
ds_s = xr.open_dataset(data_path_base+'/data/sea_ice/landmask_Antarctic_6.250km.nc')

# +
# 1 = ocean, 0 = land

land_mask_n = xr.DataArray(ds_n.z,coords={'lon': (["x","y"], lon),
                          'lat': (["x","y"], lat)},
                  dims=['x','y'], name = 'land_mask')

land_mask_s = xr.DataArray(ds_s.z,coords={'lon': (["x","y"], lon_s),
                          'lat': (["x","y"], lat_s)},
                  dims=['x','y'], name = 'land_mask')

# +
# Arctic
ds_land_mask = land_mask_n.to_dataset(dim=None, name='land_mask')
dr_land_mask = ds_land_mask.land_mask

ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90., 91., 5.0)),
                     'lon': (['lon'], np.arange(-180., 181., 5.0)),
                    }
                   )
regridder = xe.Regridder(ds_land_mask, ds_out, 'bilinear')
dr_out = regridder(dr_land_mask)
land_mask_n_regrid = dr_out.T

# Antarctic
ds_land_mask = land_mask_s.to_dataset(dim=None, name='land_mask')
dr_land_mask = ds_land_mask.land_mask

ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90., 91., 5.0)),
                     'lon': (['lon'], np.arange(-180., 181., 5.0)),
                    }
                   )
regridder = xe.Regridder(ds_land_mask, ds_out, 'bilinear')
dr_out = regridder(dr_land_mask)
land_mask_s_regrid = dr_out.T

# +
lat_grid = np.arange(-90,91,5)
lon_grid = np.arange(-180,181,5)
lon2d, lat2d = np.meshgrid(lon_grid,lat_grid,indexing='ij')


fig, axs = plt.subplots(nrows=2,ncols=2,figsize =[16,12], subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)})

fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                    wspace=0.02, hspace=0.03)


cs = plt_var_polar(fig, axs[0,0], land_mask_n.lon, land_mask_n.lat, land_mask_n, 'T_star JJA', 'N', discrete_cmap(2, 'Spectral_r'), vmin = 0, vmax = 1, n_cmap = 2)
cs2 = plt_var_polar(fig, axs[0,1], lon2d, lat2d, land_mask_n_regrid, 'T_star JJA', 'N', discrete_cmap(2, 'Spectral_r'), vmin = 0, vmax = 1, n_cmap = 2)


axs[1,0].remove()
axs[1,0] = fig.add_subplot(2, 2, 3, projection=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)) 
axs[1,1].remove()
axs[1,1] = fig.add_subplot(2, 2, 4, projection=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)) 
#######################################
# T* (K)
cs = plt_var_polar(fig, axs[1,0], land_mask_s.lon, land_mask_s.lat, land_mask_s, 'T_star JJA', 'S', discrete_cmap(2, 'Spectral_r'), vmin = 0, vmax = 1, n_cmap = 2)
cs2 = plt_var_polar(fig, axs[1,1], lon2d, lat2d, land_mask_s_regrid, 'T_star JJA', 'S', discrete_cmap(2, 'Spectral_r'), vmin = 0, vmax = 1, n_cmap = 2)

plt.savefig('landmasks.png')



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
ice_n_djf_ds = ice_n_djf.to_dataset(dim=None, name='sic_djf')
ds_out = xr.Dataset({'lon': (['lon'], np.arange(-180, 181, 5)),
                     'lat': (['lat'], np.arange(-90, 91, 5)),
                    }
                   )
regridder = xe.Regridder(ice_n_djf_ds, ds_out, 'bilinear')

dr_out = regridder(ice_n_djf).T
ice_n_djf_regrid = xr.where(dr_out >= 15.0, 100.0, dr_out)

dr_out = regridder(ice_n_jja).T
ice_n_jja_regrid = xr.where(dr_out >= 15.0, 100.0, dr_out)

dr_out = regridder(ice_s_djf).T
ice_s_djf_regrid = xr.where(dr_out >= 15.0, 100.0, dr_out)

dr_out = regridder(ice_s_jja).T
ice_s_jja_regrid = xr.where(dr_out >= 15.0, 100.0, dr_out)
# -

mask_antarctic_jja_ice = T_star_jja.where(((ice_s_jja_regrid == 0.) & (T_star_jja.lon == 0.) & (T_star_jja.lat <= -55.)) | ((ice_s_jja_regrid == 100.) & (T_star_jja.lat <= -55.) ))
mask_antarctic_jja_ocean = T_star_jja.where((T_star_jja.lon != 0) & (ice_s_jja_regrid == 0.) & (T_star_jja.lat <= -55.))


# lat_grid = np.arange(-90,91,5)
# lon_grid = np.arange(-180,181,5)
# lon2d, lat2d = np.meshgrid(lon_grid,lat_grid,indexing='ij')
#
#
# count = 0
# count_longi = 0
# count_lati = 0
#
# for longi in np.arange(-180,181,5):
#     
#     print(longi)
#         
#     for lati in np.arange(-90,-30,5):
#         
#         #if lati == -60.:
#             #print(longi, lati, mask_antarctic_jja_ice[count_longi, count_lati], np.isfinite(mask_antarctic_jja_ice[count_longi, count_lati]))
#         #print(mask_antarctic_jja_ice[count_longi, count_lati])
#         if mask_antarctic_jja_ice[count_longi, count_lati] > 0:
#         
#             try:
#                 result = histogram_masks(season, lati, longi)
#                 print('a')
#             
#                 if count == 0:
#                     hist_lo_mask = result[0]
#                     hist_mpc_mask = result[1]
#                     hist_ltmpc_mask = result[2]
#                     hist_io_mask = result[3]
#                     count += 1
#                 else:
#                     hist_lo_mask += result[0]
#                     hist_mpc_mask += result[1]
#                     hist_ltmpc_mask += result[2]
#                     hist_io_mask += result[3]
#                     count += 1
#
#             except:
#                 count_lati += 1
#                 continue
#         
#         
#         count_lati += 1
#         
#         
#     count_longi += 1
#     count_lati = 0


# freq_lo_mask = hist_lo_mask/float(hist_lo_mask.sum() + hist_mpc_mask.sum() + hist_io_mask.sum())
# freq_mpc_mask = hist_mpc_mask/float(hist_lo_mask.sum() + hist_mpc_mask.sum() + hist_io_mask.sum())
# freq_ltmpc_mask = hist_ltmpc_mask/float(hist_lo_mask.sum() + hist_mpc_mask.sum() + hist_io_mask.sum())
# freq_io_mask = hist_io_mask/float(hist_lo_mask.sum() + hist_mpc_mask.sum() + hist_io_mask.sum())
#     
# # plotting histograms
# plot_pixel_histogram(freq_lo_mask, freq_mpc_mask, freq_ltmpc_mask, freq_io_mask, hist_lo_mask.sum(), hist_mpc_mask.sum(), hist_ltmpc_mask.sum(), hist_io_mask.sum())
#
# print('Pixels in mask: ', count)

# +
#sys.exit()
# -









# +
lat_grid = np.arange(-90,91,5)
lon_grid = np.arange(-180,181,5)
lon2d, lat2d = np.meshgrid(lon_grid,lat_grid,indexing='ij')


fig, axs = plt.subplots(nrows=2,ncols=2,figsize =[12,12], subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)})

fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                    wspace=0.02, hspace=0.03)


#######################################
# T* (K)
cs = plt_var_polar(fig, axs[0,0], lon2d-2.5, lat2d-2.5, T_star_jja.where((ice_n_jja_regrid == 100.) & (T_star_jja.lat > 60.)), 'T_star JJA', 'N', discrete_cmap(16, cm_crameri.oslo), vmin = 238, vmax = 270, n_cmap = 16)
cs2 = plt_var_polar(fig, axs[0,1], lon2d-2.5, lat2d-2.5, T_star_djf.where((ice_n_djf_regrid == 100.) & (T_star_djf.lat > 60.)), 'T_star JJA', 'N', discrete_cmap(16, cm_crameri.oslo), vmin = 238, vmax = 270, n_cmap = 16)


axs[1,0].remove()
axs[1,0] = fig.add_subplot(2, 2, 3, projection=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)) 
axs[1,1].remove()
axs[1,1] = fig.add_subplot(2, 2, 4, projection=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)) 

cs3 = plt_var_polar(fig, axs[1,0], lon2d-2.5, lat2d-2.5, T_star_djf.where((ice_s_djf_regrid == 100.) & (T_star_djf.lat < -50.)), 'T_star JJA', 'S', discrete_cmap(16, cm_crameri.oslo), vmin = 238, vmax = 270, n_cmap = 16)
cs4 = plt_var_polar(fig, axs[1,1], lon2d-2.5, lat2d-2.5, T_star_jja.where((ice_s_jja_regrid == 100.) & (T_star_jja.lat < -50.)), 'T_star JJA', 'S', discrete_cmap(16, cm_crameri.oslo), vmin = 238, vmax = 270, n_cmap = 16)

axs[0,0].set_title('Summer (JJA)', fontsize = 16)
axs[0,1].set_title('Winter (DJF)', fontsize = 16)
axs[1,0].set_title('Summer (DJF)', fontsize = 16)
axs[1,1].set_title('Winter (JJA)', fontsize = 16)

plt.savefig('sea_ice_regrid.png')



# +
# Filtering
mask_arctic_jja_all = T_star_jja.where((T_star_jja.lat >= 60.) & (data_jja < 0.5))
mask_arctic_jja_ocean = T_star_jja.where((land_mask_n_regrid == 1.) & (T_star_jja.lat >= 60.) & (data_jja < 0.5))
mask_arctic_jja_land = T_star_jja.where((land_mask_n_regrid == 0.) & (T_star_jja.lat >= 60.) & (data_jja < 0.5))

mask_arctic_djf_ice = T_star_djf.where((ice_n_djf_regrid == 100.) & (T_star_djf.lat >= 60.) & (data_djf < 0.5))
mask_arctic_djf_ocean = T_star_djf.where((ice_n_djf_regrid == 0.) & (T_star_djf.lat >= 60.) & (data_djf < 0.5))
mask_arctic_djf_land = T_star_djf.where((land_mask_n_regrid == 0.) & (T_star_djf.lat >= 60.) & (data_djf < 0.5))


mask_antarctic_djf_ocean = T_star_djf.where((land_mask_s_regrid > 0.5) & (T_star_djf.lat <= -55.) & (T_star_djf.lat >= -75.) & (data_djf < 0.5))
mask_antarctic_jja_ice = T_star_jja.where(((ice_s_jja_regrid == 0.) & (T_star_jja.lon == 0.) & (T_star_jja.lat <= -55.)) | ((ice_s_jja_regrid == 100.) & (T_star_jja.lat <= -55.) & (data_jja < 0.5)))
mask_antarctic_jja_ocean = T_star_jja.where((T_star_jja.lon != 0) & (ice_s_jja_regrid == 0.) & (T_star_jja.lat <= -55.) & (data_jja < 0.5))
mask_antarctic_jja_land = T_star_jja.where((land_mask_s_regrid == 0) & (T_star_jja.lat <= -55.) & (data_jja < 0.5))



# +
lat_grid = np.arange(-90,91,5)
lon_grid = np.arange(-180,181,5)
lon2d, lat2d = np.meshgrid(lon_grid,lat_grid,indexing='ij')


fig, axs = plt.subplots(nrows=2,ncols=5,figsize =[25,12], subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)})

fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                    wspace=0.02, hspace=0.03)


#######################################
# T* (K)
cs = plt_var_polar(fig, axs[0,0], lon2d, lat2d, mask_arctic_jja_ocean, 'Arctic JJA ocean', 'N', discrete_cmap(16, cm_crameri.oslo), vmin = 238, vmax = 270, n_cmap = 16)
cs2 = plt_var_polar(fig, axs[0,1], lon2d, lat2d, mask_arctic_jja_land, 'Arctic JJA land', 'N', discrete_cmap(16, cm_crameri.oslo), vmin = 238, vmax = 270, n_cmap = 16)
cs3 = plt_var_polar(fig, axs[0,2], lon2d, lat2d, mask_arctic_djf_ice, 'Arctic DJF ice', 'N', discrete_cmap(16, cm_crameri.oslo), vmin = 238, vmax = 270, n_cmap = 16)
cs4 = plt_var_polar(fig, axs[0,3], lon2d, lat2d, mask_arctic_djf_ocean, 'Arctic DJF ocean', 'N', discrete_cmap(16, cm_crameri.oslo), vmin = 238, vmax = 270, n_cmap = 16)
cs5 = plt_var_polar(fig, axs[0,4], lon2d, lat2d, mask_arctic_djf_land, 'Arctic DJF land', 'N', discrete_cmap(16, cm_crameri.oslo), vmin = 238, vmax = 270, n_cmap = 16)


axs[1,0].remove()
axs[1,0] = fig.add_subplot(2, 5, 6, projection=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)) 
axs[1,1].remove()
#axs[1,1] = fig.add_subplot(2, 6, 8, projection=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)) 
axs[1,2].remove()
axs[1,2] = fig.add_subplot(2, 5, 8, projection=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)) 
axs[1,3].remove()
axs[1,3] = fig.add_subplot(2, 5, 9, projection=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)) 
axs[1,4].remove()
#axs[1,4] = fig.add_subplot(2, 6, 11, projection=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)) 


cs3 = plt_var_polar(fig, axs[1,0], lon2d, lat2d, mask_antarctic_djf_ocean, 'Antarctic DJF ocean', 'S', discrete_cmap(16, cm_crameri.oslo), vmin = 238, vmax = 270, n_cmap = 16)
cs4 = plt_var_polar(fig, axs[1,2], lon2d, lat2d, mask_antarctic_jja_ice, 'Antarctic JJA ice', 'S', discrete_cmap(16, cm_crameri.oslo), vmin = 238, vmax = 270, n_cmap = 16)
cs4 = plt_var_polar(fig, axs[1,3], lon2d, lat2d, mask_antarctic_jja_ocean, 'Antarctic JJA ocean', 'S', discrete_cmap(16, cm_crameri.oslo), vmin = 238, vmax = 270, n_cmap = 16)

axs[0,0].set_title('Summer (JJA): ocean', fontsize = 16)
axs[0,1].set_title('Summer (JJA): land', fontsize = 16)
axs[0,2].set_title('Winter (DJF): sea ice', fontsize = 16)
axs[0,3].set_title('Winter (DJF): open ocean', fontsize = 16)
axs[0,4].set_title('Winter (DJF): land', fontsize = 16)


axs[1,0].set_title('Summer (DJF): ocean', fontsize = 16)
axs[1,2].set_title('Winter (JJA): sea ice', fontsize = 16)
axs[1,3].set_title('Winter (JJA): open ocean', fontsize = 16)

cb_ax = fig.add_axes([0.35, 0.08, 0.3, 0.03])
cbar = fig.colorbar(cs, cax=cb_ax, ticks = np.arange(238,272,5), orientation = 'horizontal')
cbar.set_ticklabels(['-35', '-30', '-25', '-20','-15','-10','-5']) 
cbar.set_label('T* (°C)', fontsize = 16)

axs[0,0].text(0.0,1.01,'A',fontsize = 22,transform=axs[0,0].transAxes)
axs[0,1].text(0.0,1.01,'B',fontsize = 22,transform=axs[0,1].transAxes)
axs[0,2].text(0.0,1.01,'C',fontsize = 22,transform=axs[0,2].transAxes);
axs[0,3].text(0.0,1.01,'D',fontsize = 22,transform=axs[0,3].transAxes);
axs[0,4].text(0.0,1.01,'E',fontsize = 22,transform=axs[0,4].transAxes);

axs[1,0].text(0.0,1.01,'F',fontsize = 22,transform=axs[1,0].transAxes)
axs[1,2].text(0.0,1.01,'G',fontsize = 22,transform=axs[1,2].transAxes)
axs[1,3].text(0.0,1.01,'H',fontsize = 22,transform=axs[1,3].transAxes);



plt.savefig('Figures/final_paper/FigureS2_masks.pdf',format = 'pdf', bbox_inches='tight')


# +
lat_grid = np.arange(-90,91,5)
lon_grid = np.arange(-180,181,5)

a = np.zeros([73,37], dtype = float)
area = xr.DataArray(data = a, coords = [lon_grid,lat_grid], dims = ['lon','lat'], name = 'area')

count_longi = 0
count_lati = 0

for longi in np.arange(-180,181,5):
            
    for lati in np.arange(-90,91,5):
        
        area[count_longi, count_lati] = np.cos(np.deg2rad(lati))
        
        count_lati += 1
    
    count_longi += 1
    count_lati = 0


# +
mask_arctic_jja_ocean = T_star_jja.where((land_mask_n_regrid == 1.) & (T_star_jja.lat >= 60.) & (data_jja < 0.5))
mask_arctic_jja_land = T_star_jja.where((land_mask_n_regrid == 0.) & (T_star_jja.lat >= 60.) & (data_jja < 0.5))

mask_arctic_djf_ice = T_star_djf.where((ice_n_djf_regrid == 100.) & (T_star_djf.lat >= 60.) & (data_djf < 0.5))
mask_arctic_djf_ocean = T_star_djf.where((ice_n_djf_regrid == 0.) & (T_star_djf.lat >= 60.) & (data_djf < 0.5))
mask_arctic_djf_land = T_star_djf.where((land_mask_n_regrid == 0.) & (T_star_djf.lat >= 60.) & (data_djf < 0.5))


mask_antarctic_djf_ocean = T_star_djf.where((land_mask_s_regrid > 0.5) & (T_star_djf.lat <= -55.) & (T_star_djf.lat >= -75.) & (data_djf < 0.5))
mask_antarctic_jja_ice = T_star_jja.where(((ice_s_jja_regrid == 0.) & (T_star_jja.lon == 0.) & (T_star_jja.lat <= -55.)) | ((ice_s_jja_regrid == 100.) & (T_star_jja.lat <= -55.) & (data_jja < 0.5)))
mask_antarctic_jja_ocean = T_star_jja.where((T_star_jja.lon != 0) & (ice_s_jja_regrid == 0.) & (T_star_jja.lat <= -55.) & (data_jja < 0.5))
mask_antarctic_jja_land = T_star_jja.where((land_mask_s_regrid == 0) & (T_star_jja.lat <= -55.) & (data_jja < 0.5))

# +
##################### Area-weighted averaging #####################
print('################ Area-weighted averages ################')
print('')


###### Arctic:
print('##### Arctic #####')

# Summer (JJA): ocean
mask_avg = mask_arctic_jja_ocean
mask_area = area.where((land_mask_n_regrid == 1.) & (T_star_jja.lat >= 60.) & (data_jja < 0.5))

avg_sum = np.nansum(mask_area * mask_avg)
area_sum = np.nansum(mask_area)
avg_weighted = avg_sum / area_sum - 273.
print('Summer (JJA) - ocean: ', avg_weighted)

# Summer (JJA): land
mask_avg = mask_arctic_jja_land
mask_area = area.where((land_mask_n_regrid == 0.) & (T_star_jja.lat >= 60.) & (data_jja < 0.5))

avg_sum = np.nansum(mask_area * mask_avg)
area_sum = np.nansum(mask_area)
avg_weighted = avg_sum / area_sum - 273.
print('Summer (JJA) - land: ', avg_weighted)


# Winter (DJF): sea ice
mask_avg = mask_arctic_djf_ice
mask_area = area.where((ice_n_djf_regrid == 100.) & (T_star_djf.lat >= 60.) & (data_djf < 0.5))

avg_sum = np.nansum(mask_area * mask_avg)
area_sum = np.nansum(mask_area)
avg_weighted = avg_sum / area_sum - 273.
print('Winter (DJF) - sea ice: ', avg_weighted)

# Winter (DJF): open ocean
mask_avg = mask_arctic_djf_ocean
mask_area = area.where((ice_n_djf_regrid == 0.) & (T_star_djf.lat >= 60.) & (data_djf < 0.5))

avg_sum = np.nansum(mask_area * mask_avg)
area_sum = np.nansum(mask_area)
avg_weighted = avg_sum / area_sum - 273.
print('Winter (DJF) - open ocean: ', avg_weighted)

# Winter (DJF): land
mask_avg = mask_arctic_djf_land
mask_area = area.where((land_mask_n_regrid == 0.) & (T_star_djf.lat >= 60.) & (data_djf < 0.5))

avg_sum = np.nansum(mask_area * mask_avg)
area_sum = np.nansum(mask_area)
avg_weighted = avg_sum / area_sum - 273.
print('Winter (DJF) - land: ', avg_weighted)


#############################
print('')
print('##### Antarctica #####')
###### Antarctica:

# Summer (DJF): ocean
mask_avg = mask_antarctic_djf_ocean
mask_area = area.where((land_mask_s_regrid > 0.5) & (T_star_djf.lat <= -55.) & (T_star_djf.lat >= -75.) & (data_djf < 0.5))

avg_sum = np.nansum(mask_area * mask_avg)
area_sum = np.nansum(mask_area)
avg_weighted = avg_sum / area_sum - 273.
print('Summer (DJF) - ocean: ', avg_weighted)


# Winter (JJA): sea ice
mask_avg = mask_antarctic_jja_ice
mask_area = area.where(((ice_s_jja_regrid == 0.) & (T_star_jja.lon == 0.) & (T_star_jja.lat <= -55.)) | ((ice_s_jja_regrid == 100.) & (T_star_jja.lat <= -55.) & (data_jja < 0.5)))

avg_sum = np.nansum(mask_area * mask_avg)
area_sum = np.nansum(mask_area)
avg_weighted = avg_sum / area_sum - 273.
print('Winter (JJA) - sea ice: ', avg_weighted)


# Winter (JJA): open ocean
mask_avg = mask_antarctic_jja_ocean
mask_area = area.where((T_star_jja.lon != 0) & (ice_s_jja_regrid == 0.) & (T_star_jja.lat <= -55.) & (data_jja < 0.5))

avg_sum = np.nansum(mask_area * mask_avg)
area_sum = np.nansum(mask_area)
avg_weighted = avg_sum / area_sum - 273.
print('Winter (JJA) - open ocean: ', avg_weighted)


        
# -


