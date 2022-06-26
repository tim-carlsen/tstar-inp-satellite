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

import xesmf as xe





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
    
    ax.set_axis_off()
    
   

    cs = ax.pcolormesh(lon2d, lat2d, var, cmap = custom_cmap, vmin = vmin, vmax = vmax, alpha = 1.0, transform = ccrs.PlateCarree())
        
        
    return(cs)
 




# +
ds_grid = xr.open_dataset('LongitudeLatitudeGrid-n6250-Arctic.hdf')

lon = ds_grid.Longitudes
lat = ds_grid.Latitudes

ds_grid = xr.open_dataset('LongitudeLatitudeGrid-s6250-Antarctic.hdf')

lon_s = ds_grid.Longitudes
lat_s = ds_grid.Latitudes

# +
sic_avg_n_max = xr.DataArray(np.zeros([1792,1216],dtype=np.float64),coords={'lon': (["x","y"], lon),
                          'lat': (["x","y"], lat)},
                  dims=['x','y'], name = 'sic')

sic_avg_n_min = xr.DataArray(np.zeros([1792,1216],dtype=np.float64),coords={'lon': (["x","y"], lon),
                          'lat': (["x","y"], lat)},
                  dims=['x','y'], name = 'sic')

sic_avg_s_max = xr.DataArray(np.zeros([1328,1264],dtype=np.float64),coords={'lon': (["x","y"], lon_s),
                          'lat': (["x","y"], lat_s)},
                  dims=['x','y'], name = 'sic')

sic_avg_s_min = xr.DataArray(np.zeros([1328,1264],dtype=np.float64),coords={'lon': (["x","y"], lon_s),
                          'lat': (["x","y"], lat_s)},
                  dims=['x','y'], name = 'sic')

# +
# Averaging SIC over days of maximum/minimum sea ice extent in Arctic/Antarctica
###################################################################################
files = sorted(glob.glob(data_path_base+'/data/sea_ice/arctic_maximum/*-v5.4.nc'))

for f in files:
    print(f)
    ds = xr.open_dataset(f)

    sic = xr.DataArray(ds.z,coords={'lon': (["x","y"], lon),
                          'lat': (["x","y"], lat)},
                  dims=['x','y'], name = 'sic')
    sic_avg_n_max += sic
    
sic_avg_n_max = sic_avg_n_max/len(files)
###################################################################################
files = sorted(glob.glob(data_path_base+'/data/sea_ice/arctic_minimum/*-v5.4.nc'))

for f in files:
    print(f)
    ds = xr.open_dataset(f)

    sic = xr.DataArray(ds.z,coords={'lon': (["x","y"], lon),
                          'lat': (["x","y"], lat)},
                  dims=['x','y'], name = 'sic')
    sic_avg_n_min += sic
    
sic_avg_n_min = sic_avg_n_min/len(files)
###################################################################################
files = sorted(glob.glob(data_path_base+'/data/sea_ice/antarctica_maximum/*-v5.4.nc'))

for f in files:
    print(f)
    ds = xr.open_dataset(f)

    sic = xr.DataArray(ds.z,coords={'lon': (["x","y"], lon_s),
                          'lat': (["x","y"], lat_s)},
                  dims=['x','y'], name = 'sic')
    sic_avg_s_max += sic
    
sic_avg_s_max = sic_avg_s_max/len(files)
###################################################################################
files = sorted(glob.glob(data_path_base+'/data/sea_ice/antarctica_minimum/*-v5.4.nc'))

for f in files:
    print(f)
    ds = xr.open_dataset(f)

    sic = xr.DataArray(ds.z,coords={'lon': (["x","y"], lon_s),
                          'lat': (["x","y"], lat_s)},
                  dims=['x','y'], name = 'sic')
    sic_avg_s_min += sic
    
sic_avg_s_min = sic_avg_s_min/len(files)
# -

ice_n_djf = xr.where(sic_avg_n_max >= 15.0, 100.0, sic_avg_n_max)
ice_n_jja = xr.where(sic_avg_n_min >= 15.0, 100.0, sic_avg_n_min)
ice_s_djf = xr.where(sic_avg_s_min >= 15.0, 100.0, sic_avg_s_min)
ice_s_jja = xr.where(sic_avg_s_max >= 15.0, 100.0, sic_avg_s_max)


# +
lon = ice_s_djf.lon
lat = ice_s_djf.lat

ice_s_xr = xr.DataArray(ice_s_djf,coords={'lon': (["x","y"], lon),
                          'lat': (["x","y"], lat)},
                  dims=['x','y'], name = 'sic_djf')
ds_ice_s_djf = ice_s_xr.to_dataset(dim=None, name='sic_djf')
dr_ice_s_djf = ds_ice_s_djf.sic_djf


ice_s_xr = xr.DataArray(ice_s_jja,coords={'lon': (["x","y"], lon),
                          'lat': (["x","y"], lat)},
                  dims=['x','y'], name = 'sic_jja')
ds_ice_s_jja = ice_s_xr.to_dataset(dim=None, name='sic_jja')
dr_ice_s_jja = ds_ice_s_jja.sic_jja



#########################
lon = ice_n_djf.lon
lat = ice_n_djf.lat

ice_n_xr = xr.DataArray(ice_n_djf,coords={'lon': (["x","y"], lon),
                          'lat': (["x","y"], lat)},
                  dims=['x','y'], name = 'sic_djf')
ds_ice_n_djf = ice_n_xr.to_dataset(dim=None, name='sic_djf')
dr_ice_n_djf = ds_ice_n_djf.sic_djf


ice_n_xr = xr.DataArray(ice_n_jja,coords={'lon': (["x","y"], lon),
                          'lat': (["x","y"], lat)},
                  dims=['x','y'], name = 'sic_jja')
ds_ice_n_jja = ice_n_xr.to_dataset(dim=None, name='sic_jja')
dr_ice_n_jja = ds_ice_n_jja.sic_jja
# -

ds_out = xr.Dataset({'lat': (['lat'], np.arange(-89.875, 90., 0.25)),
                     'lon': (['lon'], np.arange(0.125, 360., 0.25)),
                    }
                   )
ds_out

# +
regridder = xe.Regridder(ds_ice_s_djf, ds_out, 'bilinear')
dr_out_ice_s_djf = regridder(dr_ice_s_djf)
dr_out_ice_s_djf.to_netcdf('output_files/sea_ice/ice_s_djf_regrid.nc')



ds_out = xr.Dataset({'lat': (['lat'], np.arange(-89.875, 90., 0.25)),
                     'lon': (['lon'], np.arange(0.125, 360., 0.25)),
                    }
                   )

regridder = xe.Regridder(ds_ice_s_jja, ds_out, 'bilinear')
dr_out_ice_s_jja = regridder(dr_ice_s_jja)
dr_out_ice_s_jja.to_netcdf('output_files/sea_ice/ice_s_jja_regrid.nc')

########################################
ds_out = xr.Dataset({'lat': (['lat'], np.arange(-89.875, 90., 0.25)),
                     'lon': (['lon'], np.arange(0.125, 360., 0.25)),
                    }
                   )

regridder = xe.Regridder(ds_ice_n_jja, ds_out, 'bilinear')
dr_out_ice_n_jja = regridder(dr_ice_n_jja)
dr_out_ice_n_jja.to_netcdf('output_files/sea_ice/ice_n_jja_regrid.nc')


ds_out = xr.Dataset({'lat': (['lat'], np.arange(-89.875, 90., 0.25)),
                     'lon': (['lon'], np.arange(0.125, 360., 0.25)),
                    }
                   )

regridder = xe.Regridder(ds_ice_n_djf, ds_out, 'bilinear')
dr_out_ice_n_djf = regridder(dr_ice_n_djf)
dr_out_ice_n_djf.to_netcdf('output_files/sea_ice/ice_n_djf_regrid.nc')

sys.exit()
# -




