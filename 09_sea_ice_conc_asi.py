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




# +
ds_grid = xr.open_dataset('LongitudeLatitudeGrid-n6250-Arctic.hdf')
ds_grid = xr.open_dataset('LongitudeLatitudeGrid-s6250-Antarctic.hdf')

lon = ds_grid.Longitudes
lat = ds_grid.Latitudes

# +
files = np.array(sorted(glob.glob(data_path_base + '/data/sea_ice/*/asi*s6250-*-v5.4.nc')))
month = np.zeros(len(files), dtype = 'int')

i = 0
for f in files:
    month[i] = int(f[-12:-10])
    i += 1

print(np.shape(month),month)
# -

index_jja = np.array(np.where((month == 6) | (month == 7) | (month == 8)), dtype = 'int').flatten()
index_djf = np.array(np.where((month == 12) | (month == 1) | (month == 2)), dtype = 'int').flatten()

files_jja = files[index_jja]
files_djf = files[index_djf]

# +
sic_jja = np.zeros([len(lon), len(lon[0])],dtype=np.float64)
sic_n = np.zeros([len(lon), len(lon[0])],dtype=np.float64)

for f in files_jja:
    # read file
    print(f)
    ds = xr.open_dataset(f)
    sic = xr.DataArray(ds.z,coords={'lon': (["x","y"], lon),
                          'lat': (["x","y"], lat)},
                  dims=['x','y'], name = 'sic')
    
    x = 600
    y = 500
    #print(sic[x, y])
    n = xr.where(sic >= 0.0, 1, sic)
    n = n.fillna(0)
    #print(n[x, y])
    #print(sic_jja[x, y])

    sic_jja = (sic_jja * sic_n + sic.fillna(0)) / (sic_n + n)
    #print(sic_jja[x, y])
    sic_n += n
    #print(sic_n[x, y])
    
    #print('######################')

sic_jja_xr = xr.DataArray(data = sic_jja, coords={'lon': (["x","y"], lon),
                          'lat': (["x","y"], lat)},
                  dims=['x','y'], name = 'sic')
sic_jja_xr.to_netcdf('Figures/sic_S_JJA.nc')
    


# +
sic_djf = np.zeros([len(lon), len(lon[0])],dtype=np.float64)
sic_n = np.zeros([len(lon), len(lon[0])],dtype=np.float64)

for f in files_djf:
    # read file
    print(f)
    ds = xr.open_dataset(f)
    sic = xr.DataArray(ds.z,coords={'lon': (["x","y"], lon),
                          'lat': (["x","y"], lat)},
                  dims=['x','y'], name = 'sic')
    
    x = 600
    y = 500
    #print(sic[x, y])
    n = xr.where(sic >= 0.0, 1, sic)
    n = n.fillna(0)
    #print(n[x, y])
    #print(sic_jja[x, y])

    sic_djf = (sic_djf * sic_n + sic.fillna(0)) / (sic_n + n)
    #print(sic_jja[x, y])
    sic_n += n
    #print(sic_n[x, y])
    
    #print('######################')

sic_djf_xr = xr.DataArray(data = sic_djf, coords={'lon': (["x","y"], lon),
                          'lat': (["x","y"], lat)},
                  dims=['x','y'], name = 'sic')
sic_djf_xr.to_netcdf('Figures/sic_S_DJF.nc')


sys.exit()
# -

ds = xr.open_dataset(data_path_base+'/data/sea_ice/2006/asi-n6250-20060626-v5.4.nc')
sic_xr = xr.DataArray(ds.z,coords={'lon': (["x","y"], lon),
                          'lat': (["x","y"], lat)},
                  dims=['x','y'], name = 'sic')

print(np.nanmax(sic_jja), np.nanmin(sic_jja), np.nanmax(sic_n), np.nanmin(sic_n))

print(np.shape(sic_n), np.shape(sic_jja))

x = 600
y = 500
print(lon[x,y], lat[x,y])

index = np.where((lat > 80.) & (lat < 80.0005))

lat[index]

n = xr.where(sic >= 0.0, 1, sic)
n = n.fillna(0)
n

sic_xr

sic_xr[index]

# +
plt.figure(figsize=(10,12))
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0.0, globe=None))
#ax.set_global()
sic_jja.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', add_colorbar=False)

contour = sic_jja.plot.contour(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', levels = [50.], add_colorbar=False, colors = 'r')

ax.coastlines()
#ax.set_ylim([0,90]);
# for polar stereographic projection circular plot
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)
# -

print(contour)


# +

def get_contour_verts(cn):
    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contours.append(paths)

    return contours


# -

test = get_contour_verts(contour)

np.shape(test)

# +
plt.figure(figsize=(10,12))
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0.0, globe=None))
#ax.set_global()
sic_jja.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', add_colorbar=False)

p = plt.Polygon(test, fill=False, color='w')
ax.add_artist(p)

ax.coastlines()
#ax.set_ylim([0,90]);
# for polar stereographic projection circular plot
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)
# -









from scipy.io import loadmat
annots = loadmat('monat_200808.mat')
print(annots)

con_list = [[element for element in upperElement] for upperElement in annots['iconc']]

# zip provides us with both the x and y in a tuple.
newData = list(zip(con_list[0], con_list[1]))
columns = ['x', 'y']
df = pd.DataFrame(newData, columns=columns)

print(np.nanmin(df))



file = 'ESACCI-SEAICE-L4-SICONC-AMSR_25.0kmEASE2-NH-20080807-fv2.0.nc'
ds = xr.open_dataset(file)
ds

data = ds.ice_conc[0,:,:]
lon = ds.lon
lat = ds.lat

print(np.shape(data), np.shape(lon), np.shape(lat), np.nanmax(data), np.nanmin(data))

# +
custom_cmap = cm.get_cmap('Spectral_r', 40)

fig, axs = plt.subplots(nrows=1,ncols=1,figsize =[20,20], subplot_kw={'projection': ccrs.PlateCarree()})


# plot sea ice concentration
cs = axs.pcolormesh(lon, lat, data,cmap=custom_cmap, vmin = 0, vmax = 5)
cs = axs.contour(lon, lat, data,levels = 15.)

# plot parameters
axs.set_extent([-180, 180, 40, 90], ccrs.PlateCarree())
axs.coastlines(resolution='110m', linewidth = 3.0)
#cbar = fig.colorbar(cs, ax=axs, orientation='vertical',pad=0.05, shrink = 0.8)
#cbar.set_label('Sea ice concentration (%)',fontsize=16)
# -



ice_mask_xr = xr.DataArray(data,coords={'lon': (["xc","yc"], lon),
                          'lat': (["xc","yc"], lat)},
                  dims=['xc','yc'], name = 'mask')

# +
plt.figure(figsize=(10,12))
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0.0, globe=None))
#ax.set_global()
ice_mask_xr.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', add_colorbar=False)

ice_mask_xr.plot.contour(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', levels = 15, add_colorbar=False, colors = 'teal')

ax.coastlines()
#ax.set_ylim([0,90]);
# for polar stereographic projection circular plot
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)
# -











file = 'seaice_conc_monthly_nh_197811_202012_v04r00.nc'
ds = xr.open_dataset(file)
ds

data = ds.cdr_seaice_conc_monthly[357,:,:]
lon = ds.longitude
lat = ds.latitude



print(np.shape(data), np.shape(lon), np.shape(lat))

print(np.nanmax(data), np.nanmin(data))

print(np.shape(data), np.shape(lon), np.shape(lat))

data = data.where(data < 1.0)
data

ice_mask_xr = xr.DataArray(data,coords={'lon': (["xgrid","ygrid"], lon),
                          'lat': (["xgrid","ygrid"], lat)},
                  dims=['xgrid','ygrid'], name = 'mask')


ice_mask_xr

# +
custom_cmap = cm.get_cmap('Spectral_r', 40)

fig, axs = plt.subplots(nrows=1,ncols=1,figsize =[20,20], subplot_kw={'projection': ccrs.PlateCarree()})


# plot sea ice concentration
cs = axs.pcolormesh(lon, lat, data,cmap=custom_cmap, vmin = 0, vmax = 5)

# plot parameters
axs.set_extent([-180, 180, 40, 90], ccrs.PlateCarree())
axs.coastlines(resolution='110m', linewidth = 3.0)
#cbar = fig.colorbar(cs, ax=axs, orientation='vertical',pad=0.05, shrink = 0.8)
#cbar.set_label('Sea ice concentration (%)',fontsize=16)

# +
#Flag Name Value Northern Hemisphere pole hole (the region around the pole 
#not imaged by the sensor) 251 Lakes 252 Coast/Land adjacent to ocean 253 Land 254 Missing/Fill 255

# +
custom_cmap = cm.get_cmap('Spectral_r', 40)

fig, axs = plt.subplots(nrows=1,ncols=1,figsize =[20,20], subplot_kw={'projection': ccrs.PlateCarree()})


# plot sea ice concentration
cs = axs.pcolormesh(lon, lat, data,cmap=custom_cmap, vmin = 0, vmax = 5)
cs = axs.contour(lon, lat, data,levels = 0.15)

# plot parameters
axs.set_extent([-180, 180, 40, 90], ccrs.PlateCarree())
axs.coastlines(resolution='110m', linewidth = 3.0)
plt.savefig('test.png')
#cbar = fig.colorbar(cs, ax=axs, orientation='vertical',pad=0.05, shrink = 0.8)
#cbar.set_label('Sea ice concentration (%)',fontsize=16)

sys.exit()
# -
data

# +
plt.figure(figsize=(10,12))
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0.0, globe=None))
#ax.set_global()
ice_mask_xr.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', add_colorbar=False)

#ice_mask_xr.plot.contour(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', levels = 0.15, add_colorbar=False, colors = 'teal')

ax.coastlines()
#ax.set_ylim([0,90]);
# for polar stereographic projection circular plot
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)
# -



print(np.nanmax(ice_mask_xr))
print(np.nanmin(ice_mask_xr))

sic = ma.masked_where(ice_mask_xr == 0, data)

ice_mask_xr.values[np.where(ice_mask_xr.lat.values >=85.)] = 1.








ds = xr.open_dataset('masie_all_r00_v01_2008323_4km.nc')
ds

data = ds.sea_ice_extent[0,:,:]
print(np.shape(data))

print(np.nanmax(data), np.nanmin(data))

grid = xr.open_dataset('masie_lat_lon_4km.nc')

np.shape(grid.longitude)



lon = grid.longitude.values
lat = grid.latitude.values
data = data.values

print(lon[0,0], lat[0,0])
print(lon[6143,6143], lat[6143,6143])
print(lon[0,6143], lat[0,6143])
print(lon[6143,0], lat[6143,0])

print(np.nanmax(data))

# +
custom_cmap = cm.get_cmap('Spectral_r', 40)

fig, axs = plt.subplots(nrows=1,ncols=1,figsize =[8,8], subplot_kw={'projection': ccrs.PlateCarree()})


# plot sea ice concentration
cs = axs.pcolormesh(lon, lat, data,cmap=custom_cmap, vmin = 0, vmax = 5)

# plot parameters
axs.set_extent([-180, 180, 40, 90], ccrs.PlateCarree())
axs.coastlines(resolution='110m', linewidth = 3.0)
cbar = fig.colorbar(cs, ax=axs, orientation='vertical',pad=0.05, shrink = 0.8)
cbar.set_label('Sea ice concentration (%)',fontsize=16)
plt.savefig('masie.png')

# -

sys.exit()







# +
custom_cmap = cm.get_cmap('Spectral_r', 40)

fig, axs = plt.subplots(nrows=1,ncols=1,figsize =[8,8], subplot_kw={'projection': ccrs.PlateCarree()})


# plot sea ice concentration
cs = axs.pcolormesh(grid.longitude, grid.latitude, data,cmap=custom_cmap, vmin = 0, vmax = 5)
cs = axs.contour(grid.longitude, grid.latitude, data,levels = 1., colors = 'k')

plt.savefig('masie.png')

sys.exit()
# -









def wrap_lon180(lon):
    lon = np.atleast_1d(lon).copy()
    angles = np.logical_or((lon < -180), (180 < lon))
    lon[angles] = wrap_lon360(lon[angles] + 180) - 180
    return lon
def wrap_lon360(lon):
    lon = np.atleast_1d(lon).copy()
    positive = lon > 0
    lon = lon % 360
    lon[np.logical_and(lon == 0, positive)] = 360
    return lon



import iris
import iris.analysis.cartography
import iris.quickplot as qplt
#from oceans import wrap_lon180

cube = iris.load_cube('ice_mask_180.nc','mask')

cube

cube.coord("latitude").var_name = "latitude"
cube.coord("longitude").var_name = "longitude"

new_cube, extent = iris.analysis.cartography.project(cube, ccrs.NorthPolarStereo())


cube.coord(axis='longitude').points = wrap_lon180(cube.coord(axis='longitude').points)

# +
ice_cube = iris.load_cube('ice_mask_180.nc')


fig = plt.figure()
ax = plt.subplot(projection=ccrs.NorthPolarStereo())
contour = qplt.contour(ice_cube)
# Draw coastlines
ax.coastlines()
plt.savefig('mask.png')
# Add a contour, and put the result in a variable called contour.
#contour = qplt.contour(ice_cube)

sys.exit()

# +
# Transform cube to target projection
new_cube, extent = iris.analysis.cartography.project(cube, ccrs.NorthPolarStereo())

fig = plt.figure()
# Set up axes and title
ax = plt.subplot(projection=projection)
# Set limits
ax.set_global()
# plot with Iris quickplot pcolormesh
qplt.pcolormesh(cube)
# Draw coastlines
ax.coastlines()

plt.show()

# -

sys.exit()







ds = xr.open_dataset('asi-n6250-20060831-v5.hdf')
data = ds['ASI Ice Concentration']
data.plot.contour()

ds_mask = xr.open_dataset('landmask_Arc_6.25km.hdf')
mask = ds_mask['landmask Arc 6.25 km']
mask.plot()

sic = ma.masked_where(mask == 0, data)

print(np.nanmean(data))

print(np.nanmean(sic))

ds_grid = xr.open_dataset('LongitudeLatitudeGrid-n6250-Arctic.hdf')
lon = ds_grid.Longitudes
lat = ds_grid.Latitudes


# +
class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
 
    def getX(self):
        return self.x
    def getY(self):
        return self.y

def getGrayDiff(img,currentPoint,tmpPoint):
    return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))
 
def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [ Point(0, -1),  Point(1, 0),Point(0, 1), Point(-1, 0)]
    return connects
 
def regionGrow(img,seeds,thresh,p = 1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)
 
        seedMark[currentPoint.x,currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            #grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
            #if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
            if img[Point(tmpX,tmpY).x,Point(tmpX,tmpY).y] > thresh and seedMark[tmpX,tmpY] == 0: 
                seedMark[tmpX,tmpY] = label
                seedList.append(Point(tmpX,tmpY))
    return seedMark


# -

img = data
seeds = [Point(972,498),Point(973,499),Point(974,500)]
binaryImg = regionGrow(img,seeds,15)

ice_mask_xr = xr.DataArray(binaryImg,coords={'lon': (["x","y"], lon),
                          'lat': (["x","y"], lat)},
                  dims=['x','y'], name = 'mask')
ice_mask_xr.to_netcdf('ice_mask.nc')

mask = xr.open_dataset('ice_mask.nc')
mask

mask = mask.rename({'lon': 'longitude','lat': 'latitude'})

print(np.nanmax(mask.lon), np.nanmin(mask.lon))

print(lon[0,0],lon[0,1],lon[0,2],lon[1,0],lon[1,1], lon[1,2])

print(lon[0,0]-lon[0,1],lon[0,1]-lon[0,2] )
print(lon[1,0]-lon[1,1],lon[1,1]-lon[1,2] )
print(lon[1791, 0], lon[0,1215], lon[1791,1215])

print(lon[0,0], lon[1791, 0], lon[0,1215], lon[1791,1215])
print(lat[0,0], lat[1791, 0], lat[0,1215], lat[1791,1215])

print(np.shape(lat[lon == 360]))

print(np.max(lat[lon == 360]))

lon.shape

mask

print(np.nanmax(mask.lon), np.nanmin(mask.lon))

mask.to_netcdf('ice_mask_180.nc')

mask['mask'].plot.contour()

test = mask
test.mask.values[np.where(mask.lat.values >=85.)] = 1.

test.mask.plot.contour()

# +
plt.figure(figsize=(14,6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
mask.mask.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', add_colorbar=False)
mask.mask.plot.contour(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', levels = 1, add_colorbar=False, colors = 'teal')

ax.coastlines()
ax.set_ylim([0,90]);
# -

import iris
import iris.analysis.cartography
import iris.quickplot as qplt
from oceans import wrap_lon180

#filepath = iris.sample_data_path('ice_mask.nc')
cube = iris.load_cube('ice_mask_180.nc')

cube.coord(axis='X').points = wrap_lon180(cube.coord(axis='X').points)

# +
ice_cube = iris.load_cube('ice_mask_180.nc')


fig = plt.figure()
ax = plt.subplot(projection=ccrs.NorthPolarStereo())
contour = qplt.contour(ice_cube)
# Draw coastlines
ax.coastlines()
# Add a contour, and put the result in a variable called contour.
#contour = qplt.contour(ice_cube)
# -

projection = ccrs.NorthPolarStereo()

cube

lat.values

for coord in cube.coords():
    print(coord.name())

cube.shape

cube.data

cube.coords('lon')

cube.add_dim_coord(cube.coords('lon'),0)

cube.coord('print(np.shape(lat[lon == 360]))').points

# +
# Transform cube to target projection
new_cube, extent = iris.analysis.cartography.project(cube, ccrs.NorthPolarStereo())

fig = plt.figure()
# Set up axes and title
ax = plt.subplot(projection=projection)
# Set limits
ax.set_global()
# plot with Iris quickplot pcolormesh
qplt.pcolormesh(cube)
# Draw coastlines
ax.coastlines()

plt.show()

# -









# +
plt.figure(figsize=(10,6))
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0.0, globe=None))
#ax.set_global()
test.mask.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', add_colorbar=False)

test.mask.plot.contour(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', levels = 1., add_colorbar=False, colors = 'teal')

ax.coastlines()
#ax.set_ylim([0,90]);
# for polar stereographic projection circular plot
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

# +
from cartopy.util import add_cyclic_point

dd, ll = add_cyclic_point(test.mask.values, test.mask.lon)
test_cyclic = xr.DataArray(dd, coords={'lon': (["x","y"], ll),
                          'lat': (["x","y"], lat)},
                  dims=['x','y'], name = 'mask')
# -





print(np.nanmax(mask['mask']), np.nanmin(mask['mask']))

# +
plt.figure(figsize=(14,6))
ax = plt.axes(projection=ccrs.NorthPolar())
ax.set_global()
mask['mask'].plot.contour(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', levels = 1., add_colorbar=False, colors = 'teal')
ax.coastlines()

ax.set_ylim([0,90]);
# -

sst = ds.sst.sel(time='2000-01-01', method='nearest')
fig = plt.figure(figsize=(9,6))
ax = plt.axes(projection=ccrs.Robinson())
ax.coastlines()
ax.gridlines()
sst.plot(ax=ax, transform=ccrs.PlateCarree(),
         vmin=2, vmax=30, cbar_kwargs={'shrink': 0.4})

index = np.where((np.abs(lon.values-180.) < 0.5) & (np.abs(lat.values-80.) < 0.5))

index

print(lon[972,498], lat[972,498])

seedList = []

for seed in seeds:
    seedList.append(seed)

seedList

currentPoint = seedList.pop(0)

print(currentPoint.x)
print(currentPoint.y)

label = 1

seedMark[currentPoint.x,currentPoint.y] = label

p = 1
connects = selectConnects(p)

print(connects)

for i in range(8):
    tmpX = currentPoint.x + connects[i].x
    tmpY = currentPoint.y + connects[i].y
    print(tmpX, tmpY)
    print(img[Point(tmpX,tmpY).x, Point(tmpX,tmpY).y].values)

seedMark[tmpX,tmpY] = label

seedList.append(Point(tmpX,tmpY))

grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))

grayDiff

int(img[Point(tmpX,tmpY).x,Point(tmpX,tmpY).y])

print(np.nanmax(lon))



for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
            #if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
            if int(img[Point(tmpX,tmpY).x,Point(tmpX,tmpY).y]) > thresh and seedMark[tmpX,tmpY] == 0: 
                seedMark[tmpX,tmpY] = label
                seedList.append(Point(tmpX,tmpY))



print(np.max(binaryImg))





print(np.nanmax(binaryImg), np.nanmin(binaryImg), np.nanmean(binaryImg))

# +
fig, axs = plt.subplots(nrows=1,ncols=1,figsize =[8,8], subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)})

# for polar stereographic projection circular plot
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
axs.set_boundary(circle, transform=axs.transAxes)

# plot sea ice concentration
cs = axs.pcolormesh(lon, lat, mask.mask,cmap=custom_cmap, vmin = 0, vmax = 1, transform = ccrs.PlateCarree())

# plot parameters
axs.set_extent([-180, 180, 40, 90], ccrs.PlateCarree())
axs.coastlines(resolution='110m', linewidth = 3.0)
cbar = fig.colorbar(cs, ax=axs, orientation='vertical',pad=0.05, shrink = 0.8)
cbar.set_label('Sea ice concentration (%)',fontsize=16)
# -

test = binaryImg

test[np.where(lat.values >=85.)] = 1.

# +
a=1

order = np.argsort(lon.values, axis=a)


x = np.take_along_axis(lon.values, order, axis=a)
y = np.take_along_axis(lat.values, order, axis=a)
z = np.take_along_axis(test, order, axis=a)

# +
custom_cmap = cm.get_cmap('Spectral_r', 40)

fig, axs = plt.subplots(nrows=1,ncols=1,figsize =[8,8], subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)})

# for polar stereographic projection circular plot
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
axs.set_boundary(circle, transform=axs.transAxes)

# plot sea ice concentration
cs = axs.pcolormesh(lon, lat, test.mask,cmap=custom_cmap, vmin = 0, vmax = 1, transform = ccrs.PlateCarree())
#cs2 = axs.tricontour(lon, lat, test.mask,levels=1.0, colors = 'teal', linewidth = 1.5, transform = ccrs.PlateCarree())


# plot parameters
axs.set_extent([-180, 180, 40, 90], ccrs.PlateCarree())
axs.coastlines(resolution='110m', linewidth = 3.0)
cbar = fig.colorbar(cs, ax=axs, orientation='vertical',pad=0.05, shrink = 0.8)
cbar.set_label('Sea ice concentration (%)',fontsize=16)
# +
import matplotlib.tri as tri
from scipy.interpolate import griddata
zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

ax1.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
cntr1 = ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")

# -




# +
import numpy as np
import cv2
 
class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
 
    def getX(self):
        return self.x
    def getY(self):
        return self.y

def getGrayDiff(img,currentPoint,tmpPoint):
    return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))
 
def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [ Point(0, -1),  Point(1, 0),Point(0, 1), Point(-1, 0)]
    return connects
 
def regionGrow(img,seeds,thresh,p = 1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)
 
        seedMark[currentPoint.x,currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
            #if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
            if int(img[Point(tmpX,tmpY).x,Point(tmpX,tmpY).y]) > thresh and seedMark[tmpX,tmpY] == 0: 
                seedMark[tmpX,tmpY] = label
                seedList.append(Point(tmpX,tmpY))
    return seedMark
 
img = data
seeds = [Point(721,751),Point(722,751),Point(723,751)]
binaryImg = regionGrow(img,seeds,15)
print(np.shape(binaryImg))
#cv2.imshow(' ',binaryImg)
#cv2.waitKey(0)

# -

img = data
print(img.shape)

height, weight = img.shape
seedMark = np.zeros(img.shape)
seedList = []
seeds = [Point(721,751),Point(722,751),Point(723,751)]


for seed in seeds:
    print(seed)

print(height, weight)

print(seedList)





# +
custom_cmap = cm.get_cmap('Spectral_r', 40)

fig, axs = plt.subplots(nrows=1,ncols=1,figsize =[8,8], subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)})

# for polar stereographic projection circular plot
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
axs.set_boundary(circle, transform=axs.transAxes)

# plot sea ice concentration
#cs = axs.pcolormesh(lon, lat, mask.mask,cmap=custom_cmap, vmin = 0, vmax = 1, transform = ccrs.PlateCarree())




# overlay sea ice edge (at 15% sea ice concentration)
#cs2 = axs.contour(lon.flatten(),lat.flatten(),mask.mask, levels = [1.],colors='teal',linewidths=2.0, transform = ccrs.PlateCarree())

cs = axs.contourf(lon,lat,mask.mask, cmap = custom_cmap, vmin = 0, vmax = 100, transform = ccrs.PlateCarree())


# plot parameters
axs.set_extent([-180, 180, 40, 90], ccrs.PlateCarree())
axs.coastlines(resolution='110m', linewidth = 3.0)
cbar = fig.colorbar(cs, ax=axs, orientation='vertical',pad=0.05, shrink = 0.8)
cbar.set_label('Sea ice concentration (%)',fontsize=16)

#plt.savefig('Figures/region_grow.png',bbox_inches='tight')

# -

lon.shape

sys.exit()

print(np.nanmax(lat))





file_sic = 'asi-n6250-20060831-v5.hdf'
hdf_sic = SD(file_sic, SDC.READ)

# +
datasets_dic = hdf_sic.datasets()

for idx,sds in enumerate(datasets_dic.keys()):
    print(idx,sds )       
# -

sds_obj = hdf_sic.select('ASI Ice Concentration') # select sds
sic = sds_obj.get() # get sds data

print(np.shape(sic), np.nanmax(sic), np.nanmin(sic))
print(np.max(sic), np.min(sic))

file_grid = 'LongitudeLatitudeGrid-n6250-Arctic.hdf'
hdf_grid = SD(file_grid, SDC.READ)

# +
datasets_dic = hdf_grid.datasets()

for idx,sds in enumerate(datasets_dic.keys()):
    print(idx,sds )       
# -

sds_obj = hdf_grid.select('Longitudes') # select sds
lon = sds_obj.get() # get sds data
sds_obj = hdf_grid.select('Latitudes') # select sds
lat = sds_obj.get() # get sds data

print(np.shape(lon), np.nanmax(lon), np.nanmin(lon))
print(np.shape(lat), np.nanmax(lat), np.nanmin(lat))

print(lon[0,0], lon[0,1:3], lon[0:3,0])

print(lat[0,0], lat[0,1:3], lat[0:3,0])

plt.scatter(lon.flatten(),lat.flatten())

# +
plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution='110m')
ax.gridlines()

ax.scatter(lon.flatten(), lat.flatten())

# +
fig, ax = plt.subplots(dpi=150)

x = lon
y = lat
z = sic

plt.pcolormesh(x, y, z,cmap='viridis',snap=True )#, transform = ccrs.PlateCarree())
plt.colorbar()

# +
custom_cmap = cm.get_cmap('Spectral_r', 40)

fig, axs = plt.subplots(nrows=1,ncols=2,figsize =[16,8], subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)})

# for polar stereographic projection circular plot
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
axs[0].set_boundary(circle, transform=axs[0].transAxes)
axs[1].set_boundary(circle, transform=axs[1].transAxes)

# plot sea ice concentration
cs = axs[0].pcolormesh(lon, lat, data,cmap=custom_cmap, vmin = 0, vmax = 100, transform = ccrs.PlateCarree())

cs2 = axs[1].pcolormesh(lon, lat, data,cmap=custom_cmap, vmin = 0, vmax = 100, transform = ccrs.PlateCarree())



# overlay sea ice edge (at 15% sea ice concentration)
cs2 = axs[1].contour(lon,lat,data, levels = [15.],colors='teal',linewidths=2.0, transform = ccrs.PlateCarree())
#cs2 = axs.contour(lon,lat,binaryImg, levels = [1.],colors='teal',linewidths=2.0, transform = ccrs.PlateCarree())

#cs = axs.contourf(lon,lat,sic, cmap = custom_cmap, vmin = 0, vmax = 100, transform = ccrs.PlateCarree())


# plot parameters
axs[0].set_extent([-180, 180, 40, 90], ccrs.PlateCarree())
axs[0].coastlines(resolution='110m', linewidth = 3.0)
cbar = fig.colorbar(cs, ax=axs[0], orientation='vertical',pad=0.05, shrink = 0.8)
cbar.set_label('Sea ice concentration (%)',fontsize=16)\

# plot parameters
axs[1].set_extent([-180, 180, 40, 90], ccrs.PlateCarree())
axs[1].coastlines(resolution='110m', linewidth = 3.0)
cbar = fig.colorbar(cs, ax=axs[1], orientation='vertical',pad=0.05, shrink = 0.8)
cbar.set_label('Sea ice concentration (%)',fontsize=16)



# +
a=1
b=1

order = np.argsort(lon.values, axis=a)


x = np.take_along_axis(lon.values, order, axis=a)
#x = np.take_along_axis(x, order2, axis=b)
y = np.take_along_axis(lat.values, order, axis=a)
#y = np.take_along_axis(y, order2, axis=b)

z = np.take_along_axis(binaryImg, order, axis=a)

#z = np.take_along_axis(z, order2, axis=b)


custom_cmap = cm.get_cmap('Spectral_r', 40)

fig, axs = plt.subplots(nrows=1,ncols=1,figsize =[8,8], subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)})

# for polar stereographic projection circular plot
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
axs.set_boundary(circle, transform=axs.transAxes)

# plot sea ice concentration
cs = axs.pcolormesh(lon, lat, data,cmap=custom_cmap, vmin = 0, vmax = 100, transform = ccrs.PlateCarree())




# overlay sea ice edge (at 15% sea ice concentration)
#cs2 = axs.contour(lon,lat,data, levels = [15.],colors='teal',linewidths=2.0, transform = ccrs.PlateCarree())
cs2 = axs.contour(x,y,z, levels = [1.],colors='teal',linewidths=2.0, transform = ccrs.PlateCarree())

#cs = axs.contourf(lon,lat,sic, cmap = custom_cmap, vmin = 0, vmax = 100, transform = ccrs.PlateCarree())


# plot parameters
axs.set_extent([-180, 180, 40, 90], ccrs.PlateCarree())
axs.coastlines(resolution='110m', linewidth = 3.0)
cbar = fig.colorbar(cs, ax=axs, orientation='vertical',pad=0.05, shrink = 0.8)
cbar.set_label('Sea ice concentration (%)',fontsize=16)
# -



lon





print(np.shape(sic))

print(np.shape(lon), np.shape(lat))

# +
a=1
b=1

order = np.argsort(lon, axis=a)
order2 = np.argsort(lat, axis=a)

fig, ax = plt.subplots(dpi=150)

x = np.take_along_axis(lon, order, axis=a)
#x = np.take_along_axis(x, order2, axis=b)
y = np.take_along_axis(lat, order, axis=a)
#y = np.take_along_axis(y, order2, axis=b)

z = np.take_along_axis(sic, order, axis=a)

#z = np.take_along_axis(z, order2, axis=b)


plt.pcolormesh(x, y, z,cmap='viridis',snap=True )#, transform = ccrs.PlateCarree())
plt.colorbar()
# -



























