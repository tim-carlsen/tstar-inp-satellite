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


##########################################    
def calc_cth_at_tstar(season, latitude, longitude, mode):
    
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
    
    #Calculate T* again from data
    #index = np.array(np.where(freq_lo_total-freq_ltmpc_total < 0)).flatten();
    index = np.array(np.where((freq_lo_total-freq_ltmpc_total < 0) & (hist_ltmpc > 10))).flatten();    ####### NEW !!! ###########

    
    # Cloud top height histograms
    bins_cth = np.arange(0,5000,200)
    
    if mode == 'all':
        hist_lo_cth, edges_lo = np.histogram(data_lo.CTH, bins_cth);
        hist_ltmpc_cth, edges_lo = np.histogram(data_ltmpc.CTH, bins_cth);
        hist_mpc_cth, edges_lo = np.histogram(data_mpc.CTH, bins_cth);
        hist_io_cth, edges_lo = np.histogram(data_io.CTH, bins_cth);
    else:
        hist_lo_cth, edges_lo = np.histogram(data_lo.CTH[np.where(np.abs(data_lo.CTT - np.max(bins[index])) < 1.)], bins_cth);
        hist_ltmpc_cth, edges_lo = np.histogram(data_ltmpc.CTH[np.where(np.abs(data_ltmpc.CTT - np.max(bins[index])) < 1.)], bins_cth);
        hist_mpc_cth, edges_lo = np.histogram(data_mpc.CTH[np.where(np.abs(data_mpc.CTT - np.max(bins[index])) < 1.)], bins_cth);
        hist_io_cth, edges_lo = np.histogram(data_io.CTH[np.where(np.abs(data_io.CTT - np.max(bins[index])) < 1.)], bins_cth);

    
    freq_lo_total_cth = hist_lo_cth/float(hist_lo_cth.sum() + hist_mpc_cth.sum() + hist_io_cth.sum())
    freq_ltmpc_total_cth = hist_ltmpc_cth/float(hist_lo_cth.sum() + hist_mpc_cth.sum() + hist_io_cth.sum())
    
  
    return([hist_lo_cth, hist_ltmpc_cth, hist_mpc_cth, hist_io_cth]);


##########################################      
def plot_pixel_histogram_cth(season, freq_lo_djf, freq_ltmpc_djf, n_lo_djf, n_ltmpc_djf, freq_lo_jja, freq_ltmpc_jja, n_lo_jja, n_ltmpc_jja):
    
    
    fig, axs = plt.subplots(nrows=5,ncols=2,figsize =[15,25])
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9,
                    wspace=0.32, hspace=0.23)
    
    bins = np.arange(0,5000,200)
    
    #axs[0,0].set_title('Summer (DJF)', fontsize = 16, bbox_to_anchor=(0.525, 0.85))
    axs[0,0].set_title('Summer (DJF)', fontsize = 16)

    axs[0,1].set_title('Winter (JJA)', fontsize = 16)
    
    
    for i in np.arange(0,5,1):
        
        # plot -70 last, turn around plot order
        j = 4-i
        
        if (n_lo_djf[i] > 0) or (n_ltmpc_djf[i] > 0) or (n_lo_jja[i] > 0) or (n_ltmpc_jja[i] > 0): 
            axs[i,0].barh(bins[:-1],100.*freq_lo_djf[j,:],height=200, align="edge", ec="k", label = 'LO:       N = '+str(int(n_lo_djf[j])))
            axs[i,0].barh(bins[:-1],100.*freq_ltmpc_djf[j,:],height=200, align="edge", ec="k", color = 'r', alpha = 0.6, label = 'LTMPC: N = '+str(int(n_ltmpc_djf[j])))
        
            axs[i,1].barh(bins[:-1],100.*freq_lo_jja[j,:],height=200, align="edge", ec="k", label = 'LO:       N = '+str(int(n_lo_jja[j])))
            axs[i,1].barh(bins[:-1],100.*freq_ltmpc_jja[j,:],height=200, align="edge", ec="k", color = 'r', alpha = 0.6, label = 'LTMPC: N = '+str(int(n_ltmpc_jja[j])))
        
        
            for k in np.arange(2):
                axs[i,k].set_ylim(0, 5000)
                axs[i,k].set_xlim(0,8)
         
                #axs[k].set_yticks(np.arange(-40,1,10))
                #axs[k].set_yticks(np.arange(-40,1,5), minor = True)

            
                axs[i,k].set_ylabel('Cloud top height (m)', fontsize = 14)
                axs[i,k].set_xlabel('Frequency of cloud type (%)', fontsize = 14)
            
                axs[i,k].legend(loc='upper left', bbox_to_anchor=(0.525, 0.85), fontsize = 12, frameon = True)
            
                axs[i,k].tick_params(left = True, right = False, bottom = True, top = False)
            
            

            #axs[2].arrow(-13., 4.5, 0, -1, head_length = 0.3, head_width = 1., fc = 'k', ec = 'k')
            #axs[2].text(-25,4.2,'T* = -13°C', fontsize = 14)
        
            sns.despine()
        
            str_a = ['A', 'C', 'E', 'G', 'I']
            str_b = ['B', 'D', 'F', 'H', 'J']
            
            str_lats = ['50°S', '55°S', '60°S', '65°S', '70°S']
    
            axs[i,0].text(0.95,0.93,str_a[i],fontsize = 22,transform=axs[i,0].transAxes)
            axs[i,1].text(0.95,0.93,str_b[i],fontsize = 22,transform=axs[i,1].transAxes)
            
            axs[i,0].text(0.55,0.93,str_lats[i],color = 'teal',fontsize = 22,transform=axs[i,0].transAxes)
            axs[i,1].text(0.55,0.93,str_lats[i],color = 'teal', fontsize = 22,transform=axs[i,1].transAxes)
            
            
    plt.savefig('Figures/final_paper/FigureS3_CTH.pdf',format = 'pdf', bbox_inches='tight')


# +
season = 'DJF'

cth_lo = np.zeros([5, 24],dtype=np.float64)
cth_ltmpc = np.zeros([5, 24],dtype=np.float64)
cth_mpc = np.zeros([5, 24],dtype=np.float64)
cth_io = np.zeros([5, 24],dtype=np.float64)


count = 0
count_longi = 0
count_lati = 0

for longi in np.arange(-180,181,5):
    
    print(longi)
        
    for lati in np.arange(-70,-45,5):
        
        try:
            result = calc_cth_at_tstar(season, lati, longi, 'all')
            
            cth_lo[count_lati,:] += result[0]
            cth_ltmpc[count_lati,:] += result[1]
            cth_mpc[count_lati,:] += result[2]
            cth_io[count_lati,:] += result[3]

        except:
            count_lati += 1
            continue
        
        
        count_lati += 1
        
        
    count_longi += 1
    count_lati = 0
    
    
cth_lo_djf = cth_lo
cth_ltmpc_djf = cth_ltmpc
cth_mpc_djf = cth_mpc
cth_io_djf = cth_io


# +
season = 'JJA'


cth_lo = np.zeros([5, 24],dtype=np.float64)
cth_ltmpc = np.zeros([5, 24],dtype=np.float64)
cth_mpc = np.zeros([5, 24],dtype=np.float64)
cth_io = np.zeros([5, 24],dtype=np.float64)


count = 0
count_longi = 0
count_lati = 0

for longi in np.arange(-180,181,5):
    
    print(longi)
        
    for lati in np.arange(-70,-45,5):
        
        try:
            result = calc_cth_at_tstar(season, lati, longi, 'all')
            
            cth_lo[count_lati,:] += result[0]
            cth_ltmpc[count_lati,:] += result[1]
            cth_mpc[count_lati,:] += result[2]
            cth_io[count_lati,:] += result[3]

        except:
            count_lati += 1
            continue
        
        
        count_lati += 1
        
        
    count_longi += 1
    count_lati = 0
    
cth_lo_jja = cth_lo
cth_ltmpc_jja = cth_ltmpc
cth_mpc_jja = cth_mpc
cth_io_jja = cth_io

# +
# Normalization
freq_lo_total_cth_djf = np.zeros([5, 24],dtype=np.float64)
freq_ltmpc_total_cth_djf = np.zeros([5, 24],dtype=np.float64)
n_lo_djf = np.zeros([5],dtype=np.float64)
n_ltmpc_djf = np.zeros([5],dtype=np.float64)
freq_lo_total_cth_jja = np.zeros([5, 24],dtype=np.float64)
freq_ltmpc_total_cth_jja = np.zeros([5, 24],dtype=np.float64)
n_lo_jja = np.zeros([5],dtype=np.float64)
n_ltmpc_jja = np.zeros([5],dtype=np.float64)


season = 'DJF'

cth_lo = cth_lo_djf
cth_ltmpc = cth_ltmpc_djf
cth_mpc = cth_mpc_djf
cth_io = cth_io_djf

for count_lati in np.arange(4,-1,-1):

    freq_lo_total_cth_djf[count_lati,:] = cth_lo[count_lati,:]/float(cth_lo[count_lati,:].sum() + cth_mpc[count_lati,:].sum() + cth_io[count_lati,:].sum())
    freq_ltmpc_total_cth_djf[count_lati,:] = cth_ltmpc[count_lati,:]/float(cth_lo[count_lati,:].sum() + cth_mpc[count_lati,:].sum() + cth_io[count_lati,:].sum())
    n_lo_djf[count_lati] = cth_lo[count_lati,:].sum()
    n_ltmpc_djf[count_lati] = cth_ltmpc[count_lati,:].sum()

    
season = 'JJA'

cth_lo = cth_lo_jja
cth_ltmpc = cth_ltmpc_jja
cth_mpc = cth_mpc_jja
cth_io = cth_io_jja

for count_lati in np.arange(4,-1,-1):

    freq_lo_total_cth_jja[count_lati,:] = cth_lo[count_lati,:]/float(cth_lo[count_lati,:].sum() + cth_mpc[count_lati,:].sum() + cth_io[count_lati,:].sum())
    freq_ltmpc_total_cth_jja[count_lati,:] = cth_ltmpc[count_lati,:]/float(cth_lo[count_lati,:].sum() + cth_mpc[count_lati,:].sum() + cth_io[count_lati,:].sum())
    n_lo_jja[count_lati] = cth_lo[count_lati,:].sum()
    n_ltmpc_jja[count_lati] = cth_ltmpc[count_lati,:].sum()
# -

# plotting CTH histograms
plot_pixel_histogram_cth(season, freq_lo_total_cth_djf, freq_ltmpc_total_cth_djf, n_lo_djf, n_ltmpc_djf, freq_lo_total_cth_jja, freq_ltmpc_total_cth_jja, n_lo_jja, n_ltmpc_jja)















