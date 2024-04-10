#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## read_emus_data.py
## Created by Aurélien STCHERBININE
## Last modified by Aurélien STCHERBININE : 13/09/2023

##----------------------------------------------------------------------------------------
"""Importation of EMM/EMUS data.
"""
##----------------------------------------------------------------------------------------
##----------------------------------------------------------------------------------------
## Packages
# Global
import numpy as np
from astropy.io import fits
from copy import deepcopy
from tqdm import tqdm
import datetime
import os
import glob
import locale

# English format for datetime
locale.setlocale(locale.LC_ALL, 'en_US.utf-8')

# Name of the current file
_py_file = 'read_emus_data.py'
_Version = 1.0

##-----------------------------------------------------------------------------------
## Read EMUS L2A data'
def read_emus_l2a(filename, folder='./'):
    """
    """
    # Open file
    filepath = os.path.join(folder, filename)
    hdu = fits.open(filepath)
    # Extract wavelength array
    wvl = hdu['WAVELENGTH'].data[0][0]  # (Y, wvl)
    # Extract data
    radiance = []
    radiance_err_rand = []
    radiance_err_sys = []
    for i in range(hdu['CAL'].header['NAXIS2']):
        # TODO: use FOV_GEOM here
        radiance.append(hdu['CAL'].data[i][0])
        radiance_err_rand.append(hdu['CAL'].data[i][1])
        radiance_err_sys.append(hdu['CAL'].data[i][2])
    radiance = np.array(radiance)   # calibrated radiance (X, Y, wvl)
    radiance_err_rand = np.array(radiance_err_rand) # random uncertainties (X, Y, wvl)
    radiance_err_sys = np.array(radiance_err_sys)   # systematic uncertainties (X, Y, wvl)
    # Radiance normalization (regarding the Mars/Sun distance)
    norm_factor = ( hdu['SC_GEOM'].data['V_SUN_POS_MARS'][:, 0] / 2.2279e8 )**2
    radiance_norm = np.swapaxes(deepcopy(np.swapaxes(radiance, 0, 2)) * norm_factor, 0, 2)
    # Output
    out_dict = {
        'wavelength' : wvl,
        'radiance' : radiance,
        'radiance_norm' : radiance_norm,
        'radiance_err_rand' : radiance_err_rand,
        'radiance_err_sys' : radiance_err_sys
        }
    return out_dict

import os
from astropy.io import fits
from copy import deepcopy
import numpy as np

def read_emus_l2a_fov(filename, folder='./'):
    """
    """
    # Open file
    filepath = os.path.join(folder, filename)
    hdu = fits.open(filepath)

    # Extract wavelength array
    wvl = hdu['WAVELENGTH'].data[0][0]  # (Y, wvl)

    # Extract spatial data
    mrh_alt = hdu["FOV_GEOM"].data['MRH_ALT']
    lat = hdu["FOV_GEOM"].data['LAT']
    lon = hdu["FOV_GEOM"].data['LON']
    primary = hdu[0].header

    # Extract data
    radiance = []
    radiance_err_rand = []
    radiance_err_sys = []
    for i in range(hdu['CAL'].header['NAXIS2']):
        radiance.append(hdu['CAL'].data[i][0])
        radiance_err_rand.append(hdu['CAL'].data[i][1])
        radiance_err_sys.append(hdu['CAL'].data[i][2])
    radiance = np.array(radiance)  # calibrated radiance (X, Y, wvl)
    radiance_err_rand = np.array(radiance_err_rand) # random uncertainties (X, Y, wvl)
    radiance_err_sys = np.array(radiance_err_sys)   # systematic uncertainties (X, Y, wvl)

    # Radiance normalization (regarding the Mars/Sun distance)
    norm_factor = (hdu['SC_GEOM'].data['V_SUN_POS_MARS'][:, 0] / 2.2279e8)**2
    radiance_norm = np.swapaxes(deepcopy(np.swapaxes(radiance, 0, 2)) * norm_factor, 0, 2)

    # Output
    out_dict = {
        'wavelength' : wvl,
        'radiance' : radiance,
        'radiance_norm' : radiance_norm,
        'radiance_err_rand' : radiance_err_rand,
        'radiance_err_sys' : radiance_err_sys,
        'mrh_alt' : mrh_alt,
        'lat' : lat,
        'lon' : lon,
        'date': primary["DATE"],
        'observation_start_time': primary["OBS_UTC"],
        'observation_end_time': primary["END_UTC"],
        'mar_year': primary["MARSYEAR"],
        'solar_long': primary["SOL_LONG"],
        'quality_flag': primary["QLTY_FLG"],
        'images_recieved': primary["IMG_RCV"]
    }

    return out_dict


##-----------------------------------------------------------------------------------
## End of code
##-----------------------------------------------------------------------------------
