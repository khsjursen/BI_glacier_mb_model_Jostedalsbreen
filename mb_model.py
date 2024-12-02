# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:41:20 2020

@author: kasj

-------------------------------------------------------------------
Mass-balance model
-------------------------------------------------------------------

"""

#%% Libraries

# Standard libraries
import time
import os
import math as m

# External libraries
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt

# Internal libraries
from postprocessing import get_glacier_mass_balance_upd
from postprocessing import get_glacier_mass_balance_manual_gl_frac
from postprocessing import get_monthly_mb_grid # Calculation of gridded reference mb
from postprocessing import downscale_mb_grid # Downscaling of gridded reference mb
from postprocessing import get_ds_glacier_mass_balance_pergla
from postprocessing import get_ds_glacier_mass_balance_upd
from postprocessing import get_discharge
from postprocessing import get_internal_ablation
from get_climate import get_climate
from get_climate import adjust_temp
from get_climate import adjust_prec
from pot_rad import calc_pot_rad
from update_glacier_geometry import deltah_parameterization
from update_glacier_geometry import deltah_w_downscaling

#%% Function mass_balance()
def mass_balance(ds_geo, da_gl_spec_fraction, da_ca_spec_fraction, 
                config: dict, parameters: dict, name_dir_files: dict):

    """
    Function mass_balance() runs mass balance simulations.
    
    NB! Below input parameter descriptions may not be accurate. Model is not
    set up to run discharge simulations.
    ...

    Parameters
    ----------
    ds_geo (time,Y,X) : xarray.Dataset
        Dataset for the catchment/glacier containing:
        Coordinates:
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.
        Data variables:
        elevation (time,Y,X) : xarray.DataArray (float)
            Elevation in each cell in bounding box.
        glacier_fraction (time,Y,X) : xarray.DataArray (float)
            Fraction of cell inside glacier boundary.
        Attributes:
        res : float
            Resolution of DEM (cellsize).
    da_gl_spec_fraction (BREID,time,Y,X) : xarray.DataArray (float)
        DataArray with fraction of cell inside boundary of the glacier
        specified by BREID (glacier ID).
        Coordinates:
        BREID : int
            Glacier IDs.
        time : datetime64[ns]
            Time index (yearly).  
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.
        Attributes:
        res : float
            Resolution of DEM (cellsize).
    da_ca_spec_frac (vassdragNr,Y,X) : xarray.DataArray (float)
        DataArray with fraction of cell inside boundary of the catchment
        specified by vassdragNr (catchment ID).
        Coordinates:
        vassdragNr : str
            Catchment ID.
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.
        Attributes:
        res : float
            Resolution of DEM (cellsize).
    config : dict
        Dictionary of model configuration.
    parameters : dict
        Dictionary of model parameters.
    name_dir_files : dict
        Dictionary of names of directories.

    Returns
    -------
    
    """

    #%% Get parameters from dict
    T_s = parameters['threshold_temp_snow']# Threshold temperature, snow [C]
    T_0 = parameters['threshold_temp_melt'] # Threshold temperature, melt [C]
    RC_s = parameters['rad_coeff_snow'] # Radiation coefficient for snow [mm K-1 d-1 kW-1 m2]
    RC_i = parameters['rad_coeff_ice'] # Radiation coefficient for ice [mm K-1 d-1 kW-1 m2]
    C = parameters['melt_factor'] # Global degree-day factor [mm K-1 d-1]
    C_snow = parameters['melt_factor_snow'] # Degree-day factor for snow [mm K-1 d-1]
    C_ice = parameters['melt_factor_ice'] # Degree-day factor for ice [mm K-1 d-1]
    ro_f_ice = parameters['storage_coeff_ice'] # Fraction of water in ice reservoir becoming discharge
    ro_f_snow = parameters['storage_coeff_snow'] # Fraction of water in snow reservoir becoming discharge
    ro_f_firn = parameters['storage_coeff_firn'] # Fraction of water in firn reservoir becoming discharge
    firn_fac = 0.25 # Fraction of firn becoming ice
    s_r_int = 2 # Interval for snow/rain transition [C]
    prec_corr = parameters['prec_corr_factor'] # Global precipitation correction factor
    prec_lapse_rate = parameters['prec_lapse_rate'] # Precipitation lapse rate [%/100m]
    temp_corr = parameters['temp_bias_corr'] # Global temperature correction
    temp_w_corr = parameters['temp_w_bias_corr'] # Winter temperature correction
    temp_lapse_rate = parameters['temp_lapse_rate'] # Temperature lapse rate [C/100m]
    rho_w = parameters['density_water'] # Density of water [kgm-3]
    rho_i = parameters['density_ice'] # Density of ice [kgm-3]
    
    #%% Get option to return discharge from catchment
    get_ca_dis = config['get_catchment_discharge'] # Get option to return discharge (bool)

    #%% Get model type option.
    model_type = config['model_type']

    #%% Time variables
    # Get the start and end date of the simulation.
    start_date = config['simulation_start_date']
    end_date = config['simulation_end_date']

    # Convert start and end date to ordinal (time since 01-01-0001).
    start_date_ord = dt.datetime.strptime(start_date, '%Y-%m-%d').toordinal()
    end_date_ord = dt.datetime.strptime(end_date, '%Y-%m-%d').toordinal()
    
    # Get start year, end year and length of simulation period (years).
    start_year = dt.datetime.strptime(start_date, '%Y-%m-%d').year
    end_year = dt.datetime.strptime(end_date, '%Y-%m-%d').year

    # Numpy array of simulation years.    
    yr_idx = np.arange(start_year, end_year+1)

    #%% Coordinates of area of interest
    Y_coor = np.array(ds_geo.Y.values)
    X_coor = np.array(ds_geo.X.values)
    
    #%% Get glacier IDs 
    gl_id = np.array(da_gl_spec_fraction.BREID.values) # Array of int
    
    #%% Mask arrays    
    # Sum fractions over all catchments. 
    mask_all_catchments = np.array(da_ca_spec_fraction.values)
    mask_catchment_sum = np.sum(mask_all_catchments, axis=0)
    
    if config['ref_mb'] == True:
        mask_glacier_grid = np.array(ds_geo.glacier_fraction.values)
        
    elif config['update_area_from_outline'] == True and config['update_area_type'] == 'manual':
        
        map_yrs = config['map_years']
        
        map_idx = [i for i, j in enumerate(map_yrs) if j[0] == start_year]
        upd_year = map_yrs[map_idx[0]][1]
        
        mask_glacier_grid = np.array(ds_geo.glacier_fraction.sel(time = str(upd_year) + '-10-01').values)
    else:
        # Get glacier mask from DEM Dataset.
        mask_glacier_grid = np.array(ds_geo.glacier_fraction.sel(time = str(start_year) + '-10-01').values)
    
    # Add mask of glacier cells to mask of catchment cells because for 
    # single glaicer simulation, the glacier outline could be partly outside
    # the catchment outline (e.g. Nigardsbreen & catchment).
    mask_catchment_sum = mask_catchment_sum + mask_glacier_grid        
    
    # Create array with ones in cells that are part of the total catchment
    # area. The catchment area does not change with time over simulations, so 
    # the cells that are part of the evaluated catchments will be used to mask 
    # arrays and create a vectorized grid.
    mask_catchment_1 = mask_catchment_sum.copy()
    mask_catchment_1[mask_catchment_1>0] = 1
    mask_catchment_1[mask_catchment_1==0] = np.nan
    
    # Number of cells in the total catchement area. This is the length of the
    # vector of cells over which computations are made. 
    cell_count = np.nansum(mask_catchment_1)

    # mask_glacier_1 contains 1 in all cells that are part of the 
    # glacier, zero otherwise. This array will be used to mask cells 
    # that are part of the glacier regardless of the fraction of 
    # coverage.
    mask_glacier_1_grid = mask_glacier_grid.copy()
    mask_glacier_1_grid[mask_glacier_1_grid>0] = 1
    
    # Create vector of glacier mask with ones where cells are part of
    # catchment and zeros where cells are part of the catchment, but
    # not the glacier. 
    mask_glacier_1 = mask_glacier_1_grid[~np.isnan(mask_catchment_1)]
    
    # Mask glacier mask with catchment mask. New mask contains nan for
    # cells outside of the catchment, zeros for cells that are part
    # of the catchment, but are not part of the glacier and glacier 
    # fractions in cells that are part of the glacier. 
    mask_glacier_catchment = np.multiply(mask_glacier_grid, mask_catchment_1)
    
    # Create vector of glacier mask with nans removed.
    mask_glacier = mask_glacier_catchment[~np.isnan(mask_glacier_catchment)]
    
    #%% If downscaling is true, get high-res DEM and fractions. 
    if config['downscale_mb']==True or config['geometry_change_w_downscaling']==True:

        with xr.open_dataset(name_dir_files['filepath_dem'] 
                         + name_dir_files['filename_high_res_dem']) as ds_dem_hres_out:

            ds_dem_hres = ds_dem_hres_out
            da_elevation_hres = ds_dem_hres.elevation
        
        with xr.open_dataset(name_dir_files['filepath_fractions'] 
                         + name_dir_files['filename_high_res_gl_frac']) as ds_gl_frac_hres:
                         
            da_gl_spec_fraction_hres = ds_gl_frac_hres.glacier_specific_fraction
        
        if config['geometry_change_w_downscaling']==True:

            with xr.open_dataset(name_dir_files['filepath_ice_thickness'] 
                            + name_dir_files['filename_ice_thickness']) as ds_ice_thickness:
                         
                da_ice_thickness = ds_ice_thickness.ice_thickness

    #%% Initialize numpy arrays

    # Set dtype. If dtype is np.float32 there will be some minor 
    # round-off errors in the total runoff.
    set_dtype = np.float32
    #set_dtype = np.float64
    
    # Get range of dates.
    time_range = pd.date_range(start_date, end_date)
    
    # Length of the vector (1-D) representing the cells that are part of the 
    # catchment area.
    l = (int(cell_count),)

    # Initialize arrays for grid accumulation of snow, firn, ice and rain.
    # Initialize vectors for accumulation of snow, firn, ice and rain.
    snow = np.zeros(l, dtype=set_dtype)
    firn = np.zeros(l, dtype=set_dtype)
    ice = np.zeros(l, dtype=set_dtype)
    rain = np.zeros(l, dtype=set_dtype)
    refreeze = np.zeros(l, dtype=set_dtype)
    
    # Initialize vectors for meltwater from snow, firn and ice.
    snowmelt = np.zeros(l, dtype=set_dtype) # Snowmelt (includes melt of refrozen snow)
    firnmelt = np.zeros(l, dtype=set_dtype) # Firnmelt
    icemelt = np.zeros(l, dtype=set_dtype) # Icemelt

    # 2-D grids for temporary storage of accumulation and runoff
    tot_accumulation = np.zeros((len(time_range), len(Y_coor), len(X_coor)), dtype = set_dtype)

    if config['calculate_runoff'] == True:
        
        # Initialize vectors for storing produced runoff.
        runoff_glacmelt = np.zeros(l, dtype=set_dtype) # Icemelt + firnmelt
        runoff_icemelt = np.zeros(l, dtype=set_dtype)
        runoff_firnmelt = np.zeros(l, dtype=set_dtype)
        runoff_snowmelt = np.zeros(l, dtype=set_dtype) # Snowmelt
        runoff_rain =  np.zeros(l, dtype=set_dtype) # Rain

        df_runoff = pd.DataFrame({"Year": np.tile(yr_idx, len(gl_id)),
                                  "icemelt": np.nan,
                                  "firnmelt": np.nan,
                                  "snowmelt": np.nan,
                                  "rain": np.nan,
                                  "refreeze": np.nan})
        
        # Find the year with the largest glacier extent.
        glacier_masks = np.array(ds_geo.glacier_fraction.values)
        idx = np.argmax(np.sum(glacier_masks, axis=(1,2)))

        # If the glacier mask of the first year has the largest extent, use glacier mask
        # from first year as fixed glacier mask.
        if idx == 0:
            
            mask_glacier_fixed = mask_glacier.copy()
        
        # If the glacier mask of any other year has the largest extent, use this glacier mask
        # as fixed glacier mask.
        else:

            mask_glacier_grid_fixed = np.array(ds_geo.glacier_fraction[idx,:,:].values)

            # Mask glacier mask with catchment mask. New mask contains nan for
            # cells outside of the catchment, zeros for cells that are part
            # of the catchment, but are not part of the glacier and glacier 
            # fractions in cells that are part of the glacier. 
            mask_glacier_catchment_fixed = np.multiply(mask_glacier_grid_fixed, mask_catchment_1)
    
            # Create vector of glacier mask with nans removed.
            mask_glacier_fixed = mask_glacier_catchment_fixed[~np.isnan(mask_glacier_catchment_fixed)]

    if config['calculate_discharge'] == True:

        # Initialize vectors for runoff reservoirs.
        res_snowmelt = np.zeros(l, dtype=set_dtype) # Reservoir for melt from snow. Includes melt of refrozen snow.
        res_firnmelt = np.zeros(l, dtype=set_dtype) # Reservoir for melt from firn.
        res_icemelt = np.zeros(l, dtype=set_dtype) # Reservoir for melt from ice.
        res_snowrain = np.zeros(l, dtype=set_dtype) # Reservoir for rain on snow.
        res_firnrain = np.zeros(l, dtype=set_dtype) # Reservoir for rain on firn.
        res_icerain = np.zeros(l, dtype=set_dtype) # Reservoir for rain on ice.
    
        # 2-D grids for temporary storage of runoff
        tot_runoff = np.zeros((len(time_range), len(Y_coor), len(X_coor)), dtype = set_dtype)

    
    #%% Initialize storage of results
    
    cell_res = ds_geo.res
    
    # Create a Pandas Multiindex DataFrame from values in da_mb with year and
    # glacier ID as row indices and balance as columns. 
    df_mb = pd.DataFrame(
        {"Year": np.tile(yr_idx, len(gl_id)),
         "BREID": np.repeat(gl_id, len(yr_idx)),
         "Bw": np.nan,
         "Bs": np.nan,
         "Ba": np.nan,
         "Bi": np.nan,
         "R_tot": np.nan
         })   
    
    #%% Initial calculations for first year of simulation.
     
    if config['ref_mb'] == True:
        
        da_elevation = ds_geo.elevation
        
    elif config['update_area_from_outline'] == True and config['update_area_type'] == 'manual':
        
        map_idx = [i for i, j in enumerate(map_yrs) if j[0] == start_year]
        upd_year = map_yrs[map_idx[0]][1]
        
        da_elevation = ds_geo.elevation.sel(time = str(upd_year) + '-10-01')
    
    else:
        # Get elevation for the given hydrological year.
        da_elevation = ds_geo.elevation.sel(time = str(start_year) + '-10-01')
    
    # Get numpy array of elevation.
    elevation = np.array(da_elevation.values)
    
    # Mask elevation with catchment mask and vectorize.
    elev_masked = np.multiply(elevation, mask_catchment_1)
    elev_vec = elev_masked[~np.isnan(elev_masked)]
    
    # If the model type is temperature-index with potential incoming solar
    # radiation, compute potential incoming solar radiation over the current 
    # DEM.
    if model_type == 'rad_ti':

        # Function calc_pot_rad() returns 3-D array with potential direct 
        # incoming solar radiation in each cell on each day of the year.
        while not os.path.isfile(name_dir_files['filepath_temp_prec_raw'] + 'pot_rad_init.npy'):
            
            pot_rad = calc_pot_rad(da_elevation)

            with open(name_dir_files['filepath_temp_prec_raw'] + 'pot_rad_init.npy', 'wb') as f_p:
                np.save(f_p, pot_rad, allow_pickle=True)
                f_p.close()
    
        with open(name_dir_files['filepath_temp_prec_raw'] + 'pot_rad_init.npy', 'rb') as f_p:
            pot_rad = np.load(f_p)

            if config["ref_mb"] == True:
                np.nan_to_num(pot_rad, copy=False, nan=0.0)
            
            f_p.close()
    
    #%% Mass balance calculations

    # Variables to increment to determine if climate dataset for a new year
    # needs to be loaded and if the end of the hydrological year is reached.
    old_day = 0
    year_incr = 0
    old_year = 0

    # Annual refreeze.
    refr_annual = np.zeros((366, int(cell_count)))

    # Loop through all days from start_date to end_date and calculate
    # accumulation.
    for x in range(start_date_ord,end_date_ord+1):
        #%% Get time variables.
        
        # Get year from ordinal date for reading climate data file.
        year = dt.datetime.fromordinal(x).year
   
        # Get day of year from ordinal date. To be used in determining if the 
        # end of the hydrological year is reached.
        d_of_y = dt.datetime.fromordinal(x).timetuple().tm_yday
        
        #%% Get climate data. 
        
        # If starting a new year, get the climate data for the new year using
        # function get_climate() as an xarray.Dataset. The function 
        # get_climate() loads the climate dataset into memory, reducing 
        # computation time for resampling of temperature and precipitation 
        # data. Temperature and precipitation data are returned as numpy
        # arrays with data for one year. This to avoid calling the DataArray.sel()
        # method multiple times. Variable old_year is set to year to avoid extracting climate 
        # data for the same year multiple times.
        if year != old_year:

            # Get temperature and precipitation data. 
            # If it is the first day of simulation, return also the seNorge model
            # DEM for temperature and precipitation correction to elevation.
            if x == start_date_ord:
                # Get temperature and precipitation for the given year. 
                temp_raw_year, prec_raw_year, clim_data_elev = get_climate(Y_coor, X_coor, mask_catchment_1, str(year), 
                                                                          name_dir_files, rcp = config['rcp'], climate_model = config['climate_model'], return_model_h=True)

            else:
                # Save adjusted temperature and precipitation from the previous year.
                old_temp_adjusted = temp_adjusted.copy()
                old_prec_adjusted = prec_adjusted.copy()

                # Get temperature and precipitation for the given year. 
                temp_raw_year, prec_raw_year, *_ = get_climate(Y_coor, X_coor, mask_catchment_1, str(year), 
                                                             name_dir_files, rcp = config['rcp'], climate_model = config['climate_model'])

            # Adjust seNorge temperature for the current year from the seNorge 
            # model height to the elevation of the DEM.
            temp_adjusted = adjust_temp(temp_raw_year, elev_vec, clim_data_elev, temp_corr, temp_w_corr, temp_lapse_rate)
            
            # Adjust seNorge precipitation for the current year from the seNorge
            # model height to the elevation of the DEM.
            prec_adjusted = adjust_prec(prec_raw_year, elev_vec, clim_data_elev, prec_corr, prec_lapse_rate)

            # Set year to old year.
            old_year = year 

            if x == start_date_ord:
                old_temp_adjusted = temp_adjusted.copy()
                old_prec_adjusted = prec_adjusted.copy()

            # Calculate potential refreezing over the glacier area.
            pot_refreeze = get_potential_refreeze(old_temp_adjusted, temp_adjusted, mask_glacier_1)

        #%% Get adjusted seNorge temperature and precipitation 
        
        # Get seNorge temperature data for the given day.
        temp = temp_adjusted[d_of_y-1,:]

        # Get seNorge precipitation data for the given day.
        prec = prec_adjusted[d_of_y-1,:]

        if model_type == 'rad_ti':
            #%% Get potential direct solar radiation on the given day.
            pot_rad_day_grid = pot_rad[(d_of_y-1),:,:]
        
            # Array of potential direct solar radiation masked with cells part of
            # catchment. All other cells are nan.
            pot_rad_day_masked = np.multiply(pot_rad_day_grid, mask_catchment_1)
        
            # Create vector of potential direct solar radiation with nans removed.
            pot_rad_day = pot_rad_day_masked[~np.isnan(pot_rad_day_masked)]        
   
        #%% Calculate melt of snow, firn and ice.

        # If the model type is temperature-index with radiation term.
        if model_type == 'rad_ti':
        
            # Calculate snowmelt in each cell. Ms1 is snowmelt from solar 
            # radiation and M_s2 is snow melt from temperature above threshold 
            # temperature. Melt is defined as positive.
            # RC_s is in mm m2 d-1 kw-1 and pot_rad_day is in W d m-2,
            # therefore RC_s is multiplied by 1e-3. 
            #M_s1 = RC_s * 1e-3 * pot_rad_day
            M_s1 = RC_s * 1e-3 * pot_rad_day * (temp - T_0) # Alternative melt calculation.
            M_s2 = C * (temp - T_0)

            # In cells where M_s2 is negative (temperature below melt threshold), there is no melt. 
            M_s1[M_s2<0] = 0
            M_s2[M_s2<0] = 0

            # Calculate ice melt in each cell. M_i1 is ice melt from solar 
            # radiation and M_i2 is ice melt from temperature above threshold 
            # temperature. Melt is defined as positive.
            #M_i1 = RC_i * 1e-3 * pot_rad_day 
            M_i1 = RC_i * 1e-3 * pot_rad_day * (temp - T_0)# Alternative melt calculation.
            M_i2 = C * (temp - T_0)
    
            # In cells where M_i2 is negative (temperature below melt threshold), there is no melt. 
            M_i1[M_i2<0] = 0
            M_i2[M_i2<0] = 0

            # Total snow and ice melt is the sum of the two melt contributions.
            M_s = M_s1 + M_s2
            M_i = M_i1 + M_i2
        
        # If the model type is classical temperature-index.
        else:

            # Calculate snowmelt in each cell. Melt is defined as positive
            M_s = C_snow * (temp - T_0)

            # Calculate ice melt in each cell. Melt is defined as positive.
            M_i = C_ice * (temp - T_0)
        
            # In cells where melt is negative, there is no melt. 
            M_s[M_s<0] = 0
            M_i[M_i<0] = 0
            #M_s[M_s<1e-6] = 0
            #M_i[M_i<1e-6] = 0

        # The total firn melt is the average of snow and ice melt. 
        M_f = (M_s + M_i)/2

        # Calculate accumulation of snow, rain and firn.
      
        # If there is precipiation anywhere in the grid on the given day,
        # calculate accumulation of precipitation in each cell.
        if prec.max() > 0:
            
            # If the maximum temperature in the grid is below the snow/rain
            # limit, precipitation falls as snow everywhere in the grid. Add 
            # all precipitation to the snowpack and set rain to zero 
            # everywhere in the grid.
            if temp.max() <= (T_s - s_r_int/2):
                snow = snow + prec
                rain.fill(0)
            
            # If the minimum temperature in the grid is above the snow/rain
            # limit, precipitation falls as rain everywhere in the grid. All
            # precipitation is added as rain. There is no change to the snow-
            # pack. 
            elif temp.min() >= (T_s + s_r_int/2):
                rain = prec.copy()
            
            # There is both rain and snow in the grid. Calculate the fraction
            # of precipitation that falls as snow and rain. The fraction is
            # set to 1 where all precipitation falls as snow and zero where
            # all precipitation falls as rain. Add the snow to the snowpack.
            else:
                frac = (T_s - temp) / s_r_int + 0.5
                frac[frac<0] = 0
                frac[frac>1] = 1
                snow = snow + np.multiply(prec, frac)
                rain = np.multiply(prec, (1-frac))
        
        # There is no precipitation anywhere in the grid on the given day. Set
        # rain to zero. There is no change in the snowpack. 
        else: 
            rain.fill(0)

        # Update the accumulation grids for snow, firn, ice with melt and
        # calculate the total amount of meltwater in the meltwater reservoirs
        # for snow, firn and ice.
        
        # If there is melt anywhere in the grid on the given day, update the 
        # snow, ice and firnpack with accumulation and melt and calculate the
        # amount of meltwater.
        if temp.max() > T_0:

            # Calculate snow depth after accumulation and melt. Get firn and
            # ice from previous day.
            snow_new = snow - M_s
            firn_new = firn.copy()
            ice_new = ice.copy()
            
            # In grid cells where snow depth is negative, melt firn and set
            # snow depth to zero.
            if snow_new.min() < 0:

                firn_new[snow_new<0] = (firn[snow_new<0] 
                                        + snow_new[snow_new<0] 
                                        + M_s[snow_new<0] 
                                        - M_f[snow_new<0])
                snow_new[snow_new<0] = 0
       
                # In grid cells where firn depth is negative, melt ice and set
                # firn depth to zero.
                if firn_new.min() < 0:

                    ice_new[firn_new<0] = (ice[firn_new<0] 
                                           + firn_new[firn_new<0] 
                                           + M_f[firn_new<0] 
                                           - M_i[firn_new<0])
                    # Only ice in grid cells that are part of the glacier.
                    ice_new = np.multiply(ice_new, mask_glacier_1)
                    firn_new[firn_new<0] = 0 
        
            # Calculate meltwater as the difference between old and new depth
            # of the snowpack. 
            snowmelt = snow - snow_new
            
            # Calculate refreezing of snowmelt in the glacier snowpack and update the 
            # potential refreeze. Snowmelt in a cell refreezes until the refreezing 
            # potential is exhausted. Refreeze can only occur where there is snowmelt and 
            # cannot exceed the depth of the snowpack.
            if pot_refreeze.max() > 0:
                
                # Calculate new potential refreeze given snowmelt. Snowmelt can 
                # refreeze until potential refreeze is exhausted.
                pot_refreeze_new = pot_refreeze - snowmelt
                
                # Cells where snowmelt is larger than remaining potential refreeze will
                # have negative values. Refreezing potential cannot be negative, set these
                # values to zero.
                pot_refreeze_new[pot_refreeze_new<0] = 0

                # Calculate refreezing for the current time step. 
                refreeze = pot_refreeze - pot_refreeze_new
                
                # Refreeze can not exceed snow depth.
                # For cells where refreezing exceeds snowdepth, set refreezing equal to snowdepth.
                refreeze[(snow_new - refreeze)<0] = snow_new[(snow_new - refreeze)<0]
                refreeze[refreeze<1e-6]=0

                # Update potential refreeze for next time step.
                pot_refreeze = pot_refreeze_new.copy()
            
            # No refreezing occurs if refreezing potential is exhausted.
            else:
                refreeze.fill(0)

            refr_annual[(d_of_y-1),:]= refreeze

            # Update snowmelt with portion of snowmelt that has refrozen.
            snowmelt = snowmelt - refreeze

            # Calculate meltwater from firn and ice as the difference between old and new depth
            # of the firn- and icepack. 
            firnmelt = firn - firn_new
            icemelt = ice - ice_new
            
            # Update the snow-, firn- and icepack with the new depth after
            # accumulation and melt. If there is refreezing of snowmelt, this is
            # assumed to occur in the snowpack. 
            snow = snow_new + refreeze
            firn = firn_new.copy()
            ice = ice_new.copy()

        # There is no melt in the grid on the given day. The snowpack has 
        # already been updated in the accumulation if-statement. The ice- and
        # firn pack remain the same as on the previous day.
        else:
            snowmelt.fill(0)
            firnmelt.fill(0)
            icemelt.fill(0)
            refreeze.fill(0)
    
        # Update the accumulation grid with the new snow-, firn- and icepack,
        # and add to accumulation DataArray. 
        acc = (ice + firn + snow)#.astype(np.float32)
        
        tot_accumulation[(x-start_date_ord), mask_catchment_1==1] = acc

        #%% Calculate total runoff from glacier melt, snowmelt and rain
        if config['calculate_runoff'] == True:
            
            # Compute on annual basis. Set to zero at end of hydrological year. 
            runoff_glacmelt = runoff_glacmelt + icemelt + firnmelt
            runoff_icemelt = runoff_icemelt + icemelt
            runoff_firnmelt = runoff_firnmelt + firnmelt
            runoff_snowmelt = runoff_snowmelt + snowmelt
            runoff_rain = runoff_rain + rain
      
        #%% Calculate routing of meltwater and rain to reservoirs and 
        # contribution of water sources to discharge.
        if config['calculate_discharge'] == True:
            # Calculate the total melt and rain available for discharge in the
            # catchment. Some of the water is stored and the remaining water 
            # contributes to discharge.
            dis_ice = icemelt.copy()
            dis_firn = firnmelt.copy()
            dis_snow = snowmelt.copy()
            dis_rain = rain.copy()
        
            # Discharge available from snow in the glacierized (g) and 
            # non-glacierized (og) parts of the catchment.
            dis_snow_g = np.multiply(dis_snow, mask_glacier)
            dis_snow_og = dis_snow - dis_snow_g
      
            # Available meltwater from snowmelt is the sum of current storage and 
            # new water from snow on glacier available for discharge. There is no 
            # storage of meltwater from snow in the non-glacierized parts of the 
            # catchment (?).
            res_snowmelt = res_snowmelt + dis_snow_g 
            # Actual discharge from snowmelt on the glacier is a fraction 
            # ro_f_snow of the available meltwater from snowmelt. 
            dis_snow_g = ro_f_snow * res_snowmelt
            # Update the available meltwater from snowmelt for the next time step.
            res_snowmelt = res_snowmelt - dis_snow_g
        
            # Determine the amount of rain falling on snow covered and snow-free
            # parts of the catchment.
            rain_onsnow = dis_rain.copy()
            rain_onsnow[snow==0] = 0
            rain_onground = dis_rain - rain_onsnow
        
            # Available water from rain on snow is the sum of current storage and 
            # new water from rain on snow available for discharge.
            res_snowrain = res_snowrain + rain_onsnow
            # Actual discharge from rain on snow is a fraction ro_f_snow of the 
            # available water from rain on snow. 
            dis_snowrain = ro_f_snow * res_snowrain
            # Update the available water from rain on snow for the next time step.
            res_snowrain = res_snowrain - dis_snowrain
            # Update the actual discharge from rain.
            dis_rain = dis_snowrain + rain_onground # Total discharge from rain
        
            # Available meltwater from firnmelt is the sum of current storage and 
            # new water from firnmelt available for discharge. 
            res_firnmelt = res_firnmelt + dis_firn
            # Actual discharge from firnmelt is a fraction ro_f_firn of the 
            # available meltwater from firnmelt.
            dis_firn = ro_f_firn * res_firnmelt
            # Update the available water from firnmelt for the next time step.
            res_firnmelt = res_firnmelt - dis_firn
        
            # Determine the amount of rain falling on firn covered parts of the
            # catchment where there is no snow cover. Update rain falling on 
            # the ground. 
            rain_onfirn = dis_rain.copy()
            rain_onfirn[firn==0] = 0
            with np.errstate(invalid='ignore'): # Ignore warning when comparing nan
                rain_onfirn[snow>0] = 0 
            rain_onground = dis_rain - rain_onfirn
        
            # Available water from rain on firn is the sum of current storage and 
            # new water from rain on firn available for discharge.
            res_firnrain = res_firnrain + rain_onfirn # Available storage, rain on firn 
            # Actual discharge from rain on firn is a fraction ro_f_firn of the 
            # available water from rain on firn.
            dis_firnrain = ro_f_firn * res_firnrain
            # Update the available water from rain on firn for the next time step.
            res_firnrain = res_firnrain - dis_firnrain
            # Update the actual discharge from rain.
            dis_rain = dis_firnrain + rain_onground
        
            # Available meltwater from ice melt is the sum of current storage and 
            # new water from ice melt available for discharge.    
            res_icemelt = res_icemelt + dis_ice
            # Actual discharge from ice melt is a fraction ro_f_ice of the 
            # available meltwater from ice melt.
            dis_ice = ro_f_ice * res_icemelt
            # Update the available water from ice melt for the next time step.
            res_icemelt = res_icemelt - dis_ice
        
            # Determine the amount of rain falling on ice covered parts of the
            # catchment where there is no firn or snow cover. Update rain falling 
            # on the ground. 
            rain_onice = dis_rain.copy()
            rain_onice[ice==0] = 0
            with np.errstate(invalid='ignore'): # Ignore warning when comparing nan
                rain_onice[firn>0] = 0 
                rain_onice[snow>0] = 0 
            rain_onground = dis_rain - rain_onice
        
            # Available water from rain on ice is the sum of current storage and 
            # new water from rain on ice available for discharge.
            res_icerain = res_icerain + rain_onice 
            # Actual discharge from rain on ice is a fraction ro_f_ice of the 
            # available water from rain on ice.
            dis_icerain = ro_f_ice * res_icerain 
            # Update the available water from rain on ice for the next time step.
            res_icerain = res_icerain - dis_icerain 
            # Update the actual discharge from rain.
            dis_rain = dis_icerain + rain_onground 
        
            # The actual discharge from snowmelt in the whole catchment is the sum
            # of discharge from snowmelt on the glacierized and non-glacierized 
            # part of the catchment.
            dis_snow = dis_snow_g + dis_snow_og
            # The actual discharge from firn and ice melt. 
            dis_glac = dis_ice + dis_firn
            # The actual discharge from the catchment is the sum of discharge from
            # firn and ice melt on the glacier (dis_glac), discharge from snow
            # melt in the whole catchment (dis_snow) and discharge from rain in
            # the whole catchment (dis_rain).
            dis_tot = dis_glac + dis_snow + dis_rain
        
            # Add the total runoff for day in DataArray.
            tot_runoff[(x-start_date_ord), mask_catchment_1==1] = dis_tot

        #%% Update reservoir and geometry (optional) at the end of the 
        # hydrological year

        # At the end of the hydrological year, update snow, firn and ice pack.
        # Determine if the end of the hydrological year (1. October) is
        # reached. If true, a fraction of the firn pack becomes ice and the 
        # snowpack becomes firn.
        count_x = x - start_date_ord + 1
        # If end of hydrological year YYYY-09-30 (day 272), enter this 
        # if-statement. 
        # Markus used end of october (day 305), so that the if-statement was
        # accessed on the 1. november. 
        if round(count_x % 365.25) - 272 > 0:
            if count_x > (old_day + 100):
                old_day = count_x
                year_incr = year_incr + 1
            
                ice_d0 = ice.copy()
                firn_d0 = firn.copy()
                snow_d0 = snow.copy()
            
                ice_d0 = ice_d0 + np.multiply((firn_fac*firn_d0), 
                                              mask_glacier_1)
                firn_d0 = np.multiply(((1-firn_fac)*firn_d0), mask_glacier_1) 
                
                firn_d0 = firn_d0 + snow_d0
                snow_d0 = 0
            
                snow = snow_d0
                firn = firn_d0.copy()
                ice = ice_d0.copy()
                
                # Assume all refreeze occurs during melting season.
                pot_refreeze.fill(0)

                ###da_accumulation[:,:,:] = tot_accumulation
                
                # Get mass balances for the hydrological year.
                ###da_mb.loc[dict(year=year)] = get_glacier_mass_balance(
                ###    da_accumulation, da_gl_spec_fraction, gl_id, year, start_year, start_date_ord)
                
                # if config['downscale_mb'] == True:

                #     # Further up: use if downscale_mb = True, load the high-res DEM and the high-res glacier_fraction
                #     # High res glacier fraction should incorporate changes in area over time. 

                #     # Downscale mass balance from 1km DEM to high-res DEM.
                #     # Get mass balances for the hydrological year.
                #     df_mb.loc[df_mb['Year'] == year, ['Bw','Bs','Ba']] = get_ds_glacier_mass_balance(tot_accumulation, 
                #         mask_glacier_1_grid, elevation, da_elevation_hres, da_gl_spec_fraction, da_gl_spec_fraction_hres, 
                #         gl_id, year, start_year, start_date_ord)

                # else:
                    
                #     # Get mass balances for the hydrological year using 1km DEM.
                #     df_mb.loc[df_mb['Year'] == year, ['Bw','Bs','Ba']] = get_glacier_mass_balance(
                #        tot_accumulation, da_gl_spec_fraction, gl_id, year, start_year, start_date_ord)

                if config['get_internal_ablation'] == True:
                    # Calculate internal ablation. Works for single & multi-glacier case.
                    df_mb.loc[df_mb['Year'] == year, ['Bi']] = get_internal_ablation(old_prec_adjusted, 
                        prec_adjusted, elevation, da_gl_spec_fraction, mask_catchment_1, gl_id, year, start_year)
                
                if config['calculate_runoff'] == True:
        
                    # Sum runoff contributions over fixed-gauge glacier area.
                    df_runoff.loc[df_runoff['Year'] == year, ['icemelt']] = np.sum(np.multiply(runoff_icemelt, mask_glacier))/np.sum(mask_glacier_fixed)
                    df_runoff.loc[df_runoff['Year'] == year, ['firnmelt']] = np.sum(np.multiply(runoff_firnmelt, mask_glacier))/np.sum(mask_glacier_fixed)
                    df_runoff.loc[df_runoff['Year'] == year, ['snowmelt']] = np.sum(np.multiply(runoff_snowmelt, mask_glacier_fixed))/np.sum(mask_glacier_fixed)
                    df_runoff.loc[df_runoff['Year'] == year, ['rain']] = np.sum(np.multiply(runoff_rain, mask_glacier_fixed))/np.sum(mask_glacier_fixed)
                    df_runoff.loc[df_runoff['Year'] == year, ['refreeze']] = np.sum(np.multiply(refr_annual, mask_glacier_fixed))/np.sum(mask_glacier_fixed)

                    # Reset runoff storage.
                    runoff_glacmelt.fill(0)
                    runoff_icemelt.fill(0)
                    runoff_firnmelt.fill(0)
                    runoff_snowmelt.fill(0)
                    runoff_rain.fill(0)

                # Calculate refreezing for single-glacier case.
                #if year != start_year:
                    
                    # Calculate refreezing in the given year. 
                    #df_mb.loc[df_mb['Year'] == year, ['R_tot']] = np.sum(np.multiply(refr_annual, mask_glacier))/(np.sum(mask_glacier)*1e3)
                    #refr_tot[year-start_year] = np.sum(np.multiply(refr_annual, mask_glacier))/(np.sum(mask_glacier)*1e3)

                # Reset refreeze.
                refr_annual.fill(0)

                # If the option to include glacier geometry change is true,
                # and it is not the first year of simulation, update the 
                # glacier geometry. Glacier geometry cannot be updated at the 
                # end of the first hydrological year because we cannot 
                # calculate the winter balance, and hence the annual balance,
                # for the first hydrological year.
                if config['geometry_change'] == True and year >= 2006: #year != start_year:
                    
                    # For each glacier, update the glacier geometry (ice 
                    # thickness, surface elevation and area) based on the 
                    # annual mass balance.
                    if config['geometry_change_w_downscaling'] == True:
                        
                        print('geometry_change_w_downscaling')
                        print('use_function_delta_h_downscaling')
                        frac_updated, surface_elev_updated, tot_frac_updated, frac_hr_updated, surface_elev_hr_updated, tot_frac_hr_updated, ice_thickness_updated = deltah_w_downscaling(ds_geo,
                                                                                                                                                                                            ds_dem_hres,
                                                                                                                                                                                            da_gl_spec_fraction_hres, 
                                                                                                                                                                                            da_ice_thickness, 
                                                                                                                                                                                            df_mb, 
                                                                                                                                                                                            gl_id, 
                                                                                                                                                                                            year,
                                                                                                                                                                                            rho_w,
                                                                                                                                                                                            rho_i)

                        da_gl_spec_fraction_hres.loc[dict(time = str(year) + '-10-01')] = frac_hr_updated
                        ds_dem_hres.elevation.loc[dict(time = str(year) + '-10-01')] = surface_elev_hr_updated                                  
                        ds_dem_hres.glacier_fraction.loc[dict(time = str(year)+ '-10-01')] = tot_frac_hr_updated
                        da_ice_thickness.loc[dict(time = str(year) + '-10-01')] = ice_thickness_updated
                        da_elevation_hres = ds_dem_hres.elevation
                    
                    else: 
                        (frac_updated, 
                        ice_thickness_updated, 
                        surface_elev_updated, 
                        tot_frac_updated) = deltah_parameterization(ds_geo, da_gl_spec_fraction, 
                                                                   da_ice_thickness, df_mb, gl_id, 
                                                                   year, rho_w, rho_i)
                    
                    # Add updated glacier geometry to DataArrays for the new
                    # hydrological year.
                    da_gl_spec_fraction.loc[dict(time = str(year) + '-10-01')] = frac_updated
                    ds_geo.elevation.loc[dict(time = str(year) + '-10-01')] = surface_elev_updated
                    ds_geo.glacier_fraction.loc[dict(time = str(year) + '-10-01')] = tot_frac_updated
                    print('geometry updated')
                    print(year)
            
                    # Get glacier mask from DEM Dataset.
                    mask_glacier_grid = tot_frac_updated.copy()
                    
                    # mask_glacier_1 contains 1 in all cells that are part of the 
                    # glacier, zero otherwise. This array will be used to mask cells 
                    # that are part of the glacier regardless of the fraction of 
                    # coverage.
                    mask_glacier_1_grid = mask_glacier_grid.copy()
                    mask_glacier_1_grid[mask_glacier_1_grid>0] = 1
            
                    # Create vector of glacier mask with ones where cells are part of
                    # catchment and zeros where cells are part of the catchment, but
                    # not the glacier. 
                    mask_glacier_1 = mask_glacier_1_grid[~np.isnan(mask_catchment_1)]
            
                    # Mask glacier mask with catchment mask. New mask contains nan for
                    # cells outside of the catchment, zeros for cells that are part
                    # of the catchment, but are not part of the glacier and glacier 
                    # fractions in cells that are part of the glacier. 
                    mask_glacier_catchment = np.multiply(mask_glacier_grid, mask_catchment_1)
            
                    # Create vector of glacier mask with nans removed.
                    mask_glacier = mask_glacier_catchment[~np.isnan(mask_glacier_catchment)]
            
                    # Get elevation for the given hydrological year.
                    da_elevation = ds_geo.elevation.sel(time = str(year) + '-10-01')
                    
                    # Get numpy array of elevation.
                    elevation = np.array(da_elevation.values)
                    
                    # Mask elevation with catchment mask and vectorize.
                    elev_masked = np.multiply(elevation, mask_catchment_1)
                    elev_vec = elev_masked[~np.isnan(elev_masked)]
                    
                    # Function calc_pot_rad() returns 3-D array with potential direct 
                    # incoming solar radiation in each cell on each day of the year.
                    if model_type == 'rad_ti':
                        pot_rad = calc_pot_rad(da_elevation)

                    # Adjust seNorge temperature and precipitation for the given year to the new elevation.
                    temp_adjusted = adjust_temp(temp_raw_year, elev_vec, clim_data_elev, temp_corr, temp_w_corr, temp_lapse_rate)
                    prec_adjusted = adjust_prec(prec_raw_year, elev_vec, clim_data_elev, prec_corr, prec_lapse_rate)
                
                # If geometry is to be manually updated.
                elif config['update_area_from_outline']==True:

                     if config['update_area_type'] == 'auto':
                        
                        # Get glacier mask from DEM Dataset.
                        mask_glacier_grid = np.array(ds_geo.glacier_fraction.sel(time = str(year) + '-10-01').values)

                        mask_glacier_1_grid = mask_glacier_grid.copy()
                        mask_glacier_1_grid[mask_glacier_1_grid>0] = 1
            
                        # Create vector of glacier mask with ones where cells are part of
                        # catchment and zeros where cells are part of the catchment, but
                        # not the glacier. 
                        mask_glacier_1 = mask_glacier_1_grid[~np.isnan(mask_catchment_1)]
            
                        # Mask glacier mask with catchment mask. New mask contains nan for
                        # cells outside of the catchment, zeros for cells that are part
                        # of the catchment, but are not part of the glacier and glacier 
                        # fractions in cells that are part of the glacier. 
                        mask_glacier_catchment = np.multiply(mask_glacier_grid, mask_catchment_1)

                        # Create vector of glacier mask with nans removed.
                        mask_glacier = mask_glacier_catchment[~np.isnan(mask_glacier_catchment)]

                     elif config['update_area_type'] == 'manual':
                        
                        map_idx = [i for i, j in enumerate(map_yrs) if j[0] == year]
                        upd_year = map_yrs[map_idx[0]][1]
                        
                        # Get glacier mask from DEM Dataset.
                        mask_glacier_grid = np.array(ds_geo.glacier_fraction.sel(time = str(upd_year) + '-10-01').values)

                            
                        mask_glacier_1_grid = mask_glacier_grid.copy()
                        mask_glacier_1_grid[mask_glacier_1_grid>0] = 1
            
                        # Create vector of glacier mask with ones where cells are part of
                        # catchment and zeros where cells are part of the catchment, but
                        # not the glacier. 
                        mask_glacier_1 = mask_glacier_1_grid[~np.isnan(mask_catchment_1)]
            
                        # Mask glacier mask with catchment mask. New mask contains nan for
                        # cells outside of the catchment, zeros for cells that are part
                        # of the catchment, but are not part of the glacier and glacier 
                        # fractions in cells that are part of the glacier. 
                        mask_glacier_catchment = np.multiply(mask_glacier_grid, mask_catchment_1)

                        # Create vector of glacier mask with nans removed.
                        mask_glacier = mask_glacier_catchment[~np.isnan(mask_glacier_catchment)]  
    
    
    # End of for loop

    #%% Postprocessing of mass-balance DataArray to Multiindex Dataframe.
    
    if config['downscale_mb'] == True:
        
        # Get downscaled mass balance.
        #print('downscaling mass balance')
        
        ### DOES NOT WORK CURRENTLY!
        if config['update_area_from_outline'] == True and config['multi'] == True:
                        
            # Tuple with (Y, X, resolution)
            geoinfo = (da_gl_spec_fraction.Y.values, da_gl_spec_fraction.X.values, da_gl_spec_fraction.res)
            geoinfo_hres = (da_gl_spec_fraction_hres.Y.values, da_gl_spec_fraction_hres.X.values, da_gl_spec_fraction_hres.res)
            
            for yr in yr_idx:
                
                if config['update_area_type'] == 'auto':
                    
                    #mask_glacier_grid = np.array(ds_geo.glacier_fraction.sel(time = str(year) + '-10-01').values)
                    mask_glacier_grid = np.array(da_gl_spec_fraction.sel(time = str(year) + '-10-01').values)
                    mask_glacier_grid_hres = np.array(da_gl_spec_fraction_hres.sel(time = str(year) + '-10-01').values)
                    elevation = np.array(ds_geo.elevation.sel(time = str(year) + '-10-01').values)
                    elevation_hres = np.array(da_elevation_hres.sel(time = str(year) + '-10-01').values)
                    
                elif config['update_area_type'] == 'manual':
                    
                    map_idx = [i for i, j in enumerate(map_yrs) if j[0] == yr]
                    upd_year = map_yrs[map_idx[0]][1]
                        
                    # Get glacier mask from DEM Dataset.
                    #mask_glacier_grid = np.array(ds_geo.glacier_fraction.sel(time = str(upd_year) + '-10-01').values)
                    mask_glacier_grid = np.array(da_gl_spec_fraction.sel(time = str(upd_year) + '-10-01').values)
                    mask_glacier_grid_hres = np.array(da_gl_spec_fraction_hres.sel(time = str(upd_year) + '-10-01').values)
                    elevation = np.array(ds_geo.elevation.sel(time = str(upd_year) + '-10-01').values)
                    elevation_hres = np.array(da_elevation_hres.sel(time = str(upd_year) + '-10-01').values)
                    
                mask_glacier_1_grid = mask_glacier_grid.copy()
                mask_glacier_1_grid[mask_glacier_1_grid>0] = 1
                
                # Downscale mb for the entire reference area.
                # Calculate mb for each glacier based on individual glacier grid.
                # CHECK THAT FUNCTION WORKS FOR SEVERAL GLACIERS!
                df_mb.loc[df_mb['Year'] == yr, ['Bw','Bs','Ba']] = get_ds_glacier_mass_balance_pergla(config, tot_accumulation, 
                                                                                                      mask_glacier_1_grid, elevation, da_elevation_hres, da_gl_spec_fraction, da_gl_spec_fraction_hres, 
                                                                                                      gl_id, yr, start_year, start_date_ord)                                    
           
        # Get downscaled mb for single glacier (works when we use glacier mask for one glacier at a time)
        elif (config['update_area_from_outline'] == True and config['multi'] == False): 
            
            for yr in yr_idx:
                
                # Get overall glacier mask for the given year.                
                mask_glacier_grid = np.array(ds_geo.glacier_fraction.sel(time = str(year) + '-10-01').values)

                mask_glacier_1_grid = mask_glacier_grid.copy()
                mask_glacier_1_grid[mask_glacier_1_grid>0] = 1
                
                # Get downscaled mass balance for a single glacier for a given year.
                df_mb.loc[df_mb['Year'] == yr, ['Bw','Bs','Ba']] = get_ds_glacier_mass_balance_upd(tot_accumulation, 
                                                                                                   mask_glacier_1_grid, elevation, da_elevation_hres, da_gl_spec_fraction, da_gl_spec_fraction_hres, 
                                                                                                   gl_id, yr, start_year, start_date_ord)
        else:

            for yr in yr_idx:
                
                # Get downscaled mass balance for a single glacier for a given year.
                df_mb.loc[df_mb['Year'] == yr, ['Bw','Bs','Ba']] = get_ds_glacier_mass_balance_upd(tot_accumulation, 
                                                                                                   mask_glacier_1_grid, elevation, da_elevation_hres, da_gl_spec_fraction, da_gl_spec_fraction_hres, 
                                                                                                   gl_id, yr, start_year, start_date_ord)

    # Get downscaled mb for the whole reference area on a monthly basis.
    elif config['ref_mb'] == True:

        # Save accumulation array. 
        da_accumulation = xr.DataArray(data=tot_accumulation,
                                coords= {'time': time_range,
                                         'Y': Y_coor,
                                         'X': X_coor},
                                dims=["time", "Y", "X"],
                                attrs={'Name': 'Daily_total_accumulation',
                                       'res': cell_res})
        # If monthly time resolution:
        if config['time_res'] == 'monthly':

            # Get monthly reference mb on 1x1 km grid.
            da_mb_monthly = get_monthly_mb_grid(da_accumulation, ds_geo)
            
            da_mb_monthly.to_netcdf(config['filepath_simulation_files'] + config['simulation_name'] + '/results_post_pred_ref_smb_monthly/'+ 'monthly_refmb_JOB_1km_1960_2020_'+dt.datetime.now().strftime('%H_%M_%S')+'.nc')

            df_mb = da_mb_monthly

        elif config['time_res'] == 'daily':

            da_accumulation.to_netcdf(name_dir_files['filepath_results'] + 'daily_acc_refmb_JOB_1km_1960_2020.nc')


                # Downscale arrays of monthly reference mb to high resolution grid. 
                #da_mb_monthly_hr = downscale_mb_grid(da_mb_monthly, ds_geo, ds_dem_hres)
            
                #da_mb_monthly_hr.to_netcdf(name_dir_files['filepath_results'] + 'monthly_refmb_JOB_100m_1960_2020.nc')

                # Save accumulation array. 
                #da_accumulation = xr.DataArray(data=tot_accumulation,
                #                        coords= {'time': time_range,
                #                                'Y': Y_coor,
                #                                'X': X_coor},
                #                        dims=["time", "Y", "X"],
                #                        attrs={'Name': 'Daily_total_accumulation',
                #                              'res': cell_res})
            
                #mb_mod.to_netcdf(name_dir_files['filepath_results'] + 'monthly_refmb_JOB_1km_1960_2020_test_221006.nc')

            
                #Save accumulation grid to file.
                #da_accumulation.to_netcdf(name_dir_files['filepath_results'] + 'acc_grid' + '_{time:%Y_%m_%d_%H_%M_%S}.nc'.format(time=dt.datetime.now()))
    
    # If not downscaling or reference mb, calculate glacier-wide mass balance
    # on 1 km grid. 
    elif (config['ref_mb'] == False and config['downscale_mb'] == False):
        
        # Compute glacier-wide mass balane on 1km grid using area (glacier fractions) updated from
        # outlines. If multi=True, compute glacier-wide mass balance for different glaciers.
        if config['update_area_from_outline'] == True and config['multi'] == True:

            # Manually update glacier fractions based on mapping of outlines:
            if config['update_area_type'] == 'manual':
                
                #print('we are here')

                for yr in yr_idx:

                    # For each year, check which outline year it maps to.   
                    map_idx = [i for i, j in enumerate(map_yrs) if j[0] == yr]
                    upd_year = map_yrs[map_idx[0]][1]
                        
                    # Get glacier mask to use for yr from dataset.
                    mask_glacier_grid = np.array(da_gl_spec_fraction.sel(time = str(upd_year) + '-10-01').values)

                    df_mb.loc[df_mb['Year'] == yr, ['Bw','Bs','Ba']] = get_glacier_mass_balance_manual_gl_frac(
                        tot_accumulation, mask_glacier_grid, gl_id, yr, start_year, start_date_ord)
                
                df_mb.set_index(['Year', 'BREID'], inplace = True)

            # Automatically update glacier fractions based on the year:
            # Not implemented, can potentially use the last part of the if statement below.

    # Calculate glacier-wide mass balance based on coarse grid and glacier outlines.
    else:
        
        for yr in yr_idx:
            df_mb.loc[df_mb['Year'] == yr, ['Bw','Bs','Ba']] = get_glacier_mass_balance_upd(
                tot_accumulation, da_gl_spec_fraction, gl_id, yr, start_year, start_date_ord)
        
        df_mb.set_index(['Year', 'BREID'], inplace = True)
    
    
    #%% Return function output.
    
    return df_mb #, df_runoff#, da_accumulation#, ds_geo#, da_accumulation#, refr_annual ###, da_accumulation #, df_discharge, da_gl_spec_fraction, da_ice_thickness, ds_geo

# End of function mass_balance()

#%% Function get_potential_refreeze()

def get_potential_refreeze(old_temp, curr_temp, glacier_mask_ones):

    # Make 2D array of glacier mask.
    glacier_mask_ones = np.expand_dims(glacier_mask_ones, axis = (0))

    # Temperature from the end of the previous hydrological year to the end of the previous year.
    temp_fall = np.multiply(old_temp[-92:,:], glacier_mask_ones) # 1. oct to 31. dec
    
    # Temperature from the start of the current year to the end of the current hydrological year.
    temp_spring = np.multiply(curr_temp[:-92,:], glacier_mask_ones) # 1. jan to 31. sept

    # Get mean temperature over the hydrological year.
    temp_mean = np.mean(np.vstack((temp_fall, temp_spring)), axis=0)
    
    #%% Woodward et al. (1997) model

    # Calculate potential refreezing (Woodward et al., 1997).
    X_refreeze = ((-0.0069 * temp_mean) - 0.000096) * 1e3 # mm we

    # Refreezing cannot be negative.
    X_refreeze[X_refreeze < 0] = 0

    #%% Wright et al. (2007) model

    # Specific heat capacity of ice at 0 degC.
    #c_i = 2097 # [J kg-1 C-1]

    # Depth of temperature cycle.
    #d_i = 0.5 # [m]

    # Latent heat of fusion of ice.
    #L_f = 333.5 * 1e3 # [J kg-1]

    # Get winter temperatures.
    #temp_dec = np.multiply(old_temp[-31:,:], glacier_mask_ones)
    #temp_jan_mar = np.multiply(curr_temp[:-275,:], glacier_mask_ones)

    # Get mean winter temperature.
    #temp_mean_winter = np.mean(np.vstack((temp_dec, temp_jan_mar)), axis=0)

    # Calculate potential refreezing (Wright et al., 2007).
    #X_refreeze = (((c_i * d_i)/(2 * L_f)) * ((1 - (m.pi/2))*temp_mean - temp_mean_winter)) * 1e3 # mm we

    # Refreezing cannot be negative.
    #X_refreeze[X_refreeze < 0] = 0
    #X_refreeze[X_refreeze < 1e-6] = 0

    return X_refreeze

# End of function get_potential_refreeze()

#%% End of mb_model.py
