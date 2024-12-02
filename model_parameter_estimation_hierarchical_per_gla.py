# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 09:22:35 2021

@author: kasj

Run MCMC for all glaciers using geodetic mass balance observations. 

"""
#%% Libraries

# Standard libraries
import sys
import datetime as dt
import pickle
import time

# External libraries
import pickle
import arviz as az
import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
from theano.compile.ops import as_op
from theano.tensor import _shared
import pandas as pd

# Internal libraries
from set_dirs import set_dirs
from run_model import run_model
from get_observations import get_glacier_obs
from get_observations import get_hugonnet_obs

#%% Function main runs MCMC/posterior predictive

def main(glacier_breid):
    
    main_dir = set_dirs('server')

    yr1=1966
    yr2=2006
    yr3=2019
    map_yrs = []
    for i in range(1957,2021):
        if 1957 <= i < 1987:
            map_yrs.append((i,yr1))
        elif 1987 <= i < 2013:
            map_yrs.append((i,yr2))
        elif 2013 <= i < 2021:
            map_yrs.append((i,yr3))
        else:
            print('year out of range')

    # Configure model.
    config_model = {"simulation_name": str(glacier_breid), # Name of simulation case (str)
                    "filepath_simulation_files": main_dir + 'simulation_data/', # Path to simulation files (str)
                    "model_type": 'cl_ti', # Type of melt model, choose 'cl_ti' for classical temperature-index and 'rad-ti' for temperature-index with radiation term. Default is 'cl-ti'. 
                    "simulation_start_date": '1997-01-01', # Start date (str) # geodetic, posterior & posterior predictive
                    "simulation_end_date": '2019-12-31', # End date (str) # geodetic, posterior & posterior predictive
                    #"simulation_start_date": '1957-01-01', # Posterior predictive
                    #"simulation_end_date": '2020-12-31', # Posterior predictive
                    "rcp": 'hist',
                    "ref_mb": False,
                    "time_res": 'monthly', # 'monthly' or 'daily'
                    "buffer": 1000, # buffer around glacier outline [m]
                    "multi": True, # only used if 'update_area_from_outline' is True and 'downscaling=True'.
                    "use_kss_dem": False,
                    "use_seNorge_dem": True, # Use seNorge DEM for simulations.
                    "climate_model": 'seNorge', #'kss',
                    "update_area_from_outline": True, # If True, update area from set of outlines. If False, no area update.
                    "update_area_type": 'manual', # only used if 'update_area_from_outline' is True. Automatic update ('auto') or manual/special case ('manual')
                    "map_years": map_yrs, # only used if 'update_area_from_outline' is True. Use if 'update_area_type' = 'manual'
                    "geometry_change": False, # Option to run model with (True) or without (False) geometry changes from delta-h parameterization.
                    "geometry_change_w_downscaling": False, # Option to run model with (True) or without (False) geometry changes from delta-h parameterization with downscaling.
                    "get_catchment_discharge": False, # Option to return discharge from a given catchment (bool).
                    "downscale_mb": False, # Option to downscale mb from seNorge grid to higher resolution DEM.
                    "get_internal_ablation": False, # Option to return internal ablation based on Oerlemans (2013).
                    "calculate_runoff": False, # Option to compute glacier runoff.
                    "calculate_discharge": False, # Option to compute discharge from catchments.
                    #"calibration_start_date": '1960-01-01', # Posterior predictive 
                    #"calibration_end_date": '2020-12-31', # Posterior predictive 
                    "calibration_start_date": '2000-01-01', # Start date for calibration period (str) # geodetic, posterior & posterior predictive
                    "calibration_end_date": '2019-12-31', # End date for calibration period (str) # geodetic, posterior & posterior predictive
                    "calibration_data": 'point', # For running MC method of Engelhardt et al. (2014). Choose gw (glacier-wide) or point (point) mass-balance for calibration (str).
                    "number_of_runs": 10, # For running MC method of Engelhardt et al. (2014). Number of calibration runs (int).
                    "observation_type": 'geodetic', # For running MCMC with PyMC: 'glac_gw_seasonal', 'glac_gw_annual', 'glac_gw_annual_10yr' or 'geodetic' (str). 
                    "run_posterior_predictive": True} # Set to True if running posterior predictive. Set to False otherwise.

    # Filepaths and filenames for files to be used in model run.
    dir_file_names = {"filepath_glacier_id": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/glacier_id/', # Path to file with glacier IDs
                      "filename_glacier_id": 'glacier_id.txt', # Name of .txt file with glacier IDs
                      "filepath_catchment_id": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/catchment_id/', # Path to file with catchment IDs
                      "filename_catchment_id": 'catchment_id.txt', # Name of .txt file with catchment IDs
                      "filepath_dem": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/dem/', # Filepath to local DEM
                      "filename_dem": 'dem_' + config_model['simulation_name'] + '_1966_2006_2019.nc', # Filename of 1 km DEM
                      "filename_high_res_dem": 'dem_' + config_model['simulation_name'] + '_1966_2006_2019_100m.nc', # Filename of 100 m DEM for downscaling
                      "filepath_ice_thickness": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/ice_thickness/', # Filepath to store dataset of ice thickness and bedrock topo
                      "filename_ice_thickness": 'ice_thickness_' + config_model['simulation_name'] + '_100m.nc', # Filename  of 100 m ice thickness maps
                      "filepath_fractions": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/fractions/', # Filepath to datasets of initial glacier and catchment fractions
                      "filename_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '_1966_2006_2019.nc', # Filename of 1 km glacier masks
                      "filename_high_res_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '_1966_2006_2019_100m.nc', # Filename of 100 m glacier masks
                      "filename_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '_1966_2006_2019.nc', # Filename of 1 km catchment masks
                      "filename_high_res_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '1966_2006_2019_100m.nc', # Filename of 100 m catchment masks
                      "filepath_parameters": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/parameters_point/', # Filepath to store parameters from calibration
                      "filepath_results": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/results/', # Path to store results
                      "filepath_temp_prec_raw": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/temp_prec_data/', # Path to files with local temp and precip data
                      "filepath_climate_data": main_dir + 'climate_data/', # Filepath to store/retreive climate data files 
                      "filename_climate_data": 'job_all_seNorge_2018_21.09_ref_smb', # Name of climate data files for Jostedalsbreen, seNorge 2018 vs. 21_09
                      #"filename_climate_data": 'job_ref_smb_ECEARTH_CCLM',
                      "filepath_obs": main_dir + 'observations/', # Filepath to files with observations of mass balance and discharge
                      "filepath_dem_base": main_dir + 'dem_base/', # Filepath to 100 m base DEM.
                      "filename_dem_base": 'DEM100_JOB_EPSG32633.tif', # Base DEM for whole of Jostedalsbreen, for creating local cropped DEM
                      #"filename_dem_base": 'DEM100_' + config_model['simulation_name'][:-4] + '_EPSG32633.tif',
                      #"filename_dem_base": 'dem_KSS_1km.nc',
                      "filepath_shp": main_dir + 'shape_files/', # Filepath to glacier outline shape files
                      "filename_shp_overview": 'shp_overview.csv', # Overview of shape files used if area is updated incrementally
                      "filename_shape_gl": 'cryoclim_GAO_NO_1999_2006_EDIT/cryoclim_GAO_NO_1999_2006_UTM_33N_EDIT.shp', # Filename of shape file with glacier IDs and outlines
                      "filename_shape_ca": 'regine_enhet/Nedborfelt_RegineEnhet_1.shp'} # Filename of shape file with catchment IDs and outlines     

    # Get type of observations to be used in parameter estimation.
    obs_type = config_model['observation_type']

    # Start and end date for calibration.
    start_time = config_model['calibration_start_date']
    end_time = config_model['calibration_end_date']
    
    # Get start year and end year for calibration.
    calibration_year_start = dt.datetime.strptime(start_time, '%Y-%m-%d').year
    calibration_year_end = dt.datetime.strptime(end_time, '%Y-%m-%d').year
    
    # Get list of glaciers. 
    if obs_type == 'dummy':
        id_list = ['dummy']
    else:
        df_id = pd.read_csv(dir_file_names['filepath_glacier_id'] 
                            + dir_file_names['filename_glacier_id'], sep=';')
        id_list = df_id['BREID'].values.tolist()
            
    # Get list of calibration years. 
    # NB! For later: This could be a list of non-consecutive years.
    yr_list = list(range(calibration_year_start, calibration_year_end + 1))
            
    # Get subsets of modeled and observed mass balance based on the list 
    # of glaciers for which observations are available and the list of 
    # calibration years.
    idx = pd.IndexSlice

    # *** USE THIS FOR GLACIOLOGICAL GLACIER-WIDE SEASONAL BALANCES ***
    if obs_type == 'glac_gw_seasonal':
        
        # Get DataFrame with observations.
        df_mb_obs = get_glacier_obs(dir_file_names, 'gw')  
        mb_obs_sub = df_mb_obs.loc[idx[yr_list,id_list],:]

        # NOT NEEDED: Get list of glaciers with available observations.
        ###id_list = df_mb_obs.index.get_level_values('BREID').unique().tolist()
        # Note that mb_mod_sub could contain balances for glaciers with short
        # observation records such that there are years in mb_mod_sub for 
        # which there are no observations for within the calibration period. 
        # Check which index values that are part of mb_mod_sub that are NOT 
        # part of mb_obs_sub.
        #idx_diff = mb_mod_sub.index.difference(mb_obs_sub.index)    
        # Remove rows from mb_mod_sub corresponding to idx_diff.
        #mb_mod_sub = mb_mod_sub.drop(idx_diff))

        # Get vectors of winter and summer balances.
        all_mb_obs_w = mb_obs_sub['Bw'].to_numpy(copy=True)
        all_mb_obs_s = mb_obs_sub['Bs'].to_numpy(copy=True)

        # Get standard deviation of seasonal balances.
        df_sigma = pd.read_csv(dir_file_names['filepath_obs'] + 'sigma_obs.csv', sep=';')
        sigma_w = df_sigma.loc[df_sigma['BREID'] == id_list[0], 'sigma_w'].values[0]
        sigma_s = df_sigma.loc[df_sigma['BREID'] == id_list[0], 'sigma_s'].values[0]

        # If running posterior predictive, add dummy observations for years where no 
        # observations exist, e.g. 1960 to start of record. 
        if config_model['run_posterior_predictive'] == True:

                # Get first and last year of record and calculate number of years to add.
                first_obs = df_mb_obs.index[0][0]
                last_obs = df_mb_obs.index[-1][0]
                yrs_to_add_start = first_obs - calibration_year_start
                yrs_to_add_end = calibration_year_end - last_obs
                print(yrs_to_add_start)
                print(yrs_to_add_end)

                # If dummy observations are needed at start. 
                if yrs_to_add_start > 0:
                
                    # Add dummy observations to record. 
                    add_obs_w = np.ones((yrs_to_add_start,))
                    add_obs_s = np.ones((yrs_to_add_start,))*-1
                    mb_obs_w = np.hstack((add_obs_w, all_mb_obs_w))
                    mb_obs_s = np.hstack((add_obs_s, all_mb_obs_s))
                    print(mb_obs_s.shape)

                # If dummy observations are needed at end. 
                if yrs_to_add_end > 0:

                    if yrs_to_add_start > 0:
                        all_mb_obs_w = mb_obs_w.copy()
                        all_mb_obs_s = mb_obs_s.copy()
                
                    # Add dummy observations to record. 
                    add_obs_w = np.ones((yrs_to_add_end,))
                    add_obs_s = np.ones((yrs_to_add_end,))*-1
                    mb_obs_w = np.hstack((all_mb_obs_w, add_obs_w))
                    mb_obs_s = np.hstack((all_mb_obs_s, add_obs_w))
            
                if (yrs_to_add_end == 0) and (yrs_to_add_start == 0):
                    mb_obs_w = all_mb_obs_w.copy()
                    mb_obs_s = all_mb_obs_s.copy()

                mb_obs_a = mb_obs_w + mb_obs_s

                sigma_a = df_sigma.loc[df_sigma['BREID'] == id_list[0], 'sigma_a'].values[0]

    # *** USE THIS FOR GLACIOLOGICAL GLACIER-WIDE ANNUAL BALANCES ***
    elif obs_type == 'glac_gw_annual':

        # Get DataFrame with observations.
        df_mb_obs = get_glacier_obs(dir_file_names, 'gw')  
        mb_obs_sub = df_mb_obs.loc[idx[yr_list,id_list],:]

        # Get vector of annual balances.        
        mb_obs_a = mb_obs_sub['Ba'].to_numpy(copy=True)

        # Get standard deviation of annual balance.        
        df_sigma = pd.read_csv(dir_file_names['filepath_obs'] + 'sigma_obs.csv', sep=';')
        sigma_a = df_sigma.loc[df_sigma['BREID'] == id_list[0], 'sigma_a'].values[0]
    
    # *** USE THIS FOR GLACIOLOGICAL POINT BALANCES ***
    #elif obs_type = 'glac_point':
        
        # Get DataFrame of point balances.
        #df_mb_obs = get_glacier_obs(config_model, dir_file_names, data_type='point')
    
        # Get rows in year_list (now 112 values for Austdalsbreen from 2000-2009)
        #point_obs_sub = df_mb_obs.loc[df_mb_obs['Year'].isin(yr_list)]
   
    # *** USE THIS FOR GLACIER-WIDE DECADAL BALANCES ***
    elif obs_type == 'glac_gw_annual_10yr':
    
        # Get DataFrame with observations.
        df_mb_obs = get_glacier_obs(dir_file_names, 'gw')  
        mb_obs_sub = df_mb_obs.loc[idx[yr_list,id_list],:]

        # Get vector of annual balances.        
        mb_obs_a = mb_obs_sub['Ba'].to_numpy(copy=True)

        # Get standard deviation of annual balance.        
        df_sigma = pd.read_csv(dir_file_names['filepath_obs'] + 'sigma_obs_hugonnet.csv', sep=';')
        sigma_geod = df_sigma.loc[df_sigma['BREID'] == id_list[0], 'sigma_a'].values

        # Get 10-year rates of annual balances.
        mb_obs_10yr_rates = np.array([mb_obs_a[0:10].mean(), mb_obs_a[10:20].mean()])

    # *** USE THIS FOR GEODETIC BALANCES FROM HUGONNET ET AL. (2021) ***
    elif obs_type == 'geodetic':
    
        # If multiple glaciers, get hugonnet observations and sigma for each of the glaciers and store
        # in array.
        if config_model['multi'] == True:
    
            # Get DataFrame of observations.
            df_hugonnet_mb_obs = get_hugonnet_obs(dir_file_names)

            # Get unique glacier ids
            unique_breids = df_hugonnet_mb_obs.index.get_level_values('BREID').unique()

            # Creating a mapping from old to new positions
            order_mapping = {breid: id_list.index(breid) for breid in unique_breids}

            # Sorting the DataFrame based on the order of glacier ids in glacier id file (same as modelled)
            df_hugonnet_mb_obs_sorted = df_hugonnet_mb_obs.sort_index(level=0).sort_index(level=1, key=lambda idx: idx.map(order_mapping))

            # Get 10-year balances.
            hugonnet_mb_obs = df_hugonnet_mb_obs_sorted['Ba'].to_numpy(copy=True)
            print(hugonnet_mb_obs)
            #print(hugonnet_mb_obs.shape)

            # Get sigma for each mass balance estimate from sigma based on 
            # Hugonnet et al. (2021).
            hugonnet_sigma = df_hugonnet_mb_obs_sorted['sigma'].to_numpy(copy=True)
            print(hugonnet_sigma)
            #print(hugonnet_sigma.shape)

        # If only one glacier.
        else:
            # Get DataFrame of observations.
            df_hugonnet_mb_obs = get_hugonnet_obs(dir_file_names)

            # Get 10-year balances.
            hugonnet_mb_obs = df_hugonnet_mb_obs['Ba'].to_numpy(copy=True)
            #print(hugonnet_mb_obs)

            # Get sigma for each mass balance estimate from sigma based on 
            # Hugonnet et al. (2021).
            hugonnet_sigma = df_hugonnet_mb_obs['sigma'].to_numpy(copy=True)
            #print(hugonnet_sigma)

    # *** USE THIS FOR DUMMY OBSERVATIONS, E.G. IN POSTERIOR PREDICTIVE
    elif obs_type == 'dummy':

        all_mb_obs_w = np.ones(61*82)
        all_mb_obs_s = np.ones(61*82)

        # Get standard deviation of seasonal balances.
        df_sigma = pd.read_csv(dir_file_names['filepath_obs'] + 'sigma_obs.csv', sep=';')
        sigma_w = df_sigma.loc[df_sigma['BREID'] == 2297, 'sigma_w'].values[0]
        sigma_s = df_sigma.loc[df_sigma['BREID'] == 2297, 'sigma_s'].values[0]

    else:
        sys.exit('Observation type not found.')
    
#%% Define black-box model with Theano decorator.
    
    # Set up a shell function that takes parameters to be calibrated as input.
    # In the shell, these parameters are added to the "parameters" dict and
    # the model is run as normal. 

    #@as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar], otypes=[tt.dvector, tt.dvector, tt.dvector, tt.dvector, tt.dvector, tt.dvector])
    #@as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar], otypes=[tt.dvector, tt.dvector, tt.dvector, tt.dvector])#, tt.dvector])#, tt.dvector, tt.dvector, tt.dvector, tt.dvector, tt.dvector])
    @as_op(itypes=[tt.lscalar, tt.dscalar, tt.dscalar], otypes=[tt.dvector, tt.dvector, tt.dvector, tt.dvector])#, tt.dvector])#, tt.dvector, tt.dvector, tt.dvector, tt.dvector, tt.dvector])
    #@as_op(itypes=[tt.dscalar], otypes=[tt.dvector, tt.dvector, tt.dvector, tt.dvector])#, tt.dvector])#, tt.dvector, tt.dvector, tt.dvector, tt.dvector, tt.dvector])
    #def shell(p_corr, degree_day_factor, rad_coeff_snow):#rad_ice):
    #def shell(p_corr, degree_day_factor_snow, temp_corr):#, temp_w_corr):#rad_ice):    #def shell(degree_day_factor, thresh_melt):
    def shell(gl_breid, p_corr, temp_corr):#degree_day_factor_snow):#temp_w_corr):#rad_ice):    #def shell(degree_day_factor, thresh_melt):
    #def shell(p_corr):#temp_w_corr):#rad_ice):    #def shell(degree_day_factor, thresh_melt):

        main_dir = set_dirs('server')
        
        yr1=1966
        yr2=2006
        yr3=2019
        map_yrs = []
        for i in range(1957,2021):
            if 1957 <= i < 1987:
                map_yrs.append((i,yr1))
            elif 1987 <= i < 2013:
                map_yrs.append((i,yr2))
            elif 2013 <= i < 2021:
                map_yrs.append((i,yr3))
            else:
                print('year out of range')

        #gl_breid = 

        # Configure model.
        config_model = {"simulation_name": str(gl_breid), # Name of simulation case (str)
                        "filepath_simulation_files": main_dir + 'simulation_data/', # Path to simulation files (str)
                        "model_type": 'cl_ti', # Type of melt model, choose 'cl_ti' for classical temperature-index and 'rad-ti' for temperature-index with radiation term. Default is 'cl-ti'. 
                        "simulation_start_date": '1997-01-01', # Start date (str) # geodetic, posterior & posterior predictive
                        "simulation_end_date": '2019-12-31', # End date (str) # geodetic, posteriorv & posterior predictive
                        #"simulation_start_date": '1957-01-01', # Posterior predictive
                        #"simulation_end_date": '2020-12-31', # Posterior predictive
                        "rcp": 'hist',
                        "ref_mb": False,
                        "time_res": 'monthly',
                        "buffer": 1000, # buffer around glacier outline [m]
                        "multi": True, # only used if 'update_area_from_outline' is True and 'downscaling=True'.
                        "use_kss_dem": False,
                        "use_seNorge_dem": True, # Use seNorge DEM for simulations.
                        "climate_model": 'seNorge', #'kss',
                        "update_area_from_outline": True, # If True, update area from set of outlines. If False, no area update.
                        "update_area_type": 'manual', # only used if 'update_area_from_outline' is True. Automatic update ('auto') or manual/special case ('manual')
                        "map_years": map_yrs, # only used if 'update_area_from_outline' is True. Use if 'update_area_type' = 'manual'
                        "geometry_change": False, # Option to run model with (True) or without (False) geometry changes from delta-h parameterization.
                        "geometry_change_w_downscaling": False, # Option to run model with (True) or without (False) geometry changes from delta-h parameterization with downscaling.
                        "get_catchment_discharge": False, # Option to return discharge from a given catchment (bool).
                        "downscale_mb": False, # Option to downscale mb from seNorge grid to higher resolution DEM.
                        "get_internal_ablation": False, # Option to return internal ablation based on Oerlemans (2013).
                        "calculate_runoff": False, # Option to compute glacier runoff.
                        "calculate_discharge": False, # Option to compute discharge from catchments.
                        "calibration_start_date": '2000-01-01', # Start date for calibration period (str) # geodetic, posterior & posterior predictive
                        "calibration_end_date": '2019-12-31', # End date for calibration period (str) # geodetic, posterior & posterior predictive
                        #"calibration_start_date": '1960-01-01', # Posterior predictive
                        #"calibration_end_date": '2020-12-31', # Posterior predictive
                        "calibration_data": 'point', # For running MC method of Engelhardt et al. (2014). Choose gw (glacier-wide) or point (point) mass-balance for calibration (str).
                        "number_of_runs": 10, # For running MC method of Engelhardt et al. (2014). Number of calibration runs (int).
                        "observation_type": 'geodetic', # For running MCMC with PyMC: 'glac_gw_seasonal', 'glac_gw_annual', 'glac_gw_annual_10yr' or 'geodetic' (str). 
                        "run_posterior_predictive": True} # Set to True if running posterior predictive. Set to False otherwise.

        # Get monthly temperature lapse rates from file. 
        with open(main_dir + 'lapse_rates/temp_lapse_rates.txt', 'rb') as fp:
            temp_m_lr = pickle.load(fp)
    
        # Parameters for mass balance and discharge simulations.
        parameters = {"threshold_temp_snow" : 1.0, # Threshold temperature for snow [deg C]
                      "threshold_temp_melt" : 0.0, # Threshold temperature for melt [deg C]
                      "rad_coeff_snow": 1.0,#rad_coeff_snow, # Radiation coefficient for snow (only used for RAD-TI model option)
                      "rad_coeff_ice": 1.0, #rad_coeff_snow/0.7, # Radiation coefficient for ice (only used for RAD-TI model option)
                      "melt_factor": 1.0, #degree_day_factor, # Melt factor (not used)
                      "melt_factor_snow": 3.58,#degree_day_factor_snow,#3.58, # degree_day_factor_snow #Melt factor for snow (mm w.e. degC d-1)
                      "melt_factor_ice": (3.58/0.7),#(degree_day_factor_snow/0.7), #3.58/0.7, # Melt factor for ice (mm w.e. degC d-1)
                      "storage_coeff_ice": 0.72, # Storage coefficient for ice (for runoff simulations)
                      "storage_coeff_snow": 0.19, # Storage coefficient for snow (for runoff simulations)
                      "storage_coeff_firn": 0.66, # Storage coefficient for firn (for runoff simulations)
                      "prec_corr_factor": p_corr, #1.25,#p_corr,#1.317,#p_corr, # Global precipitation correction [-]
                      "prec_lapse_rate": 0.1, # Precipitation lapse rate [100m-1] (positive upwards)
                      "temp_bias_corr": temp_corr, #-0.14,#0.0,#temp_corr, # Correction for temperature bias [C]
                      "temp_w_bias_corr": 0.0, # Correction for winter temperature bias [C]
                      "temp_lapse_rate": temp_m_lr, # Temperature lapse rate [C 100m-1] (negative upwards)
                      "density_water": 1000, # Density of water [kgm-3]
                      "density_ice": 850} # Density of ice [kgm-3]

        # Filepaths and filenames for files to be used in model run.
        dir_file_names = {"filepath_glacier_id": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/glacier_id/', # Path to file with glacier IDs
                          "filename_glacier_id": 'glacier_id.txt', # Name of .txt file with glacier IDs
                          "filepath_catchment_id": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/catchment_id/', # Path to file with catchment IDs
                          "filename_catchment_id": 'catchment_id.txt', # Name of .txt file with catchment IDs
                          "filepath_dem": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/dem/', # Filepath to local DEM
                          "filename_dem": 'dem_' + config_model['simulation_name'] + '_1966_2006_2019.nc', # Filename of 1 km DEM
                          "filename_high_res_dem": 'dem_' + config_model['simulation_name'] + '_1966_2006_2019_100m.nc', # Filename of 100 m DEM for downscaling
                          "filepath_ice_thickness": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/ice_thickness/', # Filepath to store dataset of ice thickness and bedrock topo
                          "filename_ice_thickness": 'ice_thickness_' + config_model['simulation_name'] + '_100m.nc', # Filename  of 100 m ice thickness maps
                          "filepath_fractions": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/fractions/', # Filepath to datasets of initial glacier and catchment fractions
                          "filename_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '_1966_2006_2019.nc', # Filename of 1 km glacier masks
                          "filename_high_res_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '_1966_2006_2019_100m.nc', # Filename of 100 m glacier masks
                          "filename_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '_1966_2006_2019.nc', # Filename of 1 km catchment masks
                          "filename_high_res_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '1966_2006_2019_100m.nc', # Filename of 100 m catchment masks
                          "filepath_parameters": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/parameters_point/', # Filepath to store parameters from calibration
                          "filepath_results": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/results/', # Path to store results
                          "filepath_temp_prec_raw": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/temp_prec_data/', # Path to files with local temp and precip data
                          "filepath_climate_data": main_dir + 'climate_data/', # Filepath to store/retreive climate data files 
                          "filename_climate_data": 'job_all_seNorge_2018_21.09_ref_smb', # Name of climate data files for Jostedalsbreen, seNorge 2018 vs. 21_09
                          #"filename_climate_data": 'job_ref_smb_ECEARTH_CCLM',
                          "filepath_obs": main_dir + 'observations/', # Filepath to files with observations of mass balance and discharge
                          "filepath_dem_base": main_dir + 'dem_base/', # Filepath to 100 m base DEM.
                          "filename_dem_base": 'DEM100_JOB_EPSG32633.tif', # Base DEM for whole of Jostedalsbreen, for creating local cropped DEM
                          #"filename_dem_base": 'DEM100_' + config_model['simulation_name'][:-4] + '_EPSG32633.tif',
                          #"filename_dem_base": 'dem_KSS_1km.nc',
                          "filepath_shp": main_dir + 'shape_files/', # Filepath to glacier outline shape files
                          "filename_shp_overview": 'shp_overview.csv', # Overview of shape files used if area is updated incrementally
                          "filename_shape_gl": 'cryoclim_GAO_NO_1999_2006_EDIT/cryoclim_GAO_NO_1999_2006_UTM_33N_EDIT.shp', # Filename of shape file with glacier IDs and outlines
                          "filename_shape_ca": 'regine_enhet/Nedborfelt_RegineEnhet_1.shp'} # Filename of shape file with catchment IDs and outlines     

        # Run mass balance model with parameter settings.
        mb_mod = run_model(config_model, parameters, dir_file_names)

        # Get start and end time of calibration period. 
        start_time = config_model['calibration_start_date']
        end_time = config_model['calibration_end_date']
    
        # Get start year and end year for calibration.
        calibration_year_start = dt.datetime.strptime(start_time, '%Y-%m-%d').year
        calibration_year_end = dt.datetime.strptime(end_time, '%Y-%m-%d').year

        # Get list of glaciers with available observations.
        #id_list = mb_obs["BREID"].unique().tolist() # Does not work for MultiIndex.
        id_list = mb_mod.index.get_level_values('BREID').unique().tolist()
        
        # Make list of calibration years to crop modelled balances. 
        # NB! For later: This could be a list of non-consecutive years.
        yr_list = list(range(calibration_year_start, calibration_year_end + 1))
            
        # Get subsets of modeled and observed mass balance based on the list 
        # of glaciers for which observations are available and the list of 
        # calibration years.
        idx = pd.IndexSlice
        mb_mod_sub = mb_mod.loc[idx[yr_list,id_list],:]
                
        # Get vectors of glacier-wide summer, winter and annual balances.
        mb_mod_winter = mb_mod_sub['Bw'].to_numpy(copy=True)
        mb_mod_summer = mb_mod_sub['Bs'].to_numpy(copy=True)
        mb_mod_annual = mb_mod_sub['Ba'].to_numpy(copy=True)
        
        # Calculate 10-year mass balance rates.
        mb_mod_10yr_rates = np.array([mb_mod_annual[0:10].mean(), mb_mod_annual[10:20].mean()])

        # Return modelled winter, summer, annual and 10-yr rates of mass blance.
        return mb_mod_winter, mb_mod_summer, mb_mod_annual, mb_mod_10yr_rates


#%% Define PyMC model.

    gl_breid = theano.shared(glacier_breid, name='gl_breid')

    # Declare model context.
    mb_model = pm.Model()
    
    # Set up model context.
    with mb_model:
        
        # Priors based on posterior of glacier-wide glaciological mass balance
        prec_corr = pm.TruncatedNormal("prec_corr", mu=1.25, sigma=0.04, lower=0) #sigma 0.04
        T_corr = pm.Normal("T_corr", mu=-0.14, sigma=0.34)
        #sigma_mod = pm.HalfNormal("sigma_mod", sigma=0.67) # sigma is in this case a scale parameter such that mean = scale*sqrt(2)/sqrt(pi) for the half-normal distribution
        sigma_mod_10 = 0.14


        # Uncertainty in geodetic observations (case add sigma)
        sigma_geodetic = tt.sqrt(hugonnet_sigma**2 + sigma_mod_10**2) # Posterior
        #sigma_tot_w = tt.sqrt(sigma_w_all_mean**2 + sigma_mod**2)
        #sigma_tot_s = tt.sqrt(sigma_s_all_mean**2 + sigma_mod**2)       
        
        # Expected value of outcome is the modelled mass balance.
        mb_mod_w, mb_mod_s, mb_mod_a, mb_mod_geod = shell(gl_breid, prec_corr, T_corr)#DDF_snow) #Tcorr
 
        
        # Observations.
        # Use this for Hugonnet data:
        data_mb_a_hugonnet = pm.Data('data_mb_a_hugonnet', hugonnet_mb_obs) # annual, hugonnet

        # Expected value as deterministic RVs. Saves these to inference data file.
        #Use this for hugonnet data:
        mu_mb_geod = pm.Deterministic('mu_mb_geod', mb_mod_geod)

        # Likelihood with observation and model error
        # Use this for hugonnet data
        mb_obs_geodetic = pm.Normal("mb_obs_geodetic", mu=mu_mb_geod, sigma=sigma_geodetic, observed=data_mb_a_hugonnet)


    # Sample posterior.
    # with mb_model:
        
    # #    # Choose sampler.
    # #    #step = pm.Metropolis()
    #     step = pm.DEMetropolisZ()

    #     # Draw posterior samples.
    #     idata_post = pm.sample(draws=4000, tune=2000, step=step, return_inferencedata=True, chains=4, cores=20, progressbar=True, idata_kwargs=dict(log_likelihood=False))
        
    # # Save InferenceData with posteriors to netcdf file.
    # idata_post.to_netcdf(main_dir + 'simulation_data/'+str(glacier_breid)+'/results/idata_DEMZ_t2000_s4000_c4_geodetic_'+str(glacier_breid)+'_w_sigma10_MFandP_new.nc')

    # Sample prior.
    #with mb_model:

    #     # Sample prior for given variables.
    #    start_time = time.time()
    #    prior = pm.sample_prior_predictive(10000)
    #    print("Prior predictive sampling took " + str (time.time() - start_time) + ' seconds.')

    #     # Save prior data in InferenceData object.
    #    idata_prior = az.from_pymc3(prior=prior)

    # # Save InferenceData with prior to netcdf file.
    #idata_prior.to_netcdf(main_dir + 'simulation_data/Vesledalsbreen/results/idata_DEMZ_s10000_prior_pred_wo_sigma_mod.nc')

    # Sample posterior predictive.
    with mb_model:

        # Get inference data.
        idata_sampled = az.from_netcdf(main_dir + 'simulation_data/'+str(glacier_breid)+'/results/idata_DEMZ_t2000_s4000_c4_geodetic_'+str(glacier_breid)+'_w_sigma10_TandP.nc')

        print("Starting posterior predictive sampling.")
        # Sample posterior predictive from trace.
        start_time = time.time()
        fast_post_pred = pm.fast_sample_posterior_predictive(trace=idata_sampled, samples=5000)
        print("Posterior predictive sampling took " + str(time.time() - start_time) + ' seconds.')

        # Save posterior predicitve data in InferenceData object.
        idata_post_pred = az.from_pymc3(posterior_predictive=fast_post_pred)

    # Save InferenceData with posterior predictive to netcdf file.
    idata_post_pred.to_netcdf(main_dir + 'simulation_data/'+str(glacier_breid)+'/results/idata_DEMZ_s5000_geodetic_'+str(glacier_breid)+'_w_sigma10_TandP_post_pred.nc')

if __name__ == '__main__':
    # Pcorr:
    #gl_id_list = [2246, 2250, 2255, 2258, 2265, 2266, 2271, 2273, 2280, 2281] #List 1
    #gl_id_list = [2283, 2284, 2285, 2289, 2291, 2294, 2296, 2297, 2299, 2301] #List 2 
    gl_id_list = [2305, 2308, 2309, 2311, 2316, 2318, 2319, 2320, 2321, 2322] #List 3 
    #gl_id_list = [2323, 2324, 2325, 2326, 2327, 2328, 2329, 2331, 2332, 2333] #List 4 
    #gl_id_list = [2334, 2336, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2347] #List 5 
    #gl_id_list = [2348, 2349, 2352, 2354, 2355, 2358, 2360, 2361, 2362, 2364] #List 6 
    #gl_id_list = [2367, 2369, 2451, 2453, 2457, 2459, 2461, 2463, 2465, 2468, 2471] #List 7
    #gl_id_list = [2474, 2476, 2478, 2480, 2481, 2485, 2486, 2487, 2488, 2489, 2490] #List 8 
    
    for i in gl_id_list:
    
        main(i)
