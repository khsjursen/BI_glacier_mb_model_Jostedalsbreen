# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 09:22:35 2021

@author: kasj

Run MCMC for five glaciers using glaciological seasonal mass balance observations. 

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

def main():
    
    main_dir = set_dirs('server')

    # yr1=1966
    # yr2=2006
    # yr3=2019
    map_yrs = []
    # for i in range(1957,2021):
    #     if 1957 <= i < 2000:
    #         map_yrs.append((i,yr1))
    #     elif 2000 <= i < 2013:
    #         map_yrs.append((i,yr2))
    #     elif 2013 <= i < 2021:
    #         map_yrs.append((i,yr3))
    #     else:
    #         print('year out of range')

    # For calibration with multiple glaciers:
    glacier_tup = ({"simulation_name": 'Nigardsbreen',
                    #"simulation_start_date": '1957-01-01', # Start date (str)
                    #"simulation_end_date": '2020-12-31', # End date (str) 
                    #"calibration_start_date": '1962-01-01', # Start date for calibration period (str)
                    #"calibration_end_date": '2020-12-31', # End date for calibration period (str)
                    #"simulation_start_date": '1957-01-01', # Reverse calibration years
                    #"simulation_end_date": '2019-12-31', # Reverse calibration years
                    #"calibration_start_date": '1963-01-01', # Reverse calibration years
                    #"calibration_end_date": '2019-12-31', # Reverse calibration years        
                    "simulation_start_date": '1957-01-01', # Posterior predictive
                    "simulation_end_date": '2020-12-31', # Posterior predictive
                    "calibration_start_date": '1962-01-01', # Posterior predictive
                    "calibration_end_date": '2020-12-31', # Posterior predictive
                    },
                   {"simulation_name": 'Austdalsbreen',
                    #"simulation_start_date": '1981-01-01', # Start date (str)
                    #"simulation_end_date": '2020-12-31', # End date (str) 
                    #"calibration_start_date": '1988-01-01', # Start date for calibration period (str)
                    #"calibration_end_date": '2020-12-31', # End date for calibration period (str)
                    #"simulation_start_date": '1981-01-01', # Reverse calibration years
                    #"simulation_end_date": '2019-12-31', # Reverse calibration years
                    #"calibration_start_date": '1989-01-01', # Reverse calibration years
                    #"calibration_end_date": '2019-12-31', # Reverse calibration years  
                    "simulation_start_date": '1981-01-01', # Posterior predictive
                    "simulation_end_date": '2020-12-31', # Posterior predictive
                    "calibration_start_date": '1988-01-01', # Posterior predictive
                    "calibration_end_date": '2020-12-31', # Posterior predictive
                    },
                   {"simulation_name": 'Vesledalsbreen',
                    #"simulation_start_date": '1957-01-01', # Start date (str)
                    #"simulation_end_date": '1972-12-31', # End date (str) 
                    #"calibration_start_date": '1968-01-01', # Start date for calibration period (str)
                    #"calibration_end_date": '1972-12-31', # End date for calibration period (str)
                    #"simulation_start_date": '1957-01-01', # Reverse calibration years
                    #"simulation_end_date": '1971-12-31', # Reverse calibration years
                    #"calibration_start_date": '1967-01-01', # Reverse calibration years
                    #"calibration_end_date": '1971-12-31', # Reverse calibration years 
                    "simulation_start_date": '1957-01-01', # Posterior predictive
                    "simulation_end_date": '1972-12-31', # Posterior predictive
                    "calibration_start_date": '1967-01-01', # Posterior predictive
                    "calibration_end_date": '1972-12-31', # Posterior predictive
                    },
                   {"simulation_name": 'Tunsbergdalsbreen',
                    #"simulation_start_date": '1957-01-01', # Start date (str)
                    #"simulation_end_date": '1972-12-31', # End date (str) 
                    #"calibration_start_date": '1966-01-01', # Start date for calibration period (str)
                    #"calibration_end_date": '1972-12-31', # End date for calibration period (str)
                    #"simulation_start_date": '1957-01-01', # Reverse calibration years
                    #"simulation_end_date": '1971-12-31', # Reverse calibration years
                    #"calibration_start_date": '1967-01-01', # Reverse calibration years
                    #"calibration_end_date": '1971-12-31', # Reverse calibration years 
                    "simulation_start_date": '1957-01-01', # Posterior predictive
                    "simulation_end_date": '1972-12-31', # Posterior predictive
                    "calibration_start_date": '1966-01-01', # Posterior predictive
                    "calibration_end_date": '1972-12-31', # Posterior predictive
                    },
                   {"simulation_name": 'Supphellebreen',
                    #"simulation_start_date": '1957-01-01', # Start date (str)
                    #"simulation_end_date": '1966-12-31', # End date (str) 
                    #"calibration_start_date": '1964-01-01', # Start date for calibration period (str)
                    #"calibration_end_date": '1966-12-31', # End date for calibration period (str)
                    #"simulation_start_date": '1957-01-01', # Reverse calibration years
                    #"simulation_end_date": '1965-12-31', # Reverse calibration years
                    #"calibration_start_date": '1965-01-01', # Reverse calibration years
                    #"calibration_end_date": '1967-12-31', # Reverse calibration years 
                    "simulation_start_date": '1957-01-01', # Posterior predictive
                    "simulation_end_date": '1982-12-31', # Posterior predictive
                    "calibration_start_date": '1964-01-01', # Posterior predictive
                    "calibration_end_date": '1982-12-31', # Posterior predictive
                    },
               )

    for glacier in glacier_tup:
        
        # Configure model.
        config_model = {"simulation_name": glacier['simulation_name'], # Name of simulation case (str)
                        "filepath_simulation_files": main_dir + 'simulation_data/', # Path to simulation files (str)
                        "model_type": 'cl_ti', # Type of melt model, choose 'cl_ti' for classical temperature-index and 'rad-ti' for temperature-index with radiation term. Default is 'cl-ti'. 
                        "simulation_start_date": glacier['simulation_start_date'], # Start date (str)
                        "simulation_end_date": glacier['simulation_end_date'], # End date (str)
                        "rcp": 'hist',
                        "ref_mb": False,
                        "buffer": 1000,#6000, # buffer around glacier outline [m]
                        "multi": False, # only used if 'update_area_from_outline' is True and 'downscaling=True'.
                        "use_kss_dem": False,
                        "use_seNorge_dem": True, # Use seNorge DEM for simulations.
                        "climate_model": 'seNorge', #'kss',
                        "update_area_from_outline": True, # If True, update area from set of outlines. If False, no area update.
                        "update_area_type": 'auto', # only used if 'update_area_from_outline' is True. Automatic update ('auto') or manual/special case ('manual')
                        "map_years": map_yrs, # only used if 'update_area_from_outline' is True. Use if 'update_area_type' = 'manual'
                        "geometry_change": False, # Option to run model with (True) or without (False) geometry changes from delta-h parameterization.
                        "geometry_change_w_downscaling": False, # Option to run model with (True) or without (False) geometry changes from delta-h parameterization with downscaling.
                        "get_catchment_discharge": False, # Option to return discharge from a given catchment (bool).
                        "downscale_mb": False, # Option to downscale mb from seNorge grid to higher resolution DEM.
                        "get_internal_ablation": False, # Option to return internal ablation based on Oerlemans (2013).
                        "calculate_runoff": False, # Option to compute glacier runoff.
                        "calculate_discharge": False, # Option to compute discharge from catchments.
                        "calibration_start_date": glacier['calibration_start_date'], # Start date for calibration period (str)
                        "calibration_end_date": glacier['calibration_end_date'], # End date for calibration period (str)
                        "calibration_data": 'point', # For running MC method of Engelhardt et al. (2014). Choose gw (glacier-wide) or point (point) mass-balance for calibration (str).
                        "number_of_runs": 10, # For running MC method of Engelhardt et al. (2014). Number of calibration runs (int).
                        "observation_type": 'glac_gw_seasonal', # For running MCMC with PyMC: 'glac_gw_seasonal', 'glac_gw_annual', 'glac_gw_annual_10yr' or 'geodetic' (str). 
                        "run_posterior_predictive": True} # Set to True if running posterior predictive. Set to False otherwise.

        # Filepaths and filenames for files to be used in model run.
        dir_file_names = {"filepath_glacier_id": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/glacier_id/', # Path to file with glacier IDs
                          "filename_glacier_id": 'glacier_id.txt', # Name of .txt file with glacier IDs
                          "filepath_catchment_id": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/catchment_id/', # Path to file with catchment IDs
                          "filename_catchment_id": 'catchment_id.txt', # Name of .txt file with catchment IDs
                          "filepath_dem": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/dem/', # Filepath to local DEM
                          "filename_dem": 'dem_' + config_model['simulation_name'] + '.nc', # Filename of 1 km DEM
                          "filename_high_res_dem": 'dem_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m DEM for downscaling
                          "filepath_ice_thickness": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/ice_thickness/', # Filepath to store dataset of ice thickness and bedrock topo
                          "filename_ice_thickness": 'ice_thickness_' + config_model['simulation_name'] + '_100m.nc', # Filename  of 100 m ice thickness maps
                          "filepath_fractions": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/fractions/', # Filepath to datasets of initial glacier and catchment fractions
                          "filename_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '.nc', # Filename of 1 km glacier masks
                          "filename_high_res_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m glacier masks
                          "filename_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '.nc', # Filename of 1 km catchment masks
                          "filename_high_res_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m catchment masks
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

                # If dummy observations are needed at start. 
                if yrs_to_add_start > 0:
                
                    # Add dummy observations to record. 
                    add_obs_w = np.ones((yrs_to_add_start,))
                    add_obs_s = np.ones((yrs_to_add_start,))*-1
                    mb_obs_w = np.hstack((add_obs_w, all_mb_obs_w))
                    mb_obs_s = np.hstack((add_obs_s, all_mb_obs_s))

                # If dummy observations are needed at end. 
                if yrs_to_add_end > 0:
                
                    # Add dummy observations to record. 
                    add_obs_w = np.ones((yrs_to_add_end,))
                    add_obs_s = np.ones((yrs_to_add_end,))*-1
                    mb_obs_w = np.hstack((all_mb_obs_w, add_obs_w))
                    mb_obs_s = np.hstack((all_mb_obs_s, add_obs_w))
            
                if yrs_to_add_end == 0:
                    mb_obs_w = all_mb_obs_w.copy()
                    mb_obs_s = all_mb_obs_s.copy()

                mb_obs_a = mb_obs_w + mb_obs_s

                sigma_a = df_sigma.loc[df_sigma['BREID'] == id_list[0], 'sigma_a'].values[0]
            
            # Choose every second year for calibration
            else:
                
                mb_obs_s = all_mb_obs_s[::2].copy()
                mb_obs_w = all_mb_obs_w[::2].copy()

            #print(mb_obs_s.shape)
            #print(mb_obs_w.shape)       

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
    
            # Get DataFrame of observations.
            df_hugonnet_mb_obs = get_hugonnet_obs(dir_file_names)

            # Get 10-year balances.
            hugonnet_mb_obs = df_hugonnet_mb_obs['Ba'].to_numpy(copy=True)

            # Get sigma for each mass balance estimate from sigma based on 
            # Hugonnet et al. (2021).
            hugonnet_sigma = df_hugonnet_mb_obs['sigma'].to_numpy(copy=True)

        else:
            sys.exit('Observation type not found.')

        # Nigardsbreen, Austdalsbreen, Vesledalsbreen, Tunsbergdalsbreen and Supphellebreen
        # Concat observations for all glaciers
        if glacier['simulation_name'] == 'Nigardsbreen':

            mb_obs_w_all = mb_obs_w.copy()
            mb_obs_s_all = mb_obs_s.copy()
            sigma_s_all = np.array(sigma_s)
            sigma_w_all = np.array(sigma_w)

        else:
            mb_obs_w_all = np.hstack((mb_obs_w_all, mb_obs_w)) 
            mb_obs_s_all = np.hstack((mb_obs_s_all, mb_obs_s))
            sigma_s_all = np.hstack((sigma_s_all, np.array(sigma_s)))
            sigma_w_all = np.hstack((sigma_s_all, np.array(sigma_s)))
    
#%% NIGARDSBREEN
# Define black-box model with Theano decorator.
    
    # Set up a shell function that takes parameters to be calibrated as input.
    # In the shell, these parameters are added to the "parameters" dict and
    # the model is run as normal. 

    @as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar], otypes=[tt.dvector, tt.dvector, tt.dvector])#, tt.dvector])#, tt.dvector])#, tt.dvector, tt.dvector, tt.dvector, tt.dvector, tt.dvector])
    #def shell_nig(p_corr, degree_day_factor_snow, degree_day_factor_ice):
    def shell_nig(p_corr, degree_day_factor_snow, temp_corr):

        main_dir = set_dirs('server')

        map_yrs = []

        # Configure model.
        config_model = {"simulation_name": 'Nigardsbreen', # Name of simulation case (str)
                        "filepath_simulation_files": main_dir + 'simulation_data/', # Path to simulation files (str)
                        "model_type": 'cl_ti', # Type of melt model, choose 'cl_ti' for classical temperature-index and 'rad-ti' for temperature-index with radiation term. Default is 'cl-ti'. 
                        "simulation_start_date": '1957-01-01', # Start date (str)
                        "simulation_end_date": '2020-12-31', # End date (str)
                        #"simulation_start_date": '1957-01-01', # Reverse years
                        #"simulation_end_date": '2019-12-31', # Reverse years
                        "rcp": 'hist',
                        "ref_mb": False,
                        "buffer": 1000,#6000, # buffer around glacier outline [m]
                        "multi": False, # only used if 'update_area_from_outline' is True and 'downscaling=True'.
                        "use_kss_dem": False,
                        "use_seNorge_dem": True, # Use seNorge DEM for simulations.
                        "climate_model": 'seNorge', #'kss',
                        "update_area_from_outline": True, # If True, update area from set of outlines. If False, no area update.
                        "update_area_type": 'auto', # only used if 'update_area_from_outline' is True. Automatic update ('auto') or manual/special case ('manual')
                        "map_years": map_yrs, # only used if 'update_area_from_outline' is True. Use if 'update_area_type' = 'manual'
                        "geometry_change": False, # Option to run model with (True) or without (False) geometry changes from delta-h parameterization.
                        "geometry_change_w_downscaling": False, # Option to run model with (True) or without (False) geometry changes from delta-h parameterization with downscaling.
                        "get_catchment_discharge": False, # Option to return discharge from a given catchment (bool).
                        "downscale_mb": False, # Option to downscale mb from seNorge grid to higher resolution DEM.
                        "get_internal_ablation": False, # Option to return internal ablation based on Oerlemans (2013).
                        "calculate_runoff": False, # Option to compute glacier runoff.
                        "calculate_discharge": False, # Option to compute discharge from catchments.
                        "calibration_start_date": '1962-01-01', # Start date for calibration period (str)
                        "calibration_end_date": '2020-12-31', # End date for calibration period (str)
                        #"calibration_start_date": '1963-01-01', # Reverse years
                        #"calibration_end_date": '2019-12-31', # Reverse years
                        "calibration_data": 'point', # For running MC method of Engelhardt et al. (2014). Choose gw (glacier-wide) or point (point) mass-balance for calibration (str).
                        "number_of_runs": 10, # For running MC method of Engelhardt et al. (2014). Number of calibration runs (int).
                        "observation_type": 'glac_gw_seasonal', # For running MCMC with PyMC: 'glac_gw_seasonal', 'glac_gw_annual', 'glac_gw_annual_10yr' or 'geodetic' (str). 
                        "run_posterior_predictive": True} # Set to True if running posterior predictive. Set to False otherwise.

        # Get monthly temperature lapse rates from file. 
        with open(main_dir + 'lapse_rates/temp_lapse_rates.txt', 'rb') as fp:
            temp_m_lr = pickle.load(fp)
    
        # Parameters for mass balance and discharge simulations.
        parameters = {"threshold_temp_snow" : 1.0, # Threshold temperature for snow [deg C]
                      "threshold_temp_melt" : 0.0, # Threshold temperature for melt [deg C]
                      "rad_coeff_snow": 0.0, # Radiation coefficient for snow (only used for RAD-TI model option)
                      "rad_coeff_ice": 0.0, # Radiation coefficient for ice (only used for RAD-TI model option)
                      "melt_factor": 3.5, # Melt factor (not used)
                      "melt_factor_snow": degree_day_factor_snow, # Melt factor for snow (mm w.e. degC d-1)
                      "melt_factor_ice": (degree_day_factor_snow/0.7), # Melt factor for ice (mm w.e. degC d-1)
                      "storage_coeff_ice": 0.72, # Storage coefficient for ice (for runoff simulations)
                      "storage_coeff_snow": 0.19, # Storage coefficient for snow (for runoff simulations)
                      "storage_coeff_firn": 0.66, # Storage coefficient for firn (for runoff simulations)
                      "prec_corr_factor": p_corr,#1.317,#p_corr, # Global precipitation correction [-]
                      "prec_lapse_rate": 0.1, # Precipitation lapse rate [100m-1] (positive upwards)
                      "temp_bias_corr": temp_corr, # Correction for temperature bias [C]
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
                          "filename_dem": 'dem_' + config_model['simulation_name'] + '.nc', # Filename of 1 km DEM
                          "filename_high_res_dem": 'dem_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m DEM for downscaling
                          "filepath_ice_thickness": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/ice_thickness/', # Filepath to store dataset of ice thickness and bedrock topo
                          "filename_ice_thickness": 'ice_thickness_' + config_model['simulation_name'] + '_100m.nc', # Filename  of 100 m ice thickness maps
                          "filepath_fractions": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/fractions/', # Filepath to datasets of initial glacier and catchment fractions
                          "filename_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '.nc', # Filename of 1 km glacier masks
                          "filename_high_res_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m glacier masks
                          "filename_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '.nc', # Filename of 1 km catchment masks
                          "filename_high_res_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m catchment masks
                          "filepath_parameters": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/parameters_point/', # Filepath to store parameters from calibration
                          "filepath_results": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/results/', # Path to store results
                          "filepath_temp_prec_raw": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/temp_prec_data/', # Path to files with local temp and precip data
                          "filepath_climate_data": main_dir + 'climate_data/', # Filepath to store/retreive climate data files 
                          "filename_climate_data": 'job_all_seNorge_2018_21.09_ref_smb', # Name of climate data files for Jostedalsbreen, seNorge 2018 vs. 21_09
                          #"filename_climate_data": 'job_ref_smb_ECEARTH_CCLM',
                          "filepath_obs": main_dir + 'observations/', # Filepath to files with observations of mass balance and discharge
                          "filepath_dem_base": main_dir + 'dem_base/', # Filepath to 100 m base DEM.
                          "filename_dem_base": 'DEM100_JOB_EPSG32633.tif', # Base DEM for whole of Jostedalsbreen, for creating local cropped DEM
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

        # If Austdalsbreen (BREID 2478) is in the list of glacier IDs, add the
        # calculated calving loss (NVE) to the modelled summer and annual balance. The calving
        # loss is included in the glaciological summer and annual balance observations by NVE. 
        # N! This only works for single-glacier cases with the way that it is
        # added to arrays of summer and annual balance. 
        if 2478 in id_list:

            # Get DataFrame with calving calculations.
            df_calving_obs = get_glacier_obs(dir_file_names, 'calving') 
            
            # Get calving calculations for the given period. 
            calving_obs_sub = df_calving_obs.loc[idx[yr_list,id_list],:]

            # Get vector of calving volumes.        
            calving_obs_vol = calving_obs_sub['Bcalv'].to_numpy(copy=True) # Calving volume [10e6 m3]

            # Get vector of glacier area used in mass balance observations.
            area = calving_obs_sub['Area'].to_numpy(copy=True) # Area [km2]

            # Get specific calving balance in m w.e.
            calving_spec_balance = calving_obs_vol / area
            
            # For posterior predictive
            if config_model['run_posterior_predictive'] == True:
                calving_spec_balance = np.concatenate([np.zeros(26), calving_spec_balance])

            # Add calculated calving balance (loss) to glacier wide summer and annual balance as 
            # this is not modelled.
            mb_mod_summer = mb_mod_summer + calving_spec_balance
            mb_mod_annual = mb_mod_annual + calving_spec_balance

        # Every other year of simulation used for calibration
        if config_model['run_posterior_predictive'] == False:
            
            mb_mod_winter_cropped = mb_mod_winter[::2].copy()
            mb_mod_summer_cropped = mb_mod_summer[::2].copy()
            mb_mod_annual_cropped = mb_mod_annual[::2].copy()

            return mb_mod_winter_cropped, mb_mod_summer_cropped, mb_mod_annual_cropped
        
        else:

            return mb_mod_winter, mb_mod_summer, mb_mod_annual

        # Calculate 10-year mass balance rates.
        #mb_mod_10yr_rates = np.array([mb_mod_annual[0:10].mean(), mb_mod_annual[10:20].mean()])

        # Return modelled winter, summer, annual and 10-yr rates of mass blance.
        #return mb_mod_winter_cropped, mb_mod_summer_cropped, mb_mod_annual_cropped#, mb_mod_10yr_rates

#%% AUSTDALSBREEN
# Define black-box model with Theano decorator.
    
    # Set up a shell function that takes parameters to be calibrated as input.
    # In the shell, these parameters are added to the "parameters" dict and
    # the model is run as normal. 

    @as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar], otypes=[tt.dvector, tt.dvector, tt.dvector])#, tt.dvector])#, tt.dvector])#, tt.dvector, tt.dvector, tt.dvector, tt.dvector, tt.dvector])
    #def shell_aus(p_corr, degree_day_factor_snow, degree_day_factor_ice):
    def shell_aus(p_corr, degree_day_factor_snow, temp_corr):

        main_dir = set_dirs('server')

        map_yrs = []

        # Configure model.
        config_model = {"simulation_name": 'Austdalsbreen', # Name of simulation case (str)
                        "filepath_simulation_files": main_dir + 'simulation_data/', # Path to simulation files (str)
                        "model_type": 'cl_ti', # Type of melt model, choose 'cl_ti' for classical temperature-index and 'rad-ti' for temperature-index with radiation term. Default is 'cl-ti'. 
                        "simulation_start_date": '1981-01-01', # Start date (str)
                        "simulation_end_date": '2020-12-31', # End date (str)
                        #"simulation_start_date": '1981-01-01', # Reverse years
                        #"simulation_end_date": '2019-12-31', # Reverse years
                        "rcp": 'hist',
                        "ref_mb": False,
                        "buffer": 1000,#6000, # buffer around glacier outline [m]
                        "multi": False, # only used if 'update_area_from_outline' is True and 'downscaling=True'.
                        "use_kss_dem": False,
                        "use_seNorge_dem": True, # Use seNorge DEM for simulations.
                        "climate_model": 'seNorge',#'kss',
                        "update_area_from_outline": True, # If True, update area from set of outlines. If False, no area update.
                        "update_area_type": 'auto', # only used if 'update_area_from_outline' is True. Automatic update ('auto') or manual/special case ('manual')
                        "map_years": map_yrs, # only used if 'update_area_from_outline' is True. Use if 'update_area_type' = 'manual'
                        "geometry_change": False, # Option to run model with (True) or without (False) geometry changes from delta-h parameterization.
                        "geometry_change_w_downscaling": False, # Option to run model with (True) or without (False) geometry changes from delta-h parameterization with downscaling.
                        "get_catchment_discharge": False, # Option to return discharge from a given catchment (bool).
                        "downscale_mb": False, # Option to downscale mb from seNorge grid to higher resolution DEM.
                        "get_internal_ablation": False, # Option to return internal ablation based on Oerlemans (2013).
                        "calculate_runoff": False, # Option to compute glacier runoff.
                        "calculate_discharge": False, # Option to compute discharge from catchments.
                        "calibration_start_date": '1988-01-01', # Start date for calibration period (str)
                        "calibration_end_date": '2020-12-31', # End date for calibration period (str) 
                        #"calibration_start_date": '1989-01-01', # Reverse years
                        #"calibration_end_date": '2019-12-31', # Reverse years                      
                        "calibration_data": 'point', # For running MC method of Engelhardt et al. (2014). Choose gw (glacier-wide) or point (point) mass-balance for calibration (str).
                        "number_of_runs": 10, # For running MC method of Engelhardt et al. (2014). Number of calibration runs (int).
                        "observation_type": 'glac_gw_seasonal', # For running MCMC with PyMC: 'glac_gw_seasonal', 'glac_gw_annual', 'glac_gw_annual_10yr' or 'geodetic' (str). 
                        "run_posterior_predictive": True} # Set to True if running posterior predictive. Set to False otherwise.

        # Get monthly temperature lapse rates from file. 
        with open(main_dir + 'lapse_rates/temp_lapse_rates.txt', 'rb') as fp:
            temp_m_lr = pickle.load(fp)
    
        # Parameters for mass balance and discharge simulations.
        parameters = {"threshold_temp_snow" : 1.0, # Threshold temperature for snow [deg C]
                      "threshold_temp_melt" : 0.0, # Threshold temperature for melt [deg C]
                      "rad_coeff_snow": 0.0, # Radiation coefficient for snow (only used for RAD-TI model option)
                      "rad_coeff_ice": 0.0, # Radiation coefficient for ice (only used for RAD-TI model option)
                      "melt_factor": 3.5, # Melt factor (not used)
                      "melt_factor_snow": degree_day_factor_snow, # Melt factor for snow (mm w.e. degC d-1)
                      "melt_factor_ice": (degree_day_factor_snow/0.7), # Melt factor for ice (mm w.e. degC d-1)
                      "storage_coeff_ice": 0.72, # Storage coefficient for ice (for runoff simulations)
                      "storage_coeff_snow": 0.19, # Storage coefficient for snow (for runoff simulations)
                      "storage_coeff_firn": 0.66, # Storage coefficient for firn (for runoff simulations)
                      "prec_corr_factor": p_corr,#1.317,#p_corr, # Global precipitation correction [-]
                      "prec_lapse_rate": 0.1, # Precipitation lapse rate [100m-1] (positive upwards)
                      "temp_bias_corr": temp_corr, # Correction for temperature bias [C]
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
                          "filename_dem": 'dem_' + config_model['simulation_name'] + '.nc', # Filename of 1 km DEM
                          "filename_high_res_dem": 'dem_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m DEM for downscaling
                          "filepath_ice_thickness": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/ice_thickness/', # Filepath to store dataset of ice thickness and bedrock topo
                          "filename_ice_thickness": 'ice_thickness_' + config_model['simulation_name'] + '_100m.nc', # Filename  of 100 m ice thickness maps
                          "filepath_fractions": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/fractions/', # Filepath to datasets of initial glacier and catchment fractions
                          "filename_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '.nc', # Filename of 1 km glacier masks
                          "filename_high_res_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m glacier masks
                          "filename_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '.nc', # Filename of 1 km catchment masks
                          "filename_high_res_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m catchment masks
                          "filepath_parameters": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/parameters_point/', # Filepath to store parameters from calibration
                          "filepath_results": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/results/', # Path to store results
                          "filepath_temp_prec_raw": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/temp_prec_data/', # Path to files with local temp and precip data
                          "filepath_climate_data": main_dir + 'climate_data/', # Filepath to store/retreive climate data files 
                          "filename_climate_data": 'job_all_seNorge_2018_21.09_ref_smb', # Name of climate data files for Jostedalsbreen, seNorge 2018 vs. 21_09
                          #"filename_climate_data": 'job_ref_smb_ECEARTH_CCLM',
                          "filepath_obs": main_dir + 'observations/', # Filepath to files with observations of mass balance and discharge
                          "filepath_dem_base": main_dir + 'dem_base/', # Filepath to 100 m base DEM.
                          "filename_dem_base": 'DEM100_JOB_EPSG32633.tif', # Base DEM for whole of Jostedalsbreen, for creating local cropped DEM
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

        # If Austdalsbreen (BREID 2478) is in the list of glacier IDs, add the
        # calculated calving loss (NVE) to the modelled summer and annual balance. The calving
        # loss is included in the glaciological summer and annual balance observations by NVE. 
        # N! This only works for single-glacier cases with the way that it is
        # added to arrays of summer and annual balance. 
        if 2478 in id_list:

            # Get DataFrame with calving calculations.
            df_calving_obs = get_glacier_obs(dir_file_names, 'calving') 
            
            # Get calving calculations for the given period. 
            calving_obs_sub = df_calving_obs.loc[idx[yr_list,id_list],:]

            # Get vector of calving volumes.        
            calving_obs_vol = calving_obs_sub['Bcalv'].to_numpy(copy=True) # Calving volume [10e6 m3]

            # Get vector of glacier area used in mass balance observations.
            area = calving_obs_sub['Area'].to_numpy(copy=True) # Area [km2]

            # Get specific calving balance in m w.e.
            calving_spec_balance = calving_obs_vol / area
            
            # For posterior predictive
            #if config_model['run_posterior_predictive'] == True:
            #    calving_spec_balance = np.concatenate([np.zeros(26), calving_spec_balance])

            # Add calculated calving balance (loss) to glacier wide summer and annual balance as 
            # this is not modelled.
            mb_mod_summer = mb_mod_summer + calving_spec_balance
            mb_mod_annual = mb_mod_annual + calving_spec_balance

        # Every other year of simulation used for calibration
        if config_model['run_posterior_predictive'] == False:
            
            mb_mod_winter_cropped = mb_mod_winter[::2].copy()
            mb_mod_summer_cropped = mb_mod_summer[::2].copy()
            mb_mod_annual_cropped = mb_mod_annual[::2].copy()

            return mb_mod_winter_cropped, mb_mod_summer_cropped, mb_mod_annual_cropped
        
        else:

            return mb_mod_winter, mb_mod_summer, mb_mod_annual

        # Calculate 10-year mass balance rates.
        #mb_mod_10yr_rates = np.array([mb_mod_annual[0:10].mean(), mb_mod_annual[10:20].mean()])

        # Return modelled winter, summer, annual and 10-yr rates of mass blance.
        #return mb_mod_winter, mb_mod_summer, mb_mod_annual#, mb_mod_10yr_rates

#%% VESLEDALSBREEN
# Define black-box model with Theano decorator.
    
    # Set up a shell function that takes parameters to be calibrated as input.
    # In the shell, these parameters are added to the "parameters" dict and
    # the model is run as normal. 

    @as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar], otypes=[tt.dvector, tt.dvector, tt.dvector])#, tt.dvector])#, tt.dvector])#, tt.dvector, tt.dvector, tt.dvector, tt.dvector, tt.dvector])
    #def shell_ves(p_corr, degree_day_factor_snow, degree_day_factor_ice):
    def shell_ves(p_corr, degree_day_factor_snow, temp_corr):

        main_dir = set_dirs('server')

        map_yrs = []

        # Configure model.
        config_model = {"simulation_name": 'Vesledalsbreen', # Name of simulation case (str)
                        "filepath_simulation_files": main_dir + 'simulation_data/', # Path to simulation files (str)
                        "model_type": 'cl_ti', # Type of melt model, choose 'cl_ti' for classical temperature-index and 'rad-ti' for temperature-index with radiation term. Default is 'cl-ti'. 
                        "simulation_start_date": '1957-01-01', # Start date (str)
                        "simulation_end_date": '1972-12-31', # End date (str)
                        #"simulation_start_date": '1957-01-01', # Reverse years
                        #"simulation_end_date": '1971-12-31', # Reverse years
                        "rcp": 'hist',
                        "ref_mb": False,
                        "buffer": 1000,#6000, # buffer around glacier outline [m]
                        "multi": False, # only used if 'update_area_from_outline' is True and 'downscaling=True'.
                        "use_kss_dem": False,
                        "use_seNorge_dem": True, # Use seNorge DEM for simulations.
                        "climate_model": 'seNorge',#'kss',
                        "update_area_from_outline": True, # If True, update area from set of outlines. If False, no area update.
                        "update_area_type": 'auto', # only used if 'update_area_from_outline' is True. Automatic update ('auto') or manual/special case ('manual')
                        "map_years": map_yrs, # only used if 'update_area_from_outline' is True. Use if 'update_area_type' = 'manual'
                        "geometry_change": False, # Option to run model with (True) or without (False) geometry changes from delta-h parameterization.
                        "geometry_change_w_downscaling": False, # Option to run model with (True) or without (False) geometry changes from delta-h parameterization with downscaling.
                        "get_catchment_discharge": False, # Option to return discharge from a given catchment (bool).
                        "downscale_mb": False, # Option to downscale mb from seNorge grid to higher resolution DEM.
                        "get_internal_ablation": False, # Option to return internal ablation based on Oerlemans (2013).
                        "calculate_runoff": False, # Option to compute glacier runoff.
                        "calculate_discharge": False, # Option to compute discharge from catchments.
                        "calibration_start_date": '1967-01-01', # Posterior predictive
                        #"calibration_start_date": '1968-01-01', # Start date for calibration period (str)
                        "calibration_end_date": '1972-12-31', # End date for calibration period (str)
                        #"calibration_start_date": '1967-01-01', # Reverse years
                        #"calibration_end_date": '1971-12-31', # Reverse years    
                        "calibration_data": 'point', # For running MC method of Engelhardt et al. (2014). Choose gw (glacier-wide) or point (point) mass-balance for calibration (str).
                        "number_of_runs": 10, # For running MC method of Engelhardt et al. (2014). Number of calibration runs (int).
                        "observation_type": 'glac_gw_seasonal', # For running MCMC with PyMC: 'glac_gw_seasonal', 'glac_gw_annual', 'glac_gw_annual_10yr' or 'geodetic' (str). 
                        "run_posterior_predictive": True} # Set to True if running posterior predictive. Set to False otherwise.

        # Get monthly temperature lapse rates from file. 
        with open(main_dir + 'lapse_rates/temp_lapse_rates.txt', 'rb') as fp:
            temp_m_lr = pickle.load(fp)
    
        # Parameters for mass balance and discharge simulations.
        parameters = {"threshold_temp_snow" : 1.0, # Threshold temperature for snow [deg C]
                      "threshold_temp_melt" : 0.0, # Threshold temperature for melt [deg C]
                      "rad_coeff_snow": 0.0, # Radiation coefficient for snow (only used for RAD-TI model option)
                      "rad_coeff_ice": 0.0, # Radiation coefficient for ice (only used for RAD-TI model option)
                      "melt_factor": 3.5, # Melt factor (not used)
                      "melt_factor_snow": degree_day_factor_snow, # Melt factor for snow (mm w.e. degC d-1)
                      "melt_factor_ice": (degree_day_factor_snow/0.7), # Melt factor for ice (mm w.e. degC d-1)
                      "storage_coeff_ice": 0.72, # Storage coefficient for ice (for runoff simulations)
                      "storage_coeff_snow": 0.19, # Storage coefficient for snow (for runoff simulations)
                      "storage_coeff_firn": 0.66, # Storage coefficient for firn (for runoff simulations)
                      "prec_corr_factor": p_corr,#1.317,#p_corr, # Global precipitation correction [-]
                      "prec_lapse_rate": 0.1, # Precipitation lapse rate [100m-1] (positive upwards)
                      "temp_bias_corr": temp_corr, # Correction for temperature bias [C]
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
                          "filename_dem": 'dem_' + config_model['simulation_name'] + '.nc', # Filename of 1 km DEM
                          "filename_high_res_dem": 'dem_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m DEM for downscaling
                          "filepath_ice_thickness": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/ice_thickness/', # Filepath to store dataset of ice thickness and bedrock topo
                          "filename_ice_thickness": 'ice_thickness_' + config_model['simulation_name'] + '_100m.nc', # Filename  of 100 m ice thickness maps
                          "filepath_fractions": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/fractions/', # Filepath to datasets of initial glacier and catchment fractions
                          "filename_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '.nc', # Filename of 1 km glacier masks
                          "filename_high_res_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m glacier masks
                          "filename_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '.nc', # Filename of 1 km catchment masks
                          "filename_high_res_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m catchment masks
                          "filepath_parameters": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/parameters_point/', # Filepath to store parameters from calibration
                          "filepath_results": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/results/', # Path to store results
                          "filepath_temp_prec_raw": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/temp_prec_data/', # Path to files with local temp and precip data
                          "filepath_climate_data": main_dir + 'climate_data/', # Filepath to store/retreive climate data files 
                          "filename_climate_data": 'job_all_seNorge_2018_21.09_ref_smb', # Name of climate data files for Jostedalsbreen, seNorge 2018 vs. 21_09
                          #"filename_climate_data": 'job_ref_smb_ECEARTH_CCLM',
                          "filepath_obs": main_dir + 'observations/', # Filepath to files with observations of mass balance and discharge
                          "filepath_dem_base": main_dir + 'dem_base/', # Filepath to 100 m base DEM.
                          "filename_dem_base": 'DEM100_JOB_EPSG32633.tif', # Base DEM for whole of Jostedalsbreen, for creating local cropped DEM
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

        # If Austdalsbreen (BREID 2478) is in the list of glacier IDs, add the
        # calculated calving loss (NVE) to the modelled summer and annual balance. The calving
        # loss is included in the glaciological summer and annual balance observations by NVE. 
        # N! This only works for single-glacier cases with the way that it is
        # added to arrays of summer and annual balance. 
        if 2478 in id_list:

            # Get DataFrame with calving calculations.
            df_calving_obs = get_glacier_obs(dir_file_names, 'calving') 
            
            # Get calving calculations for the given period. 
            calving_obs_sub = df_calving_obs.loc[idx[yr_list,id_list],:]

            # Get vector of calving volumes.        
            calving_obs_vol = calving_obs_sub['Bcalv'].to_numpy(copy=True) # Calving volume [10e6 m3]

            # Get vector of glacier area used in mass balance observations.
            area = calving_obs_sub['Area'].to_numpy(copy=True) # Area [km2]

            # Get specific calving balance in m w.e.
            calving_spec_balance = calving_obs_vol / area
            
            # For posterior predictive
            if config_model['run_posterior_predictive'] == True:
                calving_spec_balance = np.concatenate([np.zeros(26), calving_spec_balance])

            # Add calculated calving balance (loss) to glacier wide summer and annual balance as 
            # this is not modelled.
            mb_mod_summer = mb_mod_summer + calving_spec_balance
            mb_mod_annual = mb_mod_annual + calving_spec_balance

        # Every other year of simulation used for calibration
        if config_model['run_posterior_predictive'] == False:
            
            mb_mod_winter_cropped = mb_mod_winter[::2].copy()
            mb_mod_summer_cropped = mb_mod_summer[::2].copy()
            mb_mod_annual_cropped = mb_mod_annual[::2].copy()

            return mb_mod_winter_cropped, mb_mod_summer_cropped, mb_mod_annual_cropped
        
        else:

            return mb_mod_winter, mb_mod_summer, mb_mod_annual

        # Calculate 10-year mass balance rates.
        #mb_mod_10yr_rates = np.array([mb_mod_annual[0:10].mean(), mb_mod_annual[10:20].mean()])

        # Return modelled winter, summer, annual and 10-yr rates of mass blance.
        #return mb_mod_winter, mb_mod_summer, mb_mod_annual#, mb_mod_10yr_rates

#%% TUNSBERGDALSBREEN
# Define black-box model with Theano decorator.
    
    # Set up a shell function that takes parameters to be calibrated as input.
    # In the shell, these parameters are added to the "parameters" dict and
    # the model is run as normal. 

    @as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar], otypes=[tt.dvector, tt.dvector, tt.dvector])#, tt.dvector])#, tt.dvector])#, tt.dvector, tt.dvector, tt.dvector, tt.dvector, tt.dvector])
    #def shell_tun(p_corr, degree_day_factor_snow, degree_day_factor_ice):
    def shell_tun(p_corr, degree_day_factor_snow, temp_corr):

        main_dir = set_dirs('server')

        map_yrs = []

        # Configure model.
        config_model = {"simulation_name": 'Tunsbergdalsbreen', # Name of simulation case (str)
                        "filepath_simulation_files": main_dir + 'simulation_data/', # Path to simulation files (str)
                        "model_type": 'cl_ti', # Type of melt model, choose 'cl_ti' for classical temperature-index and 'rad-ti' for temperature-index with radiation term. Default is 'cl-ti'. 
                        "simulation_start_date": '1957-01-01', # Start date (str)
                        "simulation_end_date": '1972-12-31', # End date (str)
                        #"simulation_start_date": '1957-01-01', # Reverse years
                        #"simulation_end_date": '1971-12-31', # Reverse years
                        "rcp": 'hist',
                        "ref_mb": False,
                        "buffer": 1000,#6000, # buffer around glacier outline [m]
                        "multi": False, # only used if 'update_area_from_outline' is True and 'downscaling=True'.
                        "use_kss_dem": False,
                        "use_seNorge_dem": True, # Use seNorge DEM for simulations.
                        "climate_model": 'seNorge', #'kss'
                        "update_area_from_outline": True, # If True, update area from set of outlines. If False, no area update.
                        "update_area_type": 'auto', # only used if 'update_area_from_outline' is True. Automatic update ('auto') or manual/special case ('manual')
                        "map_years": map_yrs, # only used if 'update_area_from_outline' is True. Use if 'update_area_type' = 'manual'
                        "geometry_change": False, # Option to run model with (True) or without (False) geometry changes from delta-h parameterization.
                        "geometry_change_w_downscaling": False, # Option to run model with (True) or without (False) geometry changes from delta-h parameterization with downscaling.
                        "get_catchment_discharge": False, # Option to return discharge from a given catchment (bool).
                        "downscale_mb": False, # Option to downscale mb from seNorge grid to higher resolution DEM.
                        "get_internal_ablation": False, # Option to return internal ablation based on Oerlemans (2013).
                        "calculate_runoff": False, # Option to compute glacier runoff.
                        "calculate_discharge": False, # Option to compute discharge from catchments.
                        "calibration_start_date": '1966-01-01', # Start date for calibration period (str)
                        "calibration_end_date": '1972-12-31', # End date for calibration period (str)
                        #"calibration_start_date": '1967-01-01', # Reverse years
                        #"calibration_end_date": '1971-12-31', # Reverse years
                        "calibration_data": 'point', # For running MC method of Engelhardt et al. (2014). Choose gw (glacier-wide) or point (point) mass-balance for calibration (str).
                        "number_of_runs": 10, # For running MC method of Engelhardt et al. (2014). Number of calibration runs (int).
                        "observation_type": 'glac_gw_seasonal', # For running MCMC with PyMC: 'glac_gw_seasonal', 'glac_gw_annual', 'glac_gw_annual_10yr' or 'geodetic' (str). 
                        "run_posterior_predictive": True} # Set to True if running posterior predictive. Set to False otherwise.

        # Get monthly temperature lapse rates from file. 
        with open(main_dir + 'lapse_rates/temp_lapse_rates.txt', 'rb') as fp:
            temp_m_lr = pickle.load(fp)
    
        # Parameters for mass balance and discharge simulations.
        parameters = {"threshold_temp_snow" : 1.0, # Threshold temperature for snow [deg C]
                      "threshold_temp_melt" : 0.0, # Threshold temperature for melt [deg C]
                      "rad_coeff_snow": 0.0, # Radiation coefficient for snow (only used for RAD-TI model option)
                      "rad_coeff_ice": 0.0, # Radiation coefficient for ice (only used for RAD-TI model option)
                      "melt_factor": 3.5, # Melt factor (not used)
                      "melt_factor_snow": degree_day_factor_snow, # Melt factor for snow (mm w.e. degC d-1)
                      "melt_factor_ice": (degree_day_factor_snow/0.7), # Melt factor for ice (mm w.e. degC d-1)
                      "storage_coeff_ice": 0.72, # Storage coefficient for ice (for runoff simulations)
                      "storage_coeff_snow": 0.19, # Storage coefficient for snow (for runoff simulations)
                      "storage_coeff_firn": 0.66, # Storage coefficient for firn (for runoff simulations)
                      "prec_corr_factor": p_corr,#1.317,#p_corr, # Global precipitation correction [-]
                      "prec_lapse_rate": 0.1, # Precipitation lapse rate [100m-1] (positive upwards)
                      "temp_bias_corr": temp_corr, # Correction for temperature bias [C]
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
                          "filename_dem": 'dem_' + config_model['simulation_name'] + '.nc', # Filename of 1 km DEM
                          "filename_high_res_dem": 'dem_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m DEM for downscaling
                          "filepath_ice_thickness": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/ice_thickness/', # Filepath to store dataset of ice thickness and bedrock topo
                          "filename_ice_thickness": 'ice_thickness_' + config_model['simulation_name'] + '_100m.nc', # Filename  of 100 m ice thickness maps
                          "filepath_fractions": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/fractions/', # Filepath to datasets of initial glacier and catchment fractions
                          "filename_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '.nc', # Filename of 1 km glacier masks
                          "filename_high_res_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m glacier masks
                          "filename_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '.nc', # Filename of 1 km catchment masks
                          "filename_high_res_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m catchment masks
                          "filepath_parameters": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/parameters_point/', # Filepath to store parameters from calibration
                          "filepath_results": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/results/', # Path to store results
                          "filepath_temp_prec_raw": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/temp_prec_data/', # Path to files with local temp and precip data
                          "filepath_climate_data": main_dir + 'climate_data/', # Filepath to store/retreive climate data files 
                          "filename_climate_data": 'job_all_seNorge_2018_21.09_ref_smb', # Name of climate data files for Jostedalsbreen, seNorge 2018 vs. 21_09
                          #"filename_climate_data": 'job_ref_smb_ECEARTH_CCLM',
                          "filepath_obs": main_dir + 'observations/', # Filepath to files with observations of mass balance and discharge
                          "filepath_dem_base": main_dir + 'dem_base/', # Filepath to 100 m base DEM.
                          "filename_dem_base": 'DEM100_JOB_EPSG32633.tif', # Base DEM for whole of Jostedalsbreen, for creating local cropped DEM
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

        # If Austdalsbreen (BREID 2478) is in the list of glacier IDs, add the
        # calculated calving loss (NVE) to the modelled summer and annual balance. The calving
        # loss is included in the glaciological summer and annual balance observations by NVE. 
        # N! This only works for single-glacier cases with the way that it is
        # added to arrays of summer and annual balance. 
        if 2478 in id_list:

            # Get DataFrame with calving calculations.
            df_calving_obs = get_glacier_obs(dir_file_names, 'calving') 
            
            # Get calving calculations for the given period. 
            calving_obs_sub = df_calving_obs.loc[idx[yr_list,id_list],:]

            # Get vector of calving volumes.        
            calving_obs_vol = calving_obs_sub['Bcalv'].to_numpy(copy=True) # Calving volume [10e6 m3]

            # Get vector of glacier area used in mass balance observations.
            area = calving_obs_sub['Area'].to_numpy(copy=True) # Area [km2]

            # Get specific calving balance in m w.e.
            calving_spec_balance = calving_obs_vol / area
            
            # For posterior predictive
            if config_model['run_posterior_predictive'] == True:
                calving_spec_balance = np.concatenate([np.zeros(26), calving_spec_balance])

            # Add calculated calving balance (loss) to glacier wide summer and annual balance as 
            # this is not modelled.
            mb_mod_summer = mb_mod_summer + calving_spec_balance
            mb_mod_annual = mb_mod_annual + calving_spec_balance

        # Every other year of simulation used for calibration
        if config_model['run_posterior_predictive'] == False:
            
            mb_mod_winter_cropped = mb_mod_winter[::2].copy()
            mb_mod_summer_cropped = mb_mod_summer[::2].copy()
            mb_mod_annual_cropped = mb_mod_annual[::2].copy()

            return mb_mod_winter_cropped, mb_mod_summer_cropped, mb_mod_annual_cropped
        
        else:

            return mb_mod_winter, mb_mod_summer, mb_mod_annual

        # Calculate 10-year mass balance rates.
        #mb_mod_10yr_rates = np.array([mb_mod_annual[0:10].mean(), mb_mod_annual[10:20].mean()])

        # Return modelled winter, summer, annual and 10-yr rates of mass blance.
        #return mb_mod_winter, mb_mod_summer, mb_mod_annual#, mb_mod_10yr_rates        

#%% SUPPHELLEBREEN
# Define black-box model with Theano decorator.
    
    # Set up a shell function that takes parameters to be calibrated as input.
    # In the shell, these parameters are added to the "parameters" dict and
    # the model is run as normal. 

    @as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar], otypes=[tt.dvector, tt.dvector, tt.dvector])#, tt.dvector])#, tt.dvector])#, tt.dvector, tt.dvector, tt.dvector, tt.dvector, tt.dvector])
    def shell_sup(p_corr, degree_day_factor_snow, temp_corr):
    #def shell_sup(p_corr, degree_day_factor_snow, degree_day_factor_ice):


        main_dir = set_dirs('server')

        map_yrs = []

        # Configure model.
        config_model = {"simulation_name": 'Supphellebreen', # Name of simulation case (str)
                        "filepath_simulation_files": main_dir + 'simulation_data/', # Path to simulation files (str)
                        "model_type": 'cl_ti', # Type of melt model, choose 'cl_ti' for classical temperature-index and 'rad-ti' for temperature-index with radiation term. Default is 'cl-ti'. 
                        "simulation_start_date": '1957-01-01', # Start date (str)
                        "simulation_end_date": '1982-12-31', # Posterior predictive
                        #"simulation_start_date": '1957-01-01', # Reverse years
                        #"simulation_end_date": '1967-12-31', # Reverse years
                        #"simulation_end_date": '1966-12-31', # End date (str)
                        "rcp": 'hist',
                        "ref_mb": False,
                        "buffer": 1000,#6000, # buffer around glacier outline [m]
                        "multi": False, # only used if 'update_area_from_outline' is True and 'downscaling=True'.
                        "use_kss_dem": False,
                        "use_seNorge_dem": True, # Use seNorge DEM for simulations.
                        "climate_model": 'seNorge',#'kss',
                        "update_area_from_outline": True, # If True, update area from set of outlines. If False, no area update.
                        "update_area_type": 'auto', # only used if 'update_area_from_outline' is True. Automatic update ('auto') or manual/special case ('manual')
                        "map_years": map_yrs, # only used if 'update_area_from_outline' is True. Use if 'update_area_type' = 'manual'
                        "geometry_change": False, # Option to run model with (True) or without (False) geometry changes from delta-h parameterization.
                        "geometry_change_w_downscaling": False, # Option to run model with (True) or without (False) geometry changes from delta-h parameterization with downscaling.
                        "get_catchment_discharge": False, # Option to return discharge from a given catchment (bool).
                        "downscale_mb": False, # Option to downscale mb from seNorge grid to higher resolution DEM.
                        "get_internal_ablation": False, # Option to return internal ablation based on Oerlemans (2013).
                        "calculate_runoff": False, # Option to compute glacier runoff.
                        "calculate_discharge": False, # Option to compute discharge from catchments.
                        "calibration_start_date": '1964-01-01', # Start date for calibration period (str)
                        "calibration_end_date": '1982-12-31', # Posterior predictive
                        #"calibration_start_date": '1965-01-01', # Reverse years
                        #"calibration_end_date": '1967-12-31', # Reverse years
                        #"calibration_end_date": '1966-12-31', # End date for calibration period (str)
                        "calibration_data": 'point', # For running MC method of Engelhardt et al. (2014). Choose gw (glacier-wide) or point (point) mass-balance for calibration (str).
                        "number_of_runs": 10, # For running MC method of Engelhardt et al. (2014). Number of calibration runs (int).
                        "observation_type": 'glac_gw_seasonal', # For running MCMC with PyMC: 'glac_gw_seasonal', 'glac_gw_annual', 'glac_gw_annual_10yr' or 'geodetic' (str). 
                        "run_posterior_predictive": True} # Set to True if running posterior predictive. Set to False otherwise.

        # Get monthly temperature lapse rates from file. 
        with open(main_dir + 'lapse_rates/temp_lapse_rates.txt', 'rb') as fp:
            temp_m_lr = pickle.load(fp)
    
        # Parameters for mass balance and discharge simulations.
        parameters = {"threshold_temp_snow" : 1.0, # Threshold temperature for snow [deg C]
                      "threshold_temp_melt" : 0.0, # Threshold temperature for melt [deg C]
                      "rad_coeff_snow": 0.0, # Radiation coefficient for snow (only used for RAD-TI model option)
                      "rad_coeff_ice": 0.0, # Radiation coefficient for ice (only used for RAD-TI model option)
                      "melt_factor": 3.5, # Melt factor (not used)
                      "melt_factor_snow": degree_day_factor_snow, # Melt factor for snow (mm w.e. degC d-1)
                      "melt_factor_ice": (degree_day_factor_snow/0.7), # Melt factor for ice (mm w.e. degC d-1)
                      "storage_coeff_ice": 0.72, # Storage coefficient for ice (for runoff simulations)
                      "storage_coeff_snow": 0.19, # Storage coefficient for snow (for runoff simulations)
                      "storage_coeff_firn": 0.66, # Storage coefficient for firn (for runoff simulations)
                      "prec_corr_factor": p_corr,#1.317,#p_corr, # Global precipitation correction [-]
                      "prec_lapse_rate": 0.1, # Precipitation lapse rate [100m-1] (positive upwards)
                      "temp_bias_corr": temp_corr, # Correction for temperature bias [C]
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
                          "filename_dem": 'dem_' + config_model['simulation_name'] + '.nc', # Filename of 1 km DEM
                          "filename_high_res_dem": 'dem_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m DEM for downscaling
                          "filepath_ice_thickness": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/ice_thickness/', # Filepath to store dataset of ice thickness and bedrock topo
                          "filename_ice_thickness": 'ice_thickness_' + config_model['simulation_name'] + '_100m.nc', # Filename  of 100 m ice thickness maps
                          "filepath_fractions": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/fractions/', # Filepath to datasets of initial glacier and catchment fractions
                          "filename_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '.nc', # Filename of 1 km glacier masks
                          "filename_high_res_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m glacier masks
                          "filename_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '.nc', # Filename of 1 km catchment masks
                          "filename_high_res_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '_100m.nc', # Filename of 100 m catchment masks
                          "filepath_parameters": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/parameters_point/', # Filepath to store parameters from calibration
                          "filepath_results": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/results/', # Path to store results
                          "filepath_temp_prec_raw": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/temp_prec_data/', # Path to files with local temp and precip data
                          "filepath_climate_data": main_dir + 'climate_data/', # Filepath to store/retreive climate data files 
                          "filename_climate_data": 'job_all_seNorge_2018_21.09_ref_smb', # Name of climate data files for Jostedalsbreen, seNorge 2018 vs. 21_09
                          #"filename_climate_data": 'job_ref_smb_ECEARTH_CCLM',
                          "filepath_obs": main_dir + 'observations/', # Filepath to files with observations of mass balance and discharge
                          "filepath_dem_base": main_dir + 'dem_base/', # Filepath to 100 m base DEM.
                          "filename_dem_base": 'DEM100_JOB_EPSG32633.tif', # Base DEM for whole of Jostedalsbreen, for creating local cropped DEM
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

        # If Austdalsbreen (BREID 2478) is in the list of glacier IDs, add the
        # calculated calving loss (NVE) to the modelled summer and annual balance. The calving
        # loss is included in the glaciological summer and annual balance observations by NVE. 
        # N! This only works for single-glacier cases with the way that it is
        # added to arrays of summer and annual balance. 
        if 2478 in id_list:

            # Get DataFrame with calving calculations.
            df_calving_obs = get_glacier_obs(dir_file_names, 'calving') 
            
            # Get calving calculations for the given period. 
            calving_obs_sub = df_calving_obs.loc[idx[yr_list,id_list],:]

            # Get vector of calving volumes.        
            calving_obs_vol = calving_obs_sub['Bcalv'].to_numpy(copy=True) # Calving volume [10e6 m3]

            # Get vector of glacier area used in mass balance observations.
            area = calving_obs_sub['Area'].to_numpy(copy=True) # Area [km2]

            # Get specific calving balance in m w.e.
            calving_spec_balance = calving_obs_vol / area
            
            # For posterior predictive
            if config_model['run_posterior_predictive'] == True:
                calving_spec_balance = np.concatenate([np.zeros(26), calving_spec_balance])

            # Add calculated calving balance (loss) to glacier wide summer and annual balance as 
            # this is not modelled.
            mb_mod_summer = mb_mod_summer + calving_spec_balance
            mb_mod_annual = mb_mod_annual + calving_spec_balance

        # Every other year of simulation used for calibration
        if config_model['run_posterior_predictive'] == False:
            
            mb_mod_winter_cropped = mb_mod_winter[::2].copy()
            mb_mod_summer_cropped = mb_mod_summer[::2].copy()
            mb_mod_annual_cropped = mb_mod_annual[::2].copy()

            return mb_mod_winter_cropped, mb_mod_summer_cropped, mb_mod_annual_cropped
        
        else:
            return mb_mod_winter, mb_mod_summer, mb_mod_annual

        # Calculate 10-year mass balance rates.
        #mb_mod_10yr_rates = np.array([mb_mod_annual[0:10].mean(), mb_mod_annual[10:20].mean()])

        # Return modelled winter, summer, annual and 10-yr rates of mass blance.
        #return mb_mod_winter, mb_mod_summer, mb_mod_annual#, mb_mod_10yr_rates        

#%% Define PyMC model.

    # Declare model context.
    mb_model = pm.Model()
    
    # Set up model context.
    with mb_model:
        
        # Observations and observation uncertainty
        sigma_s_all_mean = sigma_s_all.mean()
        sigma_w_all_mean = sigma_w_all.mean()

        # Set priors for unknown model parameters.
        DDF_snow = pm.TruncatedNormal("DDF_snow", mu=4.1, sigma=1.0, lower=0) # 95% confidence interval is approximately 2.1 and 6.1 mm w.e.C-1 d-1
        prec_corr = pm.TruncatedNormal("prec_corr", mu=1.0, sigma=0.25, lower=0) # 95% confidence interval is approximately 0.5 to 1.5
        T_corr = pm.Normal("T_corr", mu=0.0, sigma=0.5) # 95% confidence interval is approximately +/- 1 degC
        sigma_mod = pm.HalfNormal("sigma_mod", sigma=0.67) # sigma is in this case a scale parameter such that mean = scale*sqrt(2)/sqrt(pi) for the half-normal distribution
        # with scale = 0.67, the mean is about 0.535 and 95% of the distribution is within 1.5 m w.e.
        # with scale = 0.45, the mean is about 0.36 and 95% of the distribution is within 1.0 m w.e. 

        # W/sigma_mod
        sigma_tot_w = tt.sqrt(sigma_w_all_mean**2 + sigma_mod**2)
        sigma_tot_s = tt.sqrt(sigma_s_all_mean**2 + sigma_mod**2)
        
        # W/o sigma_mod
        ##sigma_tot_a = sigma_a + sigma_mod
        #sigma_tot_w = sigma_w_all_mean
        #sigma_tot_s = sigma_s_all_mean
    
        # Expected value of outcome is the modelled mass balance.
        mb_mod_w_nig, mb_mod_s_nig, mb_mod_a_nig = shell_nig(prec_corr, DDF_snow, T_corr)
        mb_mod_w_aus, mb_mod_s_aus, mb_mod_a_aus = shell_aus(prec_corr, DDF_snow, T_corr)
        mb_mod_w_ves, mb_mod_s_ves, mb_mod_a_ves = shell_ves(prec_corr, DDF_snow, T_corr)
        mb_mod_w_tun, mb_mod_s_tun, mb_mod_a_tun = shell_tun(prec_corr, DDF_snow, T_corr)
        mb_mod_w_sup, mb_mod_s_sup, mb_mod_a_sup = shell_sup(prec_corr, DDF_snow, T_corr)

        mb_mod_w = tt.concatenate((mb_mod_w_nig, mb_mod_w_aus, mb_mod_w_ves, mb_mod_w_tun, mb_mod_w_sup))
        mb_mod_s = tt.concatenate((mb_mod_s_nig, mb_mod_s_aus, mb_mod_s_ves, mb_mod_s_tun, mb_mod_s_sup))

        # Observations.
        data_mb_w = pm.Data('data_mb_w', mb_obs_w_all) # winter glaciological
        data_mb_s = pm.Data('data_mb_s', mb_obs_s_all) # summer glaciological
        #data_mb_a = pm.Data('data_mb_a', mb_obs_a) # annual glaciological

        # Expected value as deterministic RVs. Saves these to inference data file.
        mu_mb_w = pm.Deterministic('mu_mb_w', mb_mod_w) # winter
        mu_mb_s = pm.Deterministic('mu_mb_s', mb_mod_s) # summer
        #mu_mb_a = pm.Deterministic('mu_mb_a', mb_mod_a) # annual
        
        # Likelihood (sampling distribution) of observations (wo sigma_mod)
        #mb_obs_winter = pm.Normal("mb_obs_winter", mu=mu_mb_w, sigma=sigma_w_all_mean, observed=data_mb_w)
        #mb_obs_summer = pm.Normal("mb_obs_summer", mu=mu_mb_s, sigma=sigma_s_all_mean, observed=data_mb_s)
        #mb_obs_annual = pm.Normal("mb_obs_annual", mu=mu_mb_a, sigma=sigma_a, observed=data_mb_a)

        # Likelihood with observation and model error (w sigma_mod)
        mb_obs_winter = pm.Normal("mb_obs_winter", mu=mu_mb_w, sigma=sigma_tot_w, observed=data_mb_w)
        mb_obs_summer = pm.Normal("mb_obs_summer", mu=mu_mb_s, sigma=sigma_tot_s, observed=data_mb_s)
        #mb_obs_annual = pm.Normal("mb_obs_annual", mu=mu_mb_a, sigma=sigma_tot_a, observed=data_mb_a)

    # Sample posterior.
    #with mb_model:
        
        # Choose sampler.
        #step = pm.Metropolis()
    #    step = pm.DEMetropolisZ()

        # Draw posterior samples.
    #    idata_post = pm.sample(draws=10000, tune=2000, step=step, return_inferencedata=True, chains=4, cores=20, progressbar=True, idata_kwargs=dict(log_likelihood=False))
        
    # Save InferenceData with posteriors to netcdf file.
    #idata_post.to_netcdf(main_dir + 'simulation_data/JOB_ref_SMB/results/idata_DEMZ_t2000_s10000_c4_gwsw_wo_sigma_NEW_prior_NEW_scale0.67_ts1.nc')

    # Sample prior.
    #with mb_model:

    #     # Sample prior for given variables.
    #    start_time = time.time()
    #    prior = pm.sample_prior_predictive(5000)
    #    print("Prior predictive sampling took " + str (time.time() - start_time) + ' seconds.')

         # Save prior data in InferenceData object.
    #    idata_prior = az.from_pymc3(prior=prior)

    # # Save InferenceData with prior to netcdf file.
    #idata_prior.to_netcdf(main_dir + 'simulation_data/JOB_ref_SMB/results/idata_DEMZ_s5000_gwsw_w_sigma_NEW_prior_scale0.67_ts1.nc')

    # Sample posterior predictive.
    with mb_model:

        # Get inference data.
        idata_sampled = az.from_netcdf(main_dir + 'simulation_data/JOB_ref_SMB/results/idata_DEMZ_t2000_s10000_c4_gwsw_wo_sigma_NEW_prior_NEW_scale0.67_ts1.nc')

        print("Starting posterior predictive sampling.")
         # Sample posterior predictive from trace.
        start_time = time.time()
        fast_post_pred = pm.fast_sample_posterior_predictive(trace=idata_sampled, samples=5000)
        print("Posterior predictive sampling took " + str(time.time() - start_time) + ' seconds.')

        # Save posterior predicitve data in InferenceData object.
        idata_post_pred = az.from_pymc3(posterior_predictive=fast_post_pred)

    # # Save InferenceData with posterior predictive to netcdf file.
    idata_post_pred.to_netcdf(main_dir + 'simulation_data/JOB_ref_SMB/results/idata_DEMZ_t2000_s10000_c4_gwsw_wo_sigma_NEW_prior_NEW_scale0.67_ts1_post_pred.nc')

if __name__ == '__main__':
    main()
