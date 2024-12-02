# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:28:56 2020

@author: kasj

-------------------------------------------------------------------
Mass-balance model
-------------------------------------------------------------------

Script to set up and run model simulations.

Function initialize_input() must be run first to set up DEMs and masks.

"""
    
#%% Libraries %%#

# Standard libraries
import pickle

# External libraries
import xarray as xr

# Internal libraries
from set_dirs import set_dirs
from preprocessing import initialize_dem
from preprocessing import get_ice_thickness
from preprocessing import catchment_specific_fraction
from mb_model import mass_balance
from parameter_optimization import optimize_MC

#%% Set directory

# Set main directory to local or server.
main_dir = set_dirs('server')

#%% Configure model

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

# Define model settings. 
# If 'get_catchment_discharge' is True, the model simulates the discharge from the catchment IDs given in the 'catchment_id' file. 
# If 'get_catchment_discharge' is False, the model does not use the outline of the catchment IDs given in the 'catchment_id' file,
# but instead run simulations based on a 'dummy' catchment created for the geographical area with a margin around the (merged) outline(s) 
# of the glacier IDs given in the 'glacier_id' file. 
config_model = {"simulation_name": 'Nigardsbreen',
                #"filepath_data": '/mirror/khsjursen/mb_model_ref_smb/',
                "filepath_simulation_files": main_dir + 'simulation_data/', # Filepath to simulation files. 
                "model_type": 'rad_ti', # Type of melt model, choose 'cl_ti' for classical temperature-index and 'rad-ti' for temperature-index with radiation term. Default is 'cl-ti'.
                "simulation_start_date": '1961-01-01', # Start date (str)
                "simulation_end_date": '2005-12-31', # End date (str)
                "rcp": "hist", # hist, rcp45 or rcp85
                "ref_mb": False,
                "buffer": 1000, #6000, # buffer around glacier outline [m]
                "multi": False, # only used if 'update_area_from_outline' is True and 'downscaling=True'.
                "use_kss_dem": False,
                "use_seNorge_dem": True,
                "climate_model": 'kss',
                "update_area_from_outline": True,
                "update_area_type": 'auto', # only used if 'update_area_from_outline' is True. Automatic update ('auto') or manual/special case ('manual')
                "map_years": map_yrs, # only used if 'update_area_from_outline' is True. Use if 'update_area_type' = 'manual'
                "geometry_change": False, # Option to run model with (True) or without (False) geometry changes
                "geometry_change_w_downscaling": False, 
                "get_catchment_discharge": False, # Option to return discharge from a given catchment (bool)
                "downscale_mb": False, # Option to downscale mb from seNorge grid to higher resolution DEM.
                "get_internal_ablation": False,
                "calculate_runoff": False,
                "calculate_discharge": False,
                "calibration_start_date": '1960-01-01', # Start date for calibration period (str)
                "calibration_end_date": '2020-12-31', # End date for calibration period (str)
                "calibration_data": 'gw', # choose gw (glacier-wide) or point (point) mass-balance for calibration
                } 

# For calibration with multiple glaciers:
glacier_tup = ({"simulation_name": 'Nigardsbreen',
                "simulation_start_date": '1957-01-01', # Start date (str)
                "simulation_end_date": '2020-12-31', # End date (str) 
                "calibration_start_date": '1962-01-01', # Start date for calibration period (str)
                "calibration_end_date": '2020-12-31', # End date for calibration period (str)
                    },
               {"simulation_name": 'Austdalsbreen',
                "simulation_start_date": '1957-01-01', # Start date (str)
                "simulation_end_date": '2020-12-31', # End date (str) 
                "calibration_start_date": '1988-01-01', # Start date for calibration period (str)
                "calibration_end_date": '2020-12-31', # End date for calibration period (str)
                    },
               {"simulation_name": 'Vesledalsbreen',
                "simulation_start_date": '1957-01-01', # Start date (str)
                "simulation_end_date": '1972-12-31', # End date (str) 
                "calibration_start_date": '1967-01-01', # Start date for calibration period (str)
                "calibration_end_date": '1972-12-31', # End date for calibration period (str)
                    },
               {"simulation_name": 'Tunsbergdalsbreen',
                "simulation_start_date": '1957-01-01', # Start date (str)
                "simulation_end_date": '1972-12-31', # End date (str) 
                "calibration_start_date": '1966-01-01', # Start date for calibration period (str)
                "calibration_end_date": '1972-12-31', # End date for calibration period (str)
                    },
               {"simulation_name": 'Supphellebreen',
                "simulation_start_date": '1957-01-01', # Start date (str)
                "simulation_end_date": '1982-12-31', # End date (str) 
                "calibration_start_date": '1964-01-01', # Start date for calibration period (str)
                "calibration_end_date": '1982-12-31', # End date for calibration period (str)
                    },
               )

#%% Declare parameters

# Get temperature lapse rates.
with open(main_dir + 'lapse_rates/temp_lapse_rates.txt', 'rb') as fp:
    temp_m_lr = pickle.load(fp)

# Parameters for mass balance and discharge simulations.
# If optimization == True, these parameter values are ignored and parameter
# values are generated in the optimization. 
parameters = {"threshold_temp_snow": 1.0,
              "threshold_temp_melt": 0.0,
              "rad_coeff_snow": 3.0,
              "rad_coeff_ice": 4.0,
              "melt_factor": 2.9,
              "melt_factor_snow": 3.9, 
              "melt_factor_ice": (3.9/0.7), 
              "storage_coeff_ice": 0.72,
              "storage_coeff_snow": 0.19,
              "storage_coeff_firn": 0.66,
              "prec_corr_factor": 1.0, 
              "prec_lapse_rate": 0.0, # [100m-1] (positive upwards)
              "temp_bias_corr": 0.0, # Correction for global temperature bias [C]
              "temp_w_bias_corr": 0.0, # Correction for winter temperature bias [C]
              "temp_lapse_rate": temp_m_lr, # [C 100m-1] (negative upwards)
              "density_water": 1000, # [kgm-3]
              "density_ice": 850} # [kgm-3]

#%% Declare filepaths

# Filepaths and filenames for files to be used in model run.
dir_file_names = {"filepath_glacier_id": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/glacier_id/',
                  #"filename_glacier_id": 'glacier_id_JOB.txt',
                  "filename_glacier_id": 'glacier_id.txt',
                  "filepath_catchment_id": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/catchment_id/',
                  #"filename_catchment_id": 'catchment_id_JOB.txt',
                  "filename_catchment_id": 'catchment_id.txt',
                  "filepath_dem": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/dem/', # Filepath to local DEM
                  "filename_dem": 'dem_' + config_model['simulation_name'] + '.nc',
                  "filename_high_res_dem": 'dem_' + config_model['simulation_name'] + '_100m.nc',
                  "filepath_ice_thickness": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/ice_thickness/', # Filepath to store dataset of ice thickness and bedrock topo
                  "filename_ice_thickness": 'ice_thickness_' + config_model['simulation_name'] + '_100m.nc',
                  "filepath_fractions": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/fractions/', # Filepath to datasets of initial glacier and catchment fractions
                  "filename_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '.nc',
                  "filename_high_res_gl_frac": 'glacier_spec_fraction_' + config_model['simulation_name'] + '_100m.nc',
                  "filename_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '.nc',
                  "filename_high_res_ca_frac": 'catchment_spec_fraction_' + config_model['simulation_name'] + '_100m.nc',
                  "filepath_parameters": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/parameters_point/', # Filepath to store parameters from calibration
                  "filepath_results": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/results/',
                  "filepath_temp_prec_raw": config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/temp_prec_data/',
                  "filepath_climate_data": main_dir + 'climate_data/', # Filepath to store/retreive climate data files 
                  "filename_climate_data": 'job_all_seNorge_2018_21.09',
                  "filepath_obs": main_dir + 'observations/', # Filepath to files with observations of mass balance and discharge
                  "filename_obs": 'massbalance_gw_JOB.csv',
                  "filepath_dem_base": main_dir + 'dem_base/',
                  "filename_dem_base": 'DEM100_JOB_EPSG32633.tif',
                  #"filename_dem_base": 'dem_KSS_1km.nc',
                  "filepath_shp": main_dir + 'shape_files/', # Filepath to shape files
                  "filename_shp_overview": 'shp_overview.csv',
                  "filename_shape_gl": 'cryoclim_GAO_NO_1999_2006/cryoclim_GAO_NO_1999_2006_UTM_33N.shp', # Filename of glacier shape file
                  #"filename_shape_gl": 'NVE_GAO_NO_2018_2019/GlacierAreaOutline_2018_2019_N_EPSG32633.shp',
                  #"filename_shape_gl_1966": 'BreF1966_basin_homog/BreF1966_basin_homog.shp',
                  "filename_shape_ca": 'regine_enhet/Nedborfelt_RegineEnhet_1.shp'} # Filename of catchment shape file 

#%%
#config = config_model
#param = parameters
#name_dir_files = dir_file_names

#%% Function initialize_input()

def initialize_input(config: dict, param: dict, name_dir_files: dict):
    """
    Initialize model DEMs, glacier masks, etc. 

    """
    
    if config['use_seNorge_dem']==True:
        # Preprocessing of 1km seNorge DEM (cropping) based on catchment IDs and 
        # calculation of glacier fraction inside each cell based on glacier IDs. 
        ds_dem, da_gl_spec_frac = initialize_dem(config, name_dir_files, seNorge=True)

    elif config['use_kss_dem']==True:
        # Preprocessing of 1km seNorge DEM (cropping) based on catchment IDs and 
        # calculation of glacier fraction inside each cell based on glacier IDs. 
        ds_dem, da_gl_spec_frac = initialize_dem(config, name_dir_files, seNorge=False, kss=True, buffer=config['buffer'])
        # Get kss dem for Nigardsbreen, Supphellebreen, Austdalsbreen, Tunsbergdalsbreen, Vesledalsbreen:
        ##ds_dem, da_gl_spec_frac = initialize_dem(config, name_dir_files, seNorge=False, kss=True, buffer=config['buffer'], seNorge_bbox=True, Y=[6853500,6840500], X=[61500,68500])
    
    else:
        # Preprocessing of 1km seNorge DEM (cropping) based on catchment IDs and 
        # calculation of glacier fraction inside each cell based on glacier IDs.
        name_dir_files['filename_high_res_dem'] = name_dir_files['filename_dem']
        name_dir_files['filename_high_res_gl_frac'] = name_dir_files['filename_gl_frac']
        
        ds_dem, da_gl_spec_frac = initialize_dem(config, name_dir_files, seNorge=False)
        # Get hres dem for Nigardsbreen, Supphellebreen, Austdalsbreen, Tunsbergdalsbreen, Vesledalsbreen:
        ##ds_dem, da_gl_spec_frac = initialize_dem(config, name_dir_files, seNorge=False, kss=False, seNorge_bbox=True, Y=[6853500,6840500], X=[61500,68500])
        
    # Get DataArray of catchment fraction for each catchment ID.
    da_ca_spec_frac = catchment_specific_fraction(ds_dem, config, name_dir_files)
    # Get hres ca for Nigardsbreen, Supphellebreen, Austdalsbreen, Tunsbergdalsbreen, Vesledalsbreen:
    #da_ca_spec_frac = catchment_specific_fraction(ds_dem, config, name_dir_files, hres=True)

    # If downscaling or geometry update is True, get high-res DEM and masks. 
    if config['downscale_mb']==True or config['geometry_change_w_downscaling'] ==True:

        if ds_dem.res != 1000:
            print('Correct DEM?')
        
        Y_coor = ds_dem.Y.values
        X_coor = ds_dem.X.values

        # Preprocessing of high_res DEM (cropping) based on catchment IDs and 
        # calculation of glacier fraction inside each cell based on glacier IDs. 
        ds_dem_hr, da_gl_spec_frac_hr = initialize_dem(config, name_dir_files, seNorge=False, seNorge_bbox=True, Y=Y_coor, X=X_coor)

        # Get DataArray of high res catchment fraction for each catchment ID.
        da_ca_spec_frac_hr = catchment_specific_fraction(ds_dem_hr, config, name_dir_files, hres=True)

        if config['geometry_change_w_downscaling'] == True:
            
            # Set the year of thickness data.
            initialize_year = '1957'
            
            # Get ice thickness from Farinotti 2019 data.
            da_ice_thickness = get_ice_thickness(ds_dem_hr, da_gl_spec_frac_hr, initialize_year, 
                                                 config, name_dir_files)

# End of function initialize_input()

#%% Function run_model()

def run_model(config: dict, param: dict, name_dir_files: dict, 
             optimize = False, multi = False, glacier_list = None):
    
    """
    Function run_model() runs mass balance and discharge simulations, or 
    parameter optimization.
    
    If function argument optimize == True, model parameter optimization is 
    performed. Model parameter sets and model performance criteria are saved
    to file. If function argument optimize == False (default) a model
    simulation with the parameters specified in param is run.
    
    There are two choices for optimization:
        1) Optimization to glacier-wide seasonal balances and daily discharge.
        2) Optimization to point (stake) seasonal balances and daily
           discharge.

    Parameters
    ----------
    config : dict
        Dictionary of model configuration.
    param : dict
        Dictionary of model parameters.
    name_dir_files: dict
        Dictionary containing names of shape files and directory.
    optimize: bool (default False)
        Value determines if optimization of parameters is to be performed. The
        default value is False (no optimization). If optimize == True, 
        optimization of model parameters is performed. 

    Returns
    -------
    UPDATE HERE AND IN mb_model!
    mb_mod : Multiindex pandas.Dataframe 
        DataFrame of modeled winter, summer and annual mass balance for each
        glacier and year of simulation. Not returned if optimize == True.
    dis_mod : pandas.DataFrame
        DataFrame of modeled daily discharge for each glacier for each day of 
        simulation. Not returned if optimize == True.
    """
    
    # If model parameter optimization is chosen, call function optimize_mc()
    # and print when optimization is finished. Parameter sets are saved to 
    # file.
    if (optimize == True and multi == True):
        
        optimize_MC(config, param, name_dir_files, multi = True, name_list = glacier_list)
        print('Optimization finished.')
    
    elif (optimize == True and multi == False):
        
        optimize_MC(config, param, name_dir_files, multi = False, name_list = None)
        print('Optimization finished.')
        
    # If model simulation is chosen, run the model with the parameters
    # specified in param. 
    else:

        # Get DEM.
        with xr.open_dataset(name_dir_files['filepath_dem'] 
                             + name_dir_files['filename_dem']) as ds_dem_out:
            ds_dem = ds_dem_out
        
        # Get DataArray with glacier fraction for each glacier ID. 
        with xr.open_dataset(name_dir_files['filepath_fractions'] 
                             + name_dir_files['filename_gl_frac']) as ds_gl_spec_frac:
            da_gl_spec_frac = ds_gl_spec_frac.glacier_specific_fraction
        
        # Get DataArray of catchment fraction for each catchment ID.
        with xr.open_dataset(name_dir_files['filepath_fractions'] 
                             + name_dir_files['filename_ca_frac']) as ds_ca_spec_frac:
            da_ca_spec_frac = ds_ca_spec_frac.catchment_specific_fraction
        
    
        # Run mass balance model.

        mb_mod = mass_balance(ds_dem, da_gl_spec_frac, da_ca_spec_frac, config, param, name_dir_files)

        return mb_mod#, runoff

# End of function run_model()

#%%

# For running simulations:
#start_time = time.time()
#run_model(config_model, parameters, dir_file_names, optimize=True, multi=True, glacier_list = glacier_tup)
#print(time.time()-start_time)
#print(mb)

#filepath_results = config_model['filepath_simulation_files'] + config_model['simulation_name'] + '/results/'

# For running simulations:
#start_time = time.time()
#mb = run_model(config_model, parameters, dir_file_names, optimize=False)
#print(time.time()-start_time)
#print(mb)

#mb.to_csv(filepath_results + 'Nigardsbreen_mb_test.csv')


#%% End of run_model.py