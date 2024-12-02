# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:49:16 2020

@author: kasj

-------------------------------------------------------------------
Mass-balance model
-------------------------------------------------------------------

SCRIPT DESCRIPTION

"""

#%% Libraries

# Standard libraries
from shapely import geometry
from shapely.ops import cascaded_union
import os
import sys

# External libraries
import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.fill import fillnodata
from rasterio.warp import reproject, Resampling
from rasterio import Affine as A

# Internal libraries

#%% Function initialize_dem()

def initialize_dem(config: dict, name_dir_files: dict, seNorge = False, kss=False, seNorge_bbox = False, Y=None, X=None, buffer=None):
    
    """
    Function initializeDEM() creates a netcdf file of elevation and glacier
    fraction information necessary to run mass-balance calculations. If the 
    netcdf file already exists in the given filepath, the existing file is
    read and returned.
    
    There are two options for choice of base DEM based on whether the input
    parameter 'seNorge' is set to True or False (default).
    
    seNorge = True: The base DEM is the seNorge DEM (1x1km) retreived from
    the MET server (thredds.met.no). 
    seNorge = False: The base DEM is a geoTiff file indicated in the 
    input parameter dictionary name_dir_files.
        
    The function crops the base DEM based on the bounding box of 
    the combined shape of selected catchments to create a DEM of the area of
    interest. The total glacier fraction in each cell of the DEM is calculated
    from the combined shape of selected glaciers. The two arrays (elevation 
    and glacier fraction) are stored in a Dataset and saved as a local netCdf 
    file. The Dataset contains a yearly time coordinate so that the elvation 
    and glacier fraction arrays can be updated each year with new elevation/
    glacier fraction depending on delta_h changes as a result of mass changes.
    
    IMPORTANT! In the current version the elevation and glacier fraction
    variables are set to be constant in time! In the future this will be 
    changed so that only the elevation and glacier fraction variable will be
    populated for the first year (remaining years contain None).
    
    Parameters
    ----------
    config : dict
        Dictionary of model configuration.
    name_dir_files : dict
        Dictionary containing name of directories, shapefiles, 
        file with glacier IDs, file with catchment IDs, and information on 
        storage of netCdf file. 
    seNorge : bool
        Default is False. Indicates wether to use seNorge DEM as base (True) 
        or to get base DEM from geoTiff file indicated by filepath and
        filename specified in 'name_dir_files'.

    Returns
    -------
    ds_dem_out : xarray.Dataset
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
    """ 
    # Filepath of DEM.
    if (seNorge == False) & (kss == False): # Use high resolution DEM
        filepath_dem = name_dir_files['filepath_dem'] + name_dir_files['filename_high_res_dem']
        filepath_mask = name_dir_files['filepath_fractions'] + name_dir_files['filename_high_res_gl_frac']    
    elif (seNorge == False) & (kss == True): # Use low resolution DEM
        filepath_dem = name_dir_files['filepath_dem'] + name_dir_files['filename_dem']
        filepath_mask = name_dir_files['filepath_fractions'] + name_dir_files['filename_gl_frac']
    else: # Use low resolution DEM
        filepath_dem = name_dir_files['filepath_dem'] + name_dir_files['filename_dem']
        filepath_mask = name_dir_files['filepath_fractions'] + name_dir_files['filename_gl_frac']

    # Get option to compute discharge from catchment containing glaciers.
    get_ca_dis = config['get_catchment_discharge']
    ref_smb = config['ref_mb']

    # While the filename does not exist in filepath, initialize a DEM Dataset 
    # with name 'filename_dem'. Return the new or existing Dataset with the 
    # name 'filename_dem'. 
    while not os.path.isfile(filepath_dem):
        
        if config['update_area_from_outline'] == True:
            # Get start and end date.
            start_date= config['simulation_start_date']
            end_date = config['simulation_end_date']

            # Create time index from end of hydrological year (1.Oct) in start year
            # to end of hydrological year in end year.    
            time_range = pd.date_range(start_date, end_date, freq ='AS-OCT')
        
        # If using seNorge DEM from thredds.met.no.
        if seNorge == True:
            # Name of filepath of seNorge DEM on thredds.met.no.
            filepath_seNorge_dem = 'https://thredds.met.no/thredds/dodsC/senorge/geoinfo/seNorge2018_dem_UTM33.nc'

            # Open DEM as xarray Dataset.
            ds_dem = xr.open_dataset(filepath_seNorge_dem)

            # Rename coordinates of 'elevation' variable in Dataset to correspond to
            # climate data (seNorge) coordinates.
            ds_dem = ds_dem.rename({'easting':'X','northing':'Y'})
    
            # Cellsize is difference between coordinate values.
            res = ds_dem.X.item(1) - ds_dem.X.item(0) # Float, cellsize = 1000

            # If catchment discharge is to be computed, base DEM area on catchment outlines.
            if get_ca_dis == True:

                # Get polygon of combined catchment area.
                pgon_catchment_area = merge_catchments(name_dir_files)
    
                # Get bounding box coordinates of merged catchment outline.
                bounding_box = pgon_catchment_area.bounds # (minx, miny, maxx, maxy) tuple

                # Adjust coordinates of bounding box to DEM cell coordinates.
                bound_n = bounding_box[3]
                bound_n -= bound_n % -100 
                bound_n = bound_n + res * 1.5
                bound_s = bounding_box[1]
                bound_s -= bound_s % +100
                bound_s = bound_s - res * 1.5
                bound_w = bounding_box[0]
                bound_w -= bound_w % +100
                bound_w = bound_w - res * 1.5
                bound_e = bounding_box[2]
                bound_e -= bound_e % -100
                bound_e = bound_e + res * 1.5 
            
            # If catchment discharge is not to be computed, base DEM area on glacier outlines.
            elif get_ca_dis == False:

                # Get polygon of combined glacier area.
                pgon_glacier_area = merge_glaciers(name_dir_files)

                # Get bounding box coordinates of merged glacier outline.
                bounding_box = pgon_glacier_area.bounds # (minx, miny, maxx, maxy) tuple

                # Number of cells wanted in buffer around outline.
                cell_buffer = buffer/res

                # Adjust coordinates of bounding box to DEM cell coordinates.
                # If only the glacier outline is used to determine the area of
                # interest, the margin is larger (2.5*res instead of 1.5*res).
                bound_no = bounding_box[3]
                bound_n = (bound_no + res * cell_buffer) #+ 5500
                bound_n -= bound_n % +500 
                bound_n = (bound_n + res) if ((bound_n-bound_no) < res * cell_buffer) & (bound_n % 1000 != 0) else (bound_n + res/2) if ((bound_n-bound_no) < res * cell_buffer) & (bound_n % 1000 == 0) else bound_n
                bound_so = bounding_box[1]
                bound_s = (bound_so - res * cell_buffer)
                bound_s -= bound_s % +500
                bound_s = (bound_n - res) if ((bound_so-bound_s) < res * cell_buffer) & (bound_s % 1000 != 0) else (bound_s - res/2) if ((bound_s-bound_so) < res * cell_buffer) & (bound_s % 1000 == 0) else bound_s                
                bound_wo = bounding_box[0]
                bound_w = (bound_wo - res * cell_buffer)
                bound_w -= bound_w % +500
                bound_w = (bound_w - res) if ((bound_wo-bound_w) < res * cell_buffer) & (bound_w % 1000 != 0) else (bound_w - res/2) if ((bound_w-bound_wo) < res * cell_buffer) & (bound_w % 1000 == 0) else bound_w                
                bound_eo = bounding_box[2]
                bound_e = (bound_eo + res * cell_buffer) #+ 5500
                bound_e -= bound_e % +500 
                bound_e = (bound_e + res) if ((bound_e-bound_eo) < res * cell_buffer) & (bound_e % 1000 != 0) else (bound_e + res/2) if ((bound_e-bound_eo) < res * cell_buffer) & (bound_e % 1000 == 0) else bound_e

            else:
                sys.exit('Specify whether or not to compute discharge from catchment.') 
    
            # Create Y and X coordinates of array.
            Y_coor = ds_dem.Y.sel(Y=slice(bound_n, bound_s))
            X_coor = ds_dem.X.sel(X=slice(bound_w, bound_e))
            
            # Crop DEM by boundaries and convert to 2-D numpy array.
            elevation = np.array(ds_dem.elevation.sel(X=slice(bound_w, bound_e), 
                                                      Y=slice(bound_n, bound_s)).values)
        
            # Close dataset
            ds_dem.close() 

        # If using climate data and DEM from Klimaservice senter
        elif kss == True:
            
            # Name of filepath and geotiff DEM file. 
            filepath_kss_dem = name_dir_files['filepath_dem_base']
            filename_kss_dem = name_dir_files['filename_dem_base']

            # Open DEM as xarray Dataset.
            ds_dem = xr.open_dataset(filepath_kss_dem + filename_kss_dem, engine='netcdf4')
    
            # Cellsize is difference between coordinate values.
            res = ds_dem.X.item(1) - ds_dem.X.item(0) # Float, cellsize = 1000

            # If catchment discharge is to be computed, base DEM area on catchment outlines.
            if get_ca_dis == True:

                # Get polygon of combined catchment area.
                pgon_catchment_area = merge_catchments(name_dir_files)
    
                # Get bounding box coordinates of merged catchment outline.
                bounding_box = pgon_catchment_area.bounds # (minx, miny, maxx, maxy) tuple

                # Adjust coordinates of bounding box to DEM cell coordinates.
                bound_n = bounding_box[3]
                bound_n -= bound_n % -100 
                bound_n = bound_n + res * 1.5
                bound_s = bounding_box[1]
                bound_s -= bound_s % +100
                bound_s = bound_s - res * 1.5
                bound_w = bounding_box[0]
                bound_w -= bound_w % +100
                bound_w = bound_w - res * 1.5
                bound_e = bounding_box[2]
                bound_e -= bound_e % -100
                bound_e = bound_e + res * 1.5 
            
            # If catchment discharge is not to be computed, base DEM area on glacier outlines.
            elif get_ca_dis == False:

                # Get polygon of combined glacier area.
                pgon_glacier_area = merge_glaciers(name_dir_files)

                # Get bounding box coordinates of merged glacier outline.
                bounding_box = pgon_glacier_area.bounds # (minx, miny, maxx, maxy) tuple

                # Number of cells wanted in buffer around outline.
                cell_buffer = buffer/res

                # Adjust coordinates of bounding box to DEM cell coordinates.
                # If only the glacier outline is used to determine the area of
                # interest, the margin is larger (2.5*res instead of 1.5*res).
                if seNorge_bbox==True:
                    bound_n = Y[0]
                    bound_s = Y[-1]
                    bound_w = X[0]
                    bound_e = X[-1]
                else:
                    bound_no = bounding_box[3]
                    bound_n = (bound_no + res * cell_buffer) #+ 5500
                    bound_n -= bound_n % +500 
                    bound_n = (bound_n + res) if ((bound_n-bound_no) < res * cell_buffer) & (bound_n % 1000 != 0) else (bound_n + res/2) if ((bound_n-bound_no) < res * cell_buffer) & (bound_n % 1000 == 0) else bound_n
                    bound_so = bounding_box[1]
                    bound_s = (bound_so - res * cell_buffer)
                    bound_s -= bound_s % +500
                    bound_s = (bound_n - res) if ((bound_so-bound_s) < res * cell_buffer) & (bound_s % 1000 != 0) else (bound_s - res/2) if ((bound_s-bound_so) < res * cell_buffer) & (bound_s % 1000 == 0) else bound_s                
                    bound_wo = bounding_box[0]
                    bound_w = (bound_wo - res * cell_buffer)
                    bound_w -= bound_w % +500
                    bound_w = (bound_w - res) if ((bound_wo-bound_w) < res * cell_buffer) & (bound_w % 1000 != 0) else (bound_w - res/2) if ((bound_w-bound_wo) < res * cell_buffer) & (bound_w % 1000 == 0) else bound_w                
                    bound_eo = bounding_box[2]
                    bound_e = (bound_eo + res * cell_buffer) #+ 5500
                    bound_e -= bound_e % +500 
                    bound_e = (bound_e + res) if ((bound_e-bound_eo) < res * cell_buffer) & (bound_e % 1000 != 0) else (bound_e + res/2) if ((bound_e-bound_eo) < res * cell_buffer) & (bound_e % 1000 == 0) else bound_e

            else:
                sys.exit('Specify whether or not to compute discharge from catchment.') 
    
            # Create Y and X coordinates of array.
            Y_coor = ds_dem.Y.sel(Y=slice(bound_n, bound_s))
            X_coor = ds_dem.X.sel(X=slice(bound_w, bound_e))
            
            # Crop DEM by boundaries and convert to 2-D numpy array.
            elevation = np.array(ds_dem.elevation.sel(X=slice(bound_w, bound_e), 
                                                      Y=slice(bound_n, bound_s)).values)
        
            # Close dataset
            ds_dem.close() 
            

        # If using geotiff DEM from file. 
        else:
            # Name of filepath and geotiff DEM file. 
            filepath_geotiff = name_dir_files['filepath_dem_base']
            filename_geotiff = name_dir_files['filename_dem_base']
            
            # Open DEM as xarray DataArray.
            da_dem = xr.open_rasterio(filepath_geotiff + filename_geotiff)
        
            # Squeeze and drop 'band' coordinate such that dataArray
            # dimensions are (y,x).
            da_dem = da_dem.squeeze(dim='band', drop=True)
            
            # Rename coordinates of dataArray variables 'x' and 'y' to 
            # 'X' and 'Y' (capitalized).
            da_dem = da_dem.rename({'x':'X', 'y':'Y'})
            
            # Get cellsize of dataArray.
            res = da_dem.res[0]

            # If seNorge bounding box is to be used as bounding coordinates.
            if seNorge_bbox == True:
                
                sn_res = 1000

                bound_n = Y[0]-res/2+sn_res/2
                bound_s = Y[-1]+res/2-sn_res/2
                bound_w = X[0]+res/2-sn_res/2
                bound_e = X[-1]-res/2+sn_res/2
            
            # If catchment discharge is to be computed, base DEM area on catchment outlines.
            elif get_ca_dis == True:

                # Get polygon of combined catchment area.
                pgon_catchment_area = merge_catchments(name_dir_files)
    
                # Get bounding box coordinates of merged catchment outline.
                bounding_box = pgon_catchment_area.bounds # (minx, miny, maxx, maxy) 
                
                # Adjust coordinates of bounding box to DEM cell coordinates.
                bound_n = bounding_box[3]
                bound_n -= bound_n % -100 
                bound_n = bound_n + res * 1.5
                bound_s = bounding_box[1]
                bound_s -= bound_s % +100
                bound_s = bound_s - res * 1.5
                bound_w = bounding_box[0]
                bound_w -= bound_w % +100
                bound_w = bound_w - res * 1.5
                bound_e = bounding_box[2]
                bound_e -= bound_e % -100
                bound_e = bound_e + res * 1.5 
            
            # If catchment discharge is not to be computed, base DEM area on glacier outlines.
            elif get_ca_dis == False:

                # Get polygon of combined glacier area.
                pgon_glacier_area = merge_glaciers(name_dir_files)

                # Get bounding box coordinates of merged glacier outline.
                bounding_box = pgon_glacier_area.bounds # (minx, miny, maxx, maxy) tuple

                # Adjust coordinates of bounding box to DEM cell coordinates.
                # If only the glacier outline is used to determine the area of
                # interest, the margin is larger (2.5*res instead of 1.5*res).
                bound_n = bounding_box[3]
                bound_n -= bound_n % -100 
                bound_n = bound_n + res * 2.5
                #bound_n = round(bounding_box[3], -3) + res * 2.5
                bound_s = bounding_box[1]
                bound_s -= bound_s % +100
                bound_s = bound_s - res * 2.5
                #bound_s = round(bounding_box[1], -3) - res * 2.5
                bound_w = bounding_box[0]
                bound_w -= bound_w % +100
                bound_w = bound_w - res * 2.5
                #bound_w = round(bounding_box[0], -3) - res * 2.5
                bound_e = bounding_box[2]
                bound_e -= bound_e % -100
                bound_e = bound_e + res*2.5 
                #bound_e = round(bounding_box[2], -3) + res * 2.5 

            else:
                sys.exit('Specify whether or not to compute discharge from catchment.')           
    
            # Create Y and X coordinates of array.
            Y_coor = da_dem.Y.sel(Y=slice(bound_n, bound_s))
            X_coor = da_dem.X.sel(X=slice(bound_w, bound_e))
            
            # Crop DEM by boundaries and convert to 2-D numpy array.
            elevation = np.array(da_dem.sel(X=slice(bound_w, bound_e), 
                                            Y=slice(bound_n, bound_s)).values)
    
            # Close dataArray file. 
            da_dem.close()
            
            print('finished cropping DEM')
            
        # Initialize DataArray of elevation with coordinates time, Y, X.
        # Empty DataArray of elevation. By not setting the data variable the 
        # DataArray is created with nan values in the shape given by dims. 
        # When a DataArray of nan is created, the dtype is automatically set 
        # to np.float64. To avoid initializing the DataArray with dtype 
        # np.float64 (memory spike), the DataArray is initialized from an 
        # empty numpy array with zeros and dtype np.float32. Then the 
        # DataArray is filled with nans. 
        
        if config['update_area_from_outline'] == True:
            da_elevation = xr.DataArray(np.empty((len(time_range),
                                                  len(Y_coor),
                                                  len(X_coor)), dtype=np.float32),
                                        coords= {'time': time_range,
                                                 'Y': Y_coor,
                                                 'X': X_coor},
                                        dims =['time', 'Y', 'X'],
                                        attrs={'Name': 'elevation',
                                               'res': res})
        
            # Set values in DataArray to nan.
            da_elevation[:] = np.nan
        
            # Assign cropped DEM (elevation) as DataArray variable for a given time 
            # index.
            # SET ELEVATION FOR ALL YEARS:
            da_elevation.loc[time_range] = elevation
        
        else:
        
            da_elevation = xr.DataArray(np.empty((len(Y_coor),
                                                  len(X_coor)), dtype=np.float32),
                                       coords= {'Y': Y_coor,
                                             'X': X_coor},
                                       dims =['Y', 'X'],
                                       attrs={'Name': 'elevation',
                                              'res': res})
        
        
            # Assign cropped DEM (elevation) as DataArray variable for a given time 
            # index.
            # SET ELEVATION FOR ALL YEARS:
            da_elevation[:] = elevation
        
        # Convert DataArray to Dataset with cropped DEM (elevation) as a variable. 
        ds_dem_out = da_elevation.to_dataset(name = 'elevation')
    
        # Add cellsize attribute to Dataset.
        ds_dem_out.attrs['res'] = res
    
        # Add description attribute to Dataset.
        ds_dem_out.attrs['description'] = 'Dataset with elevation and glacier_fraction in area of interest.'
    
        print('start initializing fraction')
        
        if config['update_area_from_outline'] == True:
            
            # Get DataArray with transient glacier fractions for each glacier ID, 
            # based on overview of shape files for each glacier.
            da_glacier_specific_fraction = glacier_specific_fraction_from_outline(ds_dem_out, config, name_dir_files)

        elif ref_smb == True:
            
            # Create array of ones with size of the domain.
            glacier_array = np.ones((1,len(Y_coor),len(X_coor)), dtype=np.float32)
            
            # If resolution is higher than 1km (seNorge resolution), fill the number of 
            # cells that equals one 1km. If seNorge is used, fill only edges. 
            if (seNorge == False) & (kss == False):
                sn_res = 1000
                n = int(sn_res/res)#+ 1?
            else: 
                n = 1
            
            # Fill edges of array with zeros.
            glacier_array[0,0:n,:] = glacier_array[0,-n:,:] = glacier_array[0,:,0:n] = glacier_array[0,:,-n:] = 0
            
            da_glacier_specific_fraction = xr.DataArray(glacier_array,
                                                        coords= {'BREID': ['Dummy'],
                                                                  'Y': Y_coor,
                                                                  'X': X_coor},
                                                        dims =['BREID','Y', 'X'],
                                                        name='glacier_specific_fraction',
                                                        attrs={'Name': 'Glacier specific fraction',
                                                                'res': res})
            
        else:    
            
            # Get DataArray with glacier fraction for each glacier ID. 
            da_glacier_specific_fraction = glacier_specific_fraction(ds_dem_out, config, name_dir_files)
        
        # The glacier mask for the total glacierized area is the sum of the masks of all 
        # glacier IDs for each year.
        da_tot_gl_frac = np.sum(da_glacier_specific_fraction, axis=0)

        print('finished initializing fraction')
        # Add glacier_fraction as a variable to the DEM Dataset.
        #ds_dem_out['glacier_fraction'] = da_glacier_fraction
        ds_dem_out['glacier_fraction'] = da_tot_gl_frac

        # Save DEM Dataset as netcdf file.
        ds_dem_out.to_netcdf(filepath_dem)

        # Save Dataarray as netcdf file. Dataarray is always saved as Dataset
        # when saved to netcdf.
        da_glacier_specific_fraction.to_netcdf(filepath_mask)
    
        # Close files.      
        ds_dem_out.close()  
        da_glacier_specific_fraction.close()     
        
    # Get Dataset from filepath and return.  
    with xr.open_dataset(filepath_mask) as ds_gl_spec_frac:

        da_gl_spec_frac_out = ds_gl_spec_frac.glacier_specific_fraction
  
    with xr.open_dataset(filepath_dem) as ds_dem:

        ds_dem_out = ds_dem
    
    return(ds_dem_out, da_gl_spec_frac_out)    
    
# End of function initialize_dem()
    
#%% Function merge_catchments()

def merge_catchments(name_dir_files: dict):
    
    """
    Function merge_catchments() returns a polygon of the entire catchment area 
    based on catchment outlines of the selected catchment IDs. 

    Catchments to be merged are given in a text file with catchment IDs 
    (vassdragNr). Based on these IDs catchment polygons are selected from a 
    shape file containing information on all catchments (source: NVE). The 
    selected polygons are merged to one outline representing the entire 
    catchment area of interest.
    
    Parameters
    ----------
    name_dir_files : dict
        Dictionary containing directories, name of shapefile and file with 
        catchment IDs.

    Returns
    -------
    pgon_merged : shapely.geometry.polygon.Polygon
        Outline of entire catchment area.
    """
    
    #%% Read filepaths and filenames.
    
    # Name of shape file directory and shape file. 
    filedir_shp = name_dir_files['filepath_shp']
    shp_file = name_dir_files['filename_shape_ca']
    filedir_id = name_dir_files['filepath_catchment_id']
    id_file = name_dir_files['filename_catchment_id']

    # Get polygons to be merged based on catchment IDs and merge polygons.
    
    # Read shape file as dataframe.
    df_ca = gpd.read_file(filedir_shp + shp_file)
    
    # Read file with catchment IDs into dataframe.
    df_id = pd.read_csv(filedir_id + id_file)
    
    # List of catchment IDs.
    id_list = df_id['vassdragNr'].values.tolist()
    
    # Reduce dataframe to only include catchments with IDs in id_list.
    df_ca_cropped = df_ca[df_ca.vassdragNr.isin(id_list)]
    
    # List of polygons.
    pgon_list = df_ca_cropped['geometry'].to_list()
    
    # Merge polygons.
    pgon_merged = cascaded_union(pgon_list)
    
    # Return polygon. 
    return(pgon_merged)

# End of function merge_catchments()

#%% Function merge_glaciers()

def merge_glaciers(name_dir_files: dict):
    
    """
    Takes in glacier shape file from NVE, selects glacier outlines to be
    merged from text file with glacier IDs and merges glacier outlines to one
    outline representing the entire glacierized area.
    
    Function merge_glaciers() returns a polygon of the entire glacierized area 
    based on glacier outlines of the selected glacier IDs. 

    Glaciers to be merged are given in a text file with glacier IDs (BREID). 
    Based on these IDs, glacier polygons are selected from a shape file 
    containing information on all glaciers (source: NVE). The selected 
    polygons are merged to one outline representing the entire glacierized 
    area of interest.
    
    Parameters
    ----------
    name_dir_files : dict
        Dictionary containing directories, name of shape file and file with 
        glacier IDs.

    Returns
    -------
    pgon_merged : shapely.geometry.polygon.Polygon
        Outline of entire glacierized area.
    """
    
    # Read filepaths and filenames.
    
    # Name of shape file directory and shape file. 
    filedir_shp = name_dir_files['filepath_shp']
    shp_file = name_dir_files['filename_shape_gl']#'_1966']
    filedir_id = name_dir_files['filepath_glacier_id']
    id_file = name_dir_files['filename_glacier_id'] 

    # Get polygons to be merged based on glacier IDs and merge polygons.
    
    # Read shape file as dataframe.
    df_gl = gpd.read_file(filedir_shp + shp_file)  
    
    # Read file with glacier IDs into dataframe.
    df_id = pd.read_csv(filedir_id + id_file, sep=';')
    
    # List of glacier IDs.
    id_list = df_id['BREID'].values.tolist()
    
    # Reduce dataframe to only include glaciers with IDs in id_list.
    df_gl_cropped = df_gl[df_gl.BREID.isin(id_list)]
    
    # List of polygons.
    pgon_list = df_gl_cropped['geometry'].to_list()
    
    # Merge polygon. The function cascaded_union returns the union of the 
    # polygon objects in the function input list.
    pgon_merged = cascaded_union(pgon_list)
    
    # Return polygon.
    return(pgon_merged)

# End of function merge_glaciers()

#%% Function glacier_specific_fraction()

def glacier_specific_fraction(ds_dem_cropped, config: dict, name_dir_files: dict):
    
    """
    Function glacierSpecificFraction() returns an xarray DataArray of the 
    glacier fraction in each cell of a given DEM for each specific glacier
    based on the glacier polygon.

    The function takes in an xarray Dataset containing a cropped DEM of the 
    area of interest as input. For each cell of the DEM the fraction of the 
    cell inside the area of each specific glacier is calculated. The
    difference from the function initializeGlacierFraction() is that the 
    current function calculates the fraction of each specific glacier in a
    cell and not the glacier fraction of the merged glacier area.
    
    Parameters
    ----------
    ds_dem_cropped : xarray.Dataset
        Dataset for the area of interest containing:
        Coordinates:
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.
        Data variables:
        elevation (time,Y,X) : xarray.DataArray (float)
            Elevation in each cell.
        Attributes:
        res : float
            Resolution of DEM (cellsize).
    config : dict
        Dictionary of model configuration.
    name_dir_files : dict
        Dictionary containing directories, names of shape files and file with 
        glacier IDs.

    Returns
    -------
    da_gl_spec_frac (BREID,time,Y,X) : xarray.DataArray (float)
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
    """
    
    # Name of shape file directory and shape file. 
    filedir_shp = name_dir_files['filepath_shp']
    gl_shp_file = name_dir_files['filename_shape_gl']
    filedir_id = name_dir_files['filepath_glacier_id']
    gl_id_file = name_dir_files['filename_glacier_id']

    # Get polygons to be merged based on glacier IDs and merge polygons.
    
    # Read shape file as dataframe.
    df_gl = gpd.read_file(filedir_shp + gl_shp_file)  
    
    # Read file with glacier IDs into dataframe.
    df_id = pd.read_csv(filedir_id + gl_id_file, sep=';')
    
    # List of glacier IDs.
    id_list = df_id['BREID'].values.tolist()
    
    # Reduce dataframe to only include glaciers with IDs in id_list.
    df_gl_cropped = df_gl[df_gl.BREID.isin(id_list)]
    
    # Initialize arrays
    
    # Cellsize is difference between coordinate values.
    cellsize = ds_dem_cropped.res # Float
    #time_range = ds_dem_cropped.time.values
    
    # # Initialize array for glacier fraction.
    gl_frac = np.empty((len(ds_dem_cropped.Y), 
                        len(ds_dem_cropped.X)), dtype = np.float32)
    gl_frac.fill(np.nan)
        
    # Initialize DataArray for glacier specific fractions.
    # Empty DataArray of glacier specific fractions. By not setting the 
    # data variable the DataArray is created with nan values in the shape
    # given by dims. When a DataArray of nan is created, the dtype is
    # automatically set to np.float64. To avoid initializing the DataArray 
    # with dtype np.float64 (memory spike), the DataArray is initialized from
    # an empty numpy array with zeros and dtype np.float32. Then the DataArray
    # is filled with nans. 
    da_gl_spec_frac = xr.DataArray(np.empty((len(id_list),# len(time_range),
                                             len(ds_dem_cropped.Y), 
                                             len(ds_dem_cropped.X)), dtype=np.float32),
                                    coords= {'BREID': id_list,
                                            #'time': time_range,
                                            'Y': ds_dem_cropped.Y.values,
                                            'X': ds_dem_cropped.X.values},
                                    #dims=["BREID","time", "Y", "X"],
                                    dims=["BREID", "Y", "X"],
                                    name='glacier_specific_fraction',
                                    attrs={'Name': 'Glacier specific fraction',
                                          'res': cellsize})
        
    # Set values in DataArray to nan.
    da_gl_spec_frac[:] = np.nan
        
    # Calculate fraction of glacier inside each cell of bounded area

    # Create 1-dim numpy arrays from X and Y coordinates of cell centers.
    Y_values = ds_dem_cropped.Y.values.astype(int) # North to south.
    X_values = ds_dem_cropped.X.values.astype(int) # West to east.
        
    # Loop through coordinate and create rectangular polygons corresponding to 
    # cells with size cellsize x cellsize. Find fraction of glacier polygon 
    # inside each rectangular polygon.       
    for i in id_list:
            
        pgon_glacier_area = df_gl_cropped.loc[
            df_gl_cropped['BREID'] == i]['geometry'].values[0]

        # If the polygon area is invalid, add buffer.
        if pgon_glacier_area.is_valid == False:
                
            pgon_glacier_area = pgon_glacier_area.buffer(0)
        
            print('Warning: Invalid polygon, using buffer.' + ' Glacier ID: ' + str(i))
            
        for y in range(0, len(Y_values)):
            for x in range(0, len(X_values)):
                pgon_cell = geometry.box((X_values[x] - cellsize / 2), 
                                         (Y_values[y] - cellsize / 2),
                                         (X_values[x] + cellsize / 2), 
                                         (Y_values[y] + cellsize / 2))
                frac_gl = pgon_cell.intersection(pgon_glacier_area).area / (pgon_cell.area)
                gl_frac[y,x] = frac_gl
        print(i)        
        
        # Add array of glacier fraction to DataArray.
        da_gl_spec_frac.loc[dict(BREID=i)] = gl_frac
        
    # Return dataarray.
    return(da_gl_spec_frac)    

# End of function glacier_specific_fraction()

#%% Function glacier_specific_fraction_from_outline()

def glacier_specific_fraction_from_outline(ds_dem_cropped, config: dict, name_dir_files: dict):
    
    """
    Returns an xarray DataArray of the transient glacier fraction in each cell 
    of a given DEM for each specific glacier based on a series of glacier outlines.

    The function takes in an xarray Dataset containing a cropped DEM of the 
    area of interest as input. For each cell and year of the DEM the fraction of the 
    cell inside the area of each specific glacier is calculated from a given
    outline. 
    
    Parameters
    ----------
    ds_dem_cropped : xarray.Dataset
        Dataset for the area of interest containing:
        Coordinates:
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.
        Data variables:
        elevation (time,Y,X) : xarray.DataArray (float)
            Elevation in each cell.
        Attributes:
        res : float
            Resolution of DEM (cellsize).
    config : dict
        Dictionary of model configuration.
    name_dir_files : dict
        Dictionary containing directories, names of shape files and file with 
        glacier IDs.

    Returns
    -------
    da_gl_spec_frac (BREID,time,Y,X) : xarray.DataArray (float)
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
    """
    
    # Get csv file with overview of glacier IDs, years and filenames of
    # outlines for the given years.
    #filepath_overview = 'C:/Users/kasj/mass_balance_model/shape_files/shp_overview_test.csv'
    filepath_overview = name_dir_files['filepath_shp'] + name_dir_files['filename_shp_overview']
    
    df_shp_overview = pd.read_csv(filepath_overview, sep=';')
    
    # Name of shape file directory and shape file. 
    filedir_id = name_dir_files['filepath_glacier_id']
    gl_id_file = name_dir_files['filename_glacier_id']

    # Get polygons to be merged based on glacier IDs and merge polygons. 
    
    # Read file with glacier IDs into dataframe.
    df_id = pd.read_csv(filedir_id + gl_id_file, sep=';')
    
    # List of glacier IDs.
    id_list = df_id['BREID'].values.tolist()
        
    # Crop file with overview of outlines based on list of glacier IDs.
    df_shp_overview_cropped = df_shp_overview[df_shp_overview.BREID.isin(id_list)]
    
    # Initialize arrays
    
    # Cellsize is difference between coordinate values.
    cellsize = ds_dem_cropped.res # Float
    time_range = ds_dem_cropped.time.values
    
    # Initialize array for glacier fraction.
    gl_frac = np.empty((len(ds_dem_cropped.Y), 
                        len(ds_dem_cropped.X)), dtype = np.float32)
    gl_frac.fill(np.nan)
        
    # Initialize DataArray for glacier specific fractions.
    # Empty DataArray of glacier specific fractions. By not setting the 
    # data variable the DataArray is created with nan values in the shape
    # given by dims. When a DataArray of nan is created, the dtype is
    # automatically set to np.float64. To avoid initializing the DataArray 
    # with dtype np.float64 (memory spike), the DataArray is initialized from
    # an empty numpy array with zeros and dtype np.float32. Then the DataArray
    # is filled with nans. 
    da_gl_spec_frac = xr.DataArray(np.empty((len(id_list), len(time_range),
                                             len(ds_dem_cropped.Y), 
                                             len(ds_dem_cropped.X)), dtype=np.float32),
                                    coords= {'BREID': id_list,
                                            'time': time_range,
                                            'Y': ds_dem_cropped.Y.values,
                                            'X': ds_dem_cropped.X.values},
                                    dims=["BREID","time", "Y", "X"],
                                    name='glacier_specific_fraction',
                                    attrs={'Name': 'Glacier specific fraction',
                                          'res': cellsize})
        
    # Set values in DataArray to nan.
    da_gl_spec_frac[:] = np.nan
        
    # Calculate fraction of glacier inside each cell of bounded area

    # Create 1-dim numpy arrays from X and Y coordinates of cell centers.
    Y_values = ds_dem_cropped.Y.values.astype(int) # North to south.
    X_values = ds_dem_cropped.X.values.astype(int) # West to east.
        
    # Loop through coordinate and create rectangular polygons corresponding to 
    # cells with size cellsize x cellsize. Find fraction of glacier polygon 
    # inside each rectangular polygon.       
    for i in id_list:
            
        # Get series of the years with different outlines for a given glacier from the 
        # overview of outlines.             
        df_gl_series = df_shp_overview_cropped.loc[df_shp_overview_cropped['BREID'] == i, ['year','outline_year','shp_file']]

        start_yr_outline = df_gl_series['outline_year'].values[0]
        start_yr_filepath = df_gl_series.loc[df_gl_series['outline_year'] == start_yr_outline]['shp_file'].values[0]
        compute_frac = True
            
        # Read shape file as dataframe.
        df_gl_shp = gpd.read_file(start_yr_filepath) 

        # Reduce dataframe to only include glaciers with IDs in id_list.
        pgon_glacier_area = df_gl_shp.loc[df_gl_shp['BREID'] == i]['geometry'].values[0]
        
            
        for yr in time_range:
                
            # Get year.
            year = pd.to_datetime(yr).year
                
            # Check year of outline for given year.
            outline_year = df_gl_series.loc[df_gl_series['year'] == year]['outline_year'].values[0]
                
            if outline_year != start_yr_outline:
                start_yr_outline = outline_year
                start_yr_filepath = df_gl_series.loc[df_gl_series['outline_year'] == start_yr_outline]['shp_file'].values[0]
                df_gl_shp = gpd.read_file(start_yr_filepath) 
                pgon_glacier_area = df_gl_shp.loc[df_gl_shp['BREID'] == i]['geometry'].values[0]
                # If new outline, compute fraction for the new outline. 
                compute_frac = True
        
            if compute_frac ==True:
                # If the polygon area is invalid, add buffer.
                if pgon_glacier_area.is_valid == False:
                
                    pgon_glacier_area = pgon_glacier_area.buffer(0)
        
                    print('Warning: Invalid polygon, using buffer.' + ' Glacier ID: ' + str(i))
            
                for y in range(0, len(Y_values)):
                    for x in range(0, len(X_values)):
                        pgon_cell = geometry.box((X_values[x] - cellsize / 2), 
                                                 (Y_values[y] - cellsize / 2),
                                                 (X_values[x] + cellsize / 2), 
                                                 (Y_values[y] + cellsize / 2))
                        frac_gl = pgon_cell.intersection(pgon_glacier_area).area / (pgon_cell.area)
                        gl_frac[y,x] = frac_gl
                compute_frac=False 

            print(yr)          
            # Add array of glacier fraction to DataArray.
            da_gl_spec_frac.loc[dict(BREID=i, time=yr)] = gl_frac
        
    # Return dataarray.
    return(da_gl_spec_frac)    

# End of function glacier_specific_fraction_from_outline()

#%% Function catchment_specific_fraction()

def catchment_specific_fraction(ds_dem_cropped, config: dict, name_dir_files:dict, hres=False):
    
    """
    Function catchment_specific_fraction() returns an xarray DataArray of the 
    glacier fraction in each cell of a given DEM for each specific glacier
    based on the glacier polygon.

    The function takes in an xarray Dataset containing a cropped DEM of the 
    area of interest as input. For each cell of the DEM the fraction of the 
    cell inside the area of each specific catchment is calculated.
    
    Parameters
    ----------
    ds_dem_cropped : xarray.Dataset
        Dataset for the area of interest containing:
        Coordinates:
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.
        Data variables:
        elevation (time,Y,X) : xarray.DataArray (float)
            Elevation in each cell.
        Attributes:
        res : float
            Resolution of DEM (cellsize).
    config : dict
        Dictionary of model configuration.
    name_dir_files : dict
        Dictionary containing directories, names of shape files and file with 
        glacier IDs.

    Returns
    -------
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
    """

    # Filepath of mask.
    if hres == True:
        filepath_mask = name_dir_files['filepath_fractions'] + name_dir_files['filename_high_res_ca_frac']
    else: 
        filepath_mask = name_dir_files['filepath_fractions'] + name_dir_files['filename_ca_frac']

    # While the filename does not exist in filepath, initialize a Dataset 
    # with name 'simulation_name'. Return the new or existing Dataset with the 
    # name 'simulation_name'. 
    while not os.path.isfile(filepath_mask):

        # Get option to compute discharge from catchment containing glaciers.
        get_ca_dis = config['get_catchment_discharge']

        # Cellsize is difference between coordinate values.
        cellsize = ds_dem_cropped.res # Float
    
        # If catchment discharge is to be computed, create a DataArray of catchment fraction
        # for each catchment ID.
        if get_ca_dis == True:

            # Name of shape file directory and shape file. 
            filedir_shp = name_dir_files['filepath_shp']
            shp_file = name_dir_files['filename_shape_ca']
            filedir_id = name_dir_files['filepath_catchment_id']
            id_file = name_dir_files['filename_catchment_id']
    
            # Read shape file of catchment outlines as dataframe.
            df_ca = gpd.read_file(filedir_shp + shp_file)
    
            # Read file with catchment IDs into dataframe.
            df_id = pd.read_csv(filedir_id + id_file)
    
            # List of catchment IDs.
            id_list = df_id['vassdragNr'].values.tolist()
    
            # Reduce dataframe to only include catchments with IDs in id_list.
            df_ca_cropped = df_ca[df_ca.vassdragNr.isin(id_list)]

            # Initialize DataArray for catchment specific fractions.
            # Empty DataArray of catchment specific fractions. By not setting the 
            # data variable the DataArray is created with nan values in the shape
            # given by dims. When a DataArray of nan is created, the dtype is
            # automatically set to np.float64. To avoid initializing the DataArray 
            # with dtype np.float64 (memory spike), the DataArray is initialized from
            # an empty numpy array with zeros and dtype np.float32. Then the DataArray
            # is filled with nans. 
            da_ca_spec_frac = xr.DataArray(np.empty((len(id_list),
                                                     len(ds_dem_cropped.Y), 
                                                     len(ds_dem_cropped.X)), dtype=np.float32),
                                           coords= {'vassdragNr': id_list,
                                                    'Y': ds_dem_cropped.Y.values,
                                                    'X': ds_dem_cropped.X.values},
                                           dims=["vassdragNr", "Y", "X"],
                                           name='catchment_specific_fraction',
                                           attrs={'Name': 'Catchment specific fraction',
                                                  'res': cellsize})
        
            # Set values in DataArray to nan.
            da_ca_spec_frac[:] = np.nan      

            # Create 1-dim numpy arrays from X and Y coordinates of cell centers.
            Y_values = ds_dem_cropped.Y.values.astype(int) # North to south.
            X_values = ds_dem_cropped.X.values.astype(int) # West to east.

            # Initialize array for catchment fraction.
            ca_frac = np.empty((len(ds_dem_cropped.Y), 
                                len(ds_dem_cropped.X)), dtype = np.float32)
            ca_frac.fill(np.nan)

            # Loop through coordinate and create rectangular polygons corresponding to 
            # cells with size cellsize x cellsize. Find fraction of catchment polygon 
            # inside each rectangular polygon.       
            for i in id_list:
                pgon_catchment_area = df_ca_cropped.loc[
                    df_ca_cropped['vassdragNr'] == i]['geometry'].values[0]
    
                for y in range(0, len(Y_values)):
                    for x in range(0, len(X_values)):
                        pgon_cell = geometry.box((X_values[x] - cellsize / 2), 
                                                 (Y_values[y] - cellsize / 2),
                                                 (X_values[x] + cellsize / 2), 
                                                 (Y_values[y] + cellsize / 2))
                        frac_ca = (pgon_cell.intersection(pgon_catchment_area).area 
                                   / (pgon_cell.area))
                        ca_frac[y,x] = frac_ca
            
                # Add array of catchment fraction to DataArray.
                da_ca_spec_frac.loc[dict(vassdragNr=i)] = ca_frac
        
        # If catchment discharge is not to be computed, base DEM area on glacier outlines.
        elif get_ca_dis == False:

            # Create "dummy" list of catchment IDs.
            id_list = ['Dummy']

            # Initialize DataArray for catchment specific fractions. If catchment
            # discharge is not to be computed, a 3-D DataArray based on the combined
            # glacier mask is used.
            da_ca_spec_frac = xr.DataArray(np.zeros((len(id_list),
                                                    len(ds_dem_cropped.Y), 
                                                    len(ds_dem_cropped.X)), dtype=np.float32),
                                           coords= {'vassdragNr': id_list,
                                                    'Y': ds_dem_cropped.Y.values,
                                                    'X': ds_dem_cropped.X.values},
                                           dims=["vassdragNr", "Y", "X"],
                                           name='catchment_specific_fraction',
                                           attrs={'Name': 'Catchment specific fraction',
                                                  'res': cellsize})
            
            # Get combined glacier area (containes 0 for cells outside glacier, 0 < values <=1 for 
            # cells inside glacier area).
            if config['ref_mb'] == True:
                
                # Glacier fraction has dimensions [Y,X]
                glacier_area = ds_dem_cropped.glacier_fraction.values
                
                # Set all cells that are part of the glacier to ones.
                # If reference mass balance, all cells part of the glacier
                # area should already be 1. 
                glacier_area[glacier_area > 0] = 1

            else:
                
                # Glacier fraction has dimensions [time,Y,X]. Sum over dimension time.
                glacier_area = np.sum(ds_dem_cropped.glacier_fraction.values, axis=0)

                # Set all cells that are part of the glacier to ones. 
                glacier_area[glacier_area > 0] = 1

            # Add array of catchment fraction to DataArray.
            # The catchment fraction now contains 0 for all cells that are not part of the 
            # glacier and 1 for all cells that are part of the glacier.
            da_ca_spec_frac.loc[dict(vassdragNr='Dummy')] = glacier_area

        else:
            sys.exit('Specify whether or not to compute discharge from catchment.')
    
        # Save Dataarray as netcdf file. Dataarray is always saved as Dataset
        # when saved to netcdf.
        da_ca_spec_frac.to_netcdf(filepath_mask)
    
        # Close files.
        da_ca_spec_frac.close()       
        
    # Get Dataset from filepath and return.   
    with xr.open_dataset(filepath_mask) as ds_ca_spec_frac:
        
        # Get dataarray from dataset.
        da_ca_spec_frac = ds_ca_spec_frac.catchment_specific_fraction
        
        # Return dataarray.
        return(da_ca_spec_frac)    
    
# End of function catchment_specific_fraction()

#%% Function get_ice_thickness()
# This function initializes ice thickness from Farinotti 2019. 

def get_ice_thickness(ds_dem_cropped, da_gl_specific_fraction,
                           init_yr: str, config: dict, name_dir_files: dict):
    
    """
    The function get_ice_thickness gets the ice thickness for each
    glacier ID from the consensus estimate of Farinotti et al. (2019). 
    It returns a DataArray of ice thickness for each
    glacier ID and hydrological year. 
    
    IMPORTANT! In the current version the ice thickness variable is set to be 
    constant in time (same ice thickness for the entire time range). In the 
    future this can be changed so that only the ice thickness variable may be 
    populated for the first year (remaining years contain None).
    
    Parameters
    ----------
    ds_dem_cropped : xarray.Dataset
        Dataset for the area of interest containing:
        Coordinates:
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.
        Data variables:
        elevation (time,Y,X) : xarray.DataArray (float)
            Elevation in each cell.
        Attributes:
        res : float
            Resolution of DEM (cellsize).
    da_gl_specific_fraction (BREID,time,Y,X) : xarray.DataArray (float)
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
    init_yr : str
        Year of intitialization (start year).        
    config: dict
        Dictionary of model configuration.
    name_dir_files : dict
        Dictionary containing name of directories, shapefiles, 
        file with glacier IDs, file with catchment IDs, and information on 
        storage of netCdf file. 

    Returns
    -------
    da_ice_thickness_out : xarray.DataArray
        DataArray containing:
        Coordinates:
        BREID : int
            Glacier IDs.
        time : datetime64[ns]
            Time index (yearly).  
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.
        Data variables:
        ice_thickness (BREID,time,Y,X) : xarray.DataArray (float)
            Elevation of bed in each cell.
        Attributes:
        res : float
            Resolution of DEM (cellsize).
    """ 
    #%% Create file with initial ice thickness distribution 
    
    filepath = name_dir_files['filepath_ice_thickness'] + name_dir_files['filename_ice_thickness']

    # While the filename does not exist in filepath, initialize a Dataset 
    # with name 'simulation_name'. Return the new or existing Dataset with the 
    # name 'simulation_name'. 
    while not os.path.isfile(filepath):
        
        # Name of filepath of ice thickness data storage:
        filepath_ice_thickness_files = '/mirror/khsjursen/mass_balance_model/data/'
        # Name of filepath of surface DEM storage:
        #filepath_surface_dem_files = 'C:/Users/kasj/OneDrive - Høgskulen på Vestlandet/Data/surface_DEMs_RGI60-08/'
        # Name of filepath of file with glacier IDs:
        filepath_id = name_dir_files['filepath_glacier_id']
        # Name of file with glacier IDs:
        gl_id_file = name_dir_files['filename_glacier_id']

        #%% Get glacier IDS: BREID & corresponding RGIID 
    
        # Read file with glacier IDs into dataframe.
        df_id = pd.read_csv(filepath_id + gl_id_file, sep=';')
    
        # List of glacier IDs (BREID).
        breid_list = df_id['BREID'].values.tolist()
        
        # List of glacier IDs (RGIID).
        rgiid_list = df_id['RGIID'].values.tolist()
        
        #%% Initialize DataArrays
        
        # Get the time range (representing n hydrological years).
        time_range = ds_dem_cropped.time.values
    
        # Cellsize is difference between coordinate values.
        cellsize = ds_dem_cropped.res # Float
    
        # Initialize DataArray for bed topography for each glacier.
        # Empty DataArray of bed topography. By not setting the 
        # data variable the DataArray is created with nan values in the shape
        # given by dims. When a DataArray of nan is created, the dtype is
        # automatically set to np.float64. To avoid initializing the DataArray 
        # with dtype np.float64 (memory spike), the DataArray is initialized from
        # an empty numpy array with zeros and dtype np.float32. Then the DataArray
        # is filled with nans. 
        da_ice_thickness = xr.DataArray(np.empty((len(breid_list),
                                                  len(time_range),
                                                  len(ds_dem_cropped.Y.values),
                                                  len(ds_dem_cropped.X.values)), 
                                                 dtype=np.float32),
                                        coords= {'BREID': breid_list,
                                                 'time': time_range,
                                                 'Y': ds_dem_cropped.Y.values,
                                                 'X': ds_dem_cropped.X.values},
                                        dims=["BREID", "time", "Y", "X"],
                                        name='ice_thickness',
                                        attrs={'Name': 'Glacier ice thickness',
                                               'res': cellsize})   
        
        # Set values in DataArray to nan.
        da_ice_thickness[:] = np.nan

        #%% Define projection, shape and transform of destination
        # Source data has projection WGS84, UTM32 (EPSG:32632). Must be 
        # converted to WGS84, UTM33 (EPSG:32633). 

        # Define destination projection (WGS84, UTM33).
        dst_crs = {'init': 'EPSG:32633'}  
        
        # Get shape of destination dataset (DEM).
        dst_shape = (ds_dem_cropped.elevation.shape[1], 
                     ds_dem_cropped.elevation.shape[2]) 

        # Set transform of destination dataset. First two arguments of 
        # A.translation should be coordinates of upper left corner. 
        # Note! Not the coordinates of the cell center!         
        dst_transform = (A.translation(ds_dem_cropped.X.values[0] - cellsize/2, 
                                       ds_dem_cropped.Y.values[0] + cellsize/2) 
                         * A.scale(cellsize, -cellsize))
        
        # Initialize numpy arrays of destination datasets.
        ice_thickness = np.zeros(dst_shape, np.float32)
        #surface_dem = np.zeros(dst_shape, np.float32)
        #bed_topo = np.zeros(dst_shape, np.float32)

        #%% Get data and resample
        
        # Loop through list of RGIID, get ice thickness and surface DEM,
        # resample and reproject to base DEM and calculate base topography
        for i in range(0, len(rgiid_list)):
            
            # Get RGIID from list of glacier IDs.
            rgiid = rgiid_list[i]
            
            # Get BREID from list of glacier IDs.
            breid = breid_list[i]
            
            # Get glacier fraction for the given BREID and initialization year
            # and drop one dimension. 
            glacier_fraction = da_gl_specific_fraction.sel(dict(time=init_yr,
                                                                BREID=breid)).values.squeeze()
            
            # Create mask of glacier area with ones in cells that are part of
            # the glacier are, nan otherwise.
            mask_glacier = glacier_fraction.copy()
            mask_glacier[mask_glacier==0] = np.nan
            mask_glacier[mask_glacier>0] = 1

            # Load ice thickness dataset for RGIID.
            # Coordinates are in WGS84 / UTM32.
            # Resolution is 25x25m.
            with rasterio.open(filepath_ice_thickness_files + rgiid 
                               + '_thickness.tif', 'r') as src:
                
                # Get source projection.
                src_crs = src.crs
                # Get source transform.
                src_transform = src.transform
                # Read source dataset as array.
                source = src.read(1)
                # Set value of cells with zero ice thickness to nan.
                source[source==0] = np.nan
            
                # Resample and reproject ice thickness data to resolution and 
                # projection of base DEM. 
                reproject(
                    source,
                    ice_thickness,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    src_nodata = np.nan,
                    dst_nodata = np.nan,
                    resampling=Resampling.bilinear)
                
                # Create mask of ice thickness data with zeros in place of 
                # nodata values. 
                ice_thickness_mask = np.copy(ice_thickness)
                ice_thickness_mask[np.isnan(ice_thickness_mask)] = 0
                
                # Fill values in ice thickness array that are zero in mask.
                # Entire array is now filled with interpolated values. 
                ice_thickness_filled = rasterio.fill.fillnodata(ice_thickness, 
                                                                mask=ice_thickness_mask)
                
                # Mask filled ice thickness array with mask of glacier area.
                # All cells that are part of the glacier now have an ice 
                # thickness value from resampling or filling. Cells outside 
                # the glacier area have value nan. 
                ice_thickness_filled = np.multiply(ice_thickness_filled, 
                                                   mask_glacier)
    
                # Add array of bedrock topography for the given glacier ID to 
                # DataArray.
                da_ice_thickness.loc[dict(BREID=breid)] = ice_thickness_filled
                
        # Save ice thickness DataArray as netcdf file. Dataarray is 
        # automatically converted to Dataset when calling to_netcdf().
        da_ice_thickness.to_netcdf(filepath)
        
        # Close.
        da_ice_thickness.close()
        
    # Get Dataset from filepath and return.   
    with xr.open_dataset(filepath) as ds_ice_thickness:
        
        # Get DataArray from Dataset.
        da_ice_thickness_out = ds_ice_thickness.ice_thickness
        
        # Return DataArray
        return(da_ice_thickness_out)   

# End of function get_ice_thickness()

#%% End of preprocessing.py %%#