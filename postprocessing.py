# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 09:43:24 2021

@author: kasj

-------------------------------------------------------------------
Mass-balance model
-------------------------------------------------------------------

Contains functions to extract mass balance and discharge.

"""

#%% Libraries

# Standard libraries

# External libraries
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import numexpr as ne
from scipy import stats
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline

# Internal libraries

#%% Function get_glacier_mass_balance()

def get_glacier_mass_balance_upd(accumulation, da_gl_spec_frac, gl_id, year: int, first_yr: int, start_date_ordinal: int):   

    """
    Calculates mass balance (winter, summer and annual balance) for each 
    glacier for the given year.
    
    Winter balance is calculated as the difference between the maximum 
    accumulation over the specific glacier area during the current 
    hydrological year (01.10. in year-1 until 30.09. in year) and the minimum 
    accumulation over the specific glacier during the past ablation season 
    (01.04. in year-1 until 30.09. in year-1).
    
    Summer balance is calculated as the difference between the minimum 
    accumulation over the specific glacier area during the current ablation 
    season (01.04. in year until 30.09. in year) and the maximum accumulation 
    over the specific glacier area during the current hydrological year 
    (01.10. in year-1 until 30.09. in year). 
    
    Calculations are done for each glacier specified in BREID.
    
    If year is equal to the first year of simulation, balances are set to nan.
    
    Parameters
    ----------
    da_acc (time,Y,X) : xarray.DataArray (float)
        DataArray containing total daily accumulation in each cell. 
        Coordinates:
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.
        Attributes:
        res : float
            Resolution of DEM (cellsize).
    da_gl_spec_frac (BREID,time,Y,X) : xarray.DataArray (float)
        DataArray with fraction of cell inside glacier boundary.
        Coordinates:
        BREID : int
            Glacier IDs.
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.    
    year : int
        Year for which to calculate mass balance.
    first_year : int
        First year of simulation.

    Returns
    -------
    mb(BREID,3) : numpy.array (float)
        Array of balances (columns: winter, summer, annual) for each glacier
        (rows) in BREID for the given year.
    """
    
    # Initialize array of balances for the given year, for each glacier in 
    # gl_id. Fill with nan because the first year will always be nan as the
    # balance cannot be computed. 
    mb = np.empty((len(gl_id),3), dtype=np.float32)
    mb.fill(np.nan)
    
    # If year is not the first year of simulation, compute balances. If 
    # year is first year of simulation, return array of nan.
    if year != first_yr:
        
        # Get start and end of period for calculation.
        start_date = dt.datetime.strptime(str(year-1) + '-03-31', '%Y-%m-%d')
        end_date = dt.datetime.strptime(str(year) + '-10-31', '%Y-%m-%d')
        hyd_yr = dt.datetime.strptime(str(year-1) + '-10-01', '%Y-%m-%d')
        idx_start_ord = dt.datetime.toordinal(start_date)
        idx_end_ord = dt.datetime.toordinal(end_date)
        idx_start = idx_start_ord - start_date_ordinal
        idx_end = idx_end_ord - start_date_ordinal + 1
        
        # Crop accumulation DataArray with start and end dates of hydrological 
        # year and convert to numpy array. Expand dimensions for
        # multiplication.
        glacier_acc = accumulation[idx_start:idx_end,:,:]
        ###glacier_acc = np.array(da_acc.sel(time = slice(start_date, 
        ###                                           end_date)).values)
        ###glacier_acc = np.expand_dims(glacier_acc, axis=0)
        glacier_acc = np.expand_dims(glacier_acc, axis=0)

        # Get all glacier masks for the given hydrological year (the 
        # hydrological year that has just ended). Expand dimensions for
        # multiplication.
        mask_glacier_all = np.array(da_gl_spec_frac.sel(time = hyd_yr).values)
        mask_glacier_all = np.expand_dims(mask_glacier_all, axis=1)
                
        # Get accumulation over each specific glacier for the given
        # hydrological year by multiplying specific glacier masks with the 
        # accumulation grid.
        gl_spec_acc = ne.evaluate("""glacier_acc * mask_glacier_all""".replace(" ",""))
        #gl_spec_acc = np.multiply(glacier_acc, mask_glacier_all)
        
        # Sum accumulation over all cells of each specific glacier.
        #glacier_acc = ne.evaluate("""sum(gl_spec_acc_ne, 2)""", {'gl_spec_acc_ne': ne.evaluate("""sum(gl_spec_acc, 3)""")})       
        glacier_acc_d = gl_spec_acc.sum(axis=(2,3)) / mask_glacier_all.sum(axis=(2,3))
        
        # Winter balance is max accumulation over the current hydrological 
        # year minus the minimum value during the previous years ablation
        # period.
        mb[:,0] = (np.amax(glacier_acc_d[:,-365:None],axis=1) 
                    - np.amin(glacier_acc_d[:,None:(365-90)],axis=1))
        
        # Summer balance is the minimum acccumulation over this years  
        # ablation period minus the maximum value during the current 
        # hydrological year.
        mb[:,1] = (np.amin(glacier_acc_d[:,365:None], axis=1) 
                       - np.amax(glacier_acc_d[:,-365:None], axis=1))
        
        # Annual balance is sum of winter and summer balance.
        mb[:,2] = mb[:,0] + mb[:,1]
        
        # Convert balances to m w.e.
        mb = mb/1e3
        
    # Return array of mass balances. Will be nan for first year.
    return mb

# End of function get_glacier_mass_balance()

#%%

def get_glacier_mass_balance_manual_gl_frac(accumulation, mask_glacier_all, gl_id, year: int, first_yr: int, start_date_ordinal: int):   

    """
    THIS FUNCTION IS THE SAME AS get_glacier_mass_balance_upd, but instead of taking the whole
    da_gl_spec_frac as a parameter, it uses mask_glacier_all with selected glacier mask outside function.
    
    Parameters
    ----------
    da_acc (time,Y,X) : xarray.DataArray (float)
        DataArray containing total daily accumulation in each cell. 
        Coordinates:
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.
        Attributes:
        res : float
            Resolution of DEM (cellsize).
    da_gl_spec_frac (BREID,time,Y,X) : xarray.DataArray (float)
        DataArray with fraction of cell inside glacier boundary.
        Coordinates:
        BREID : int
            Glacier IDs.
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.    
    year : int
        Year for which to calculate mass balance.
    first_year : int
        First year of simulation.

    Returns
    -------
    mb(BREID,3) : numpy.array (float)
        Array of balances (columns: winter, summer, annual) for each glacier
        (rows) in BREID for the given year.
    """
    
    # Initialize array of balances for the given year, for each glacier in 
    # gl_id. Fill with nan because the first year will always be nan as the
    # balance cannot be computed. 
    mb = np.empty((len(gl_id),3), dtype=np.float32)
    mb.fill(np.nan)
    
    # If year is not the first year of simulation, compute balances. If 
    # year is first year of simulation, return array of nan.
    if year != first_yr:
        
        # Get start and end of period for calculation.
        start_date = dt.datetime.strptime(str(year-1) + '-03-31', '%Y-%m-%d')
        end_date = dt.datetime.strptime(str(year) + '-10-31', '%Y-%m-%d')
        hyd_yr = dt.datetime.strptime(str(year-1) + '-10-01', '%Y-%m-%d')
        idx_start_ord = dt.datetime.toordinal(start_date)
        idx_end_ord = dt.datetime.toordinal(end_date)
        idx_start = idx_start_ord - start_date_ordinal
        idx_end = idx_end_ord - start_date_ordinal + 1
        
        # Crop accumulation DataArray with start and end dates of hydrological 
        # year and convert to numpy array. Expand dimensions for
        # multiplication.
        glacier_acc = accumulation[idx_start:idx_end,:,:]
        ###glacier_acc = np.array(da_acc.sel(time = slice(start_date, 
        ###                                           end_date)).values)
        ###glacier_acc = np.expand_dims(glacier_acc, axis=0)
        glacier_acc = np.expand_dims(glacier_acc, axis=0)

        # Get all glacier masks for the given hydrological year (the 
        # hydrological year that has just ended). Expand dimensions for
        # multiplication.
        #mask_glacier_all = np.array(da_gl_spec_frac.sel(time = hyd_yr).values) #THIS IS MOVED OUTSIDE FUNCTON
        mask_glacier_all = np.expand_dims(mask_glacier_all, axis=1)
                
        # Get accumulation over each specific glacier for the given
        # hydrological year by multiplying specific glacier masks with the 
        # accumulation grid.
        gl_spec_acc = ne.evaluate("""glacier_acc * mask_glacier_all""".replace(" ",""))
        #gl_spec_acc = np.multiply(glacier_acc, mask_glacier_all)
        
        # Sum accumulation over all cells of each specific glacier.
        #glacier_acc = ne.evaluate("""sum(gl_spec_acc_ne, 2)""", {'gl_spec_acc_ne': ne.evaluate("""sum(gl_spec_acc, 3)""")})       
        glacier_acc_d = gl_spec_acc.sum(axis=(2,3)) / mask_glacier_all.sum(axis=(2,3))
        
        # Winter balance is max accumulation over the current hydrological 
        # year minus the minimum value during the previous years ablation
        # period.
        mb[:,0] = (np.amax(glacier_acc_d[:,-365:None],axis=1) 
                    - np.amin(glacier_acc_d[:,None:(365-90)],axis=1))
        
        # Summer balance is the minimum acccumulation over this years  
        # ablation period minus the maximum value during the current 
        # hydrological year.
        mb[:,1] = (np.amin(glacier_acc_d[:,365:None], axis=1) 
                       - np.amax(glacier_acc_d[:,-365:None], axis=1))
        
        # Annual balance is sum of winter and summer balance.
        mb[:,2] = mb[:,0] + mb[:,1]
        
        # Convert balances to m w.e.
        mb = mb/1e3
        
    # Return array of mass balances. Will be nan for first year.
    return mb

# End of function get_glacier_mass_balance()

#%% Function get_glacier_mass_balance()

def get_glacier_mass_balance(accumulation, da_gl_spec_frac, gl_id, year: int, first_yr: int, start_date_ordinal: int):   

    """
    Calculates mass balance (winter, summer and annual balance) for each 
    glacier for the given year.
    
    Winter balance is calculated as the difference between the maximum 
    accumulation over the specific glacier area during the current 
    hydrological year (01.10. in year-1 until 30.09. in year) and the minimum 
    accumulation over the specific glacier during the past ablation season 
    (01.04. in year-1 until 30.09. in year-1).
    
    Summer balance is calculated as the difference between the minimum 
    accumulation over the specific glacier area during the current ablation 
    season (01.04. in year until 30.09. in year) and the maximum accumulation 
    over the specific glacier area during the current hydrological year 
    (01.10. in year-1 until 30.09. in year). 
    
    Calculations are done for each glacier specified in BREID.
    
    If year is equal to the first year of simulation, balances are set to nan.
    
    Parameters
    ----------
    da_acc (time,Y,X) : xarray.DataArray (float)
        DataArray containing total daily accumulation in each cell. 
        Coordinates:
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.
        Attributes:
        res : float
            Resolution of DEM (cellsize).
    da_gl_spec_frac (BREID,time,Y,X) : xarray.DataArray (float)
        DataArray with fraction of cell inside glacier boundary.
        Coordinates:
        BREID : int
            Glacier IDs.
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.    
    year : int
        Year for which to calculate mass balance.
    first_year : int
        First year of simulation.

    Returns
    -------
    mb(BREID,3) : numpy.array (float)
        Array of balances (columns: winter, summer, annual) for each glacier
        (rows) in BREID for the given year.
    """
    
    # Initialize array of balances for the given year, for each glacier in 
    # gl_id. Fill with nan because the first year will always be nan as the
    # balance cannot be computed. 
    mb = np.empty((len(gl_id),3), dtype=np.float32)
    mb.fill(np.nan)
    
    # If year is not the first year of simulation, compute balances. If 
    # year is first year of simulation, return array of nan.
    if year != first_yr:
        
        # Get start and end of period for calculation.
        start_date = dt.datetime.strptime(str(year-1) + '-03-31', '%Y-%m-%d')
        end_date = dt.datetime.strptime(str(year) + '-09-30', '%Y-%m-%d')
        hyd_yr = dt.datetime.strptime(str(year-1) + '-10-01', '%Y-%m-%d')
        idx_start_ord = dt.datetime.toordinal(start_date)
        idx_end_ord = dt.datetime.toordinal(end_date)
        idx_start = idx_start_ord - start_date_ordinal
        idx_end = idx_end_ord - start_date_ordinal + 1
        
        # Crop accumulation DataArray with start and end dates of hydrological 
        # year and convert to numpy array. Expand dimensions for
        # multiplication.
        glacier_acc = accumulation[idx_start:idx_end,:,:]
        ###glacier_acc = np.array(da_acc.sel(time = slice(start_date, 
        ###                                           end_date)).values)
        ###glacier_acc = np.expand_dims(glacier_acc, axis=0)
        glacier_acc = np.expand_dims(glacier_acc, axis=0)

        # Get all glacier masks for the given hydrological year (the 
        # hydrological year that has just ended). Expand dimensions for
        # multiplication.
        mask_glacier_all = np.array(da_gl_spec_frac.sel(time = hyd_yr).values)
        mask_glacier_all = np.expand_dims(mask_glacier_all, axis=1)
        
        #time2=time.time()
        
        # Get accumulation over each specific glacier for the given
        # hydrological year by multiplying specific glacier masks with the 
        # accumulation grid.
        gl_spec_acc = ne.evaluate("""glacier_acc * mask_glacier_all""".replace(" ",""))
        #gl_spec_acc = np.multiply(glacier_acc, mask_glacier_all)
        
        # Sum accumulation over all cells of each specific glacier.
        #glacier_acc = ne.evaluate("""sum(gl_spec_acc_ne, 2)""", {'gl_spec_acc_ne': ne.evaluate("""sum(gl_spec_acc, 3)""")})       
        glacier_acc_d = gl_spec_acc.sum(axis=(2,3)) / mask_glacier_all.sum(axis=(2,3))
        
        # Winter balance is max accumulation over the current hydrological 
        # year minus the minimum value during the previous years ablation
        # period.
        mb[:,0] = (np.amax(glacier_acc_d[:,-365:None],axis=1) 
                   - np.amin(glacier_acc_d[:,None:(365-90)],axis=1))
        
        # Summer balance is the minimum acccumulation over this years  
        # ablation period minus the maximum value during the current 
        # hydrological year.
        mb[:,1] = (np.amin(glacier_acc_d[:,365:None], axis=1) 
                       - np.amax(glacier_acc_d[:,-365:None], axis=1))
        
        # Annual balance is sum of winter and summer balance.
        mb[:,2] = mb[:,0] + mb[:,1]
        
        # Convert balances to m w.e.
        mb = mb/1e3
        
        #print(time.time()-time2)

    # Return array of mass balances. Will be nan for first year.
    return mb

# End of function get_glacier_mass_balance()

#%% Function get_ds_glacier_mass_balance_pergla()

def get_ds_glacier_mass_balance_pergla(accumulation, mask_glacier_1_gridded, elev, da_elev_hr, da_gl_spec_frac, da_gl_spec_frac_hr, gl_id, year: int, first_yr: int, start_date_ordinal: int):   

    """
    Calculates and downscales mass balance from coarse to fine resolution
    DEM and glacier mask for a given year and set of glaciers. 
    
    Mass balance (winter and summer) is calculated in each cell of the coarse 
    glacier grid.

    Winter balance is calculated as the difference between the maximum 
    accumulation over the specific glacier area during the current 
    hydrological year (01.10. in year-1 until 30.09. in year) and the minimum 
    accumulation over the specific glacier during the past ablation season 
    (01.04. in year-1 until 30.09. in year-1).
    
    Summer balance is calculated as the difference between the minimum 
    accumulation over the specific glacier area during the current ablation 
    season (01.04. in year until 30.09. in year) and the maximum accumulation 
    over the specific glacier area during the current hydrological year 
    (01.10. in year-1 until 30.09. in year). 

    Uses the statistical downscaling method by Noel et al. (2016). Regression parameters 
    are determined for the equation:

    mb_c = a_c + b_c * h_c
    
    Where mb_c is the mass balance in the coarse grid cell, and h_c is the elevation
    in the coarse grid cell. First b_c is determined by regression to each ice-covered
    cell using the current cell and available neighboring cells (from 3-9 cells in total).
    The a_c is determined by using the b_c together with mb_c in the current grid cell.

    When arrays of a_c and b_c are determined, these are bilinearly interpolated
    to the high-resolution grid to get arrays a_h and b_h. The regression parameters
    a_h and b_h are then used together with the elevation, h_h, in the high-resolution
    DEM to calculate the high-resolution mass balance, mb_h.  
    
    Calculations are done for each glacier specified in gl_id.
    
    If year is equal to the first year of simulation, balances are set to nan.
    
    Parameters
    ----------
    tot_accumulation (time,Y,X) : numpy.array (float)
        Numpy array containing total daily accumulation in each cell. 
        Coarse resolution.
    mask_glacier_1_gridded (Y,X) : numpy.array (float)
        Numpy array containing ones in all cells that have glacier cover,
        zero otherwise. Coarse resolution.
    elev (Y,X) : numpy.array (float)
        Numpy array containing elevation of coarse grid.
    da_elev_hr (time,Y,X) : xarray.DataArray (float)
        DataArray with elevation for high-resolution grid with time.
        Coordinates:
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.  
    da_gl_spec_frac (BREID,time,Y,X) : xarray.DataArray (float)
        DataArray with fraction of cell inside glacier boundary.
        Coarse grid.
        Coordinates:
        BREID : int
            Glacier IDs.
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.   
    da_gl_spec_frac_hr (BREID,time,Y,X) : xarray.DataArray (float)
        DataArray with fraction of cell inside glacier boundary.
        High-resolution grid.
        Coordinates:
        BREID : int
            Glacier IDs.
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.  
    gl_id : list (int)
        List of glacier IDs (BREID).
    year : int
        Year for which to calculate mass balance.
    first_year : int
        First year of simulation.
    start_date_ordinal : dt.datetime
        Ordinal date for start of simulations.

    Returns
    -------
    mb(BREID,3) : numpy.array (float)
        Array of balances (columns: winter, summer, annual) for each glacier
        (rows) in BREID for the given year.
    """
    
    # Initialize array of balances for the given year, for each glacier in 
    # gl_id. Fill with nan because the first year will always be nan as the
    # balance cannot be computed. 
    mb = np.empty((len(gl_id),3), dtype=np.float32)
    mb.fill(np.nan)
    
    # If year is not the first year of simulation, compute balances. If 
    # year is first year of simulation, return array of nan.
    if year != first_yr:

        # Get start and end of period for calculation.
        start_date = dt.datetime.strptime(str(year-1) + '-03-31', '%Y-%m-%d')
        end_date = dt.datetime.strptime(str(year) + '-10-31', '%Y-%m-%d')
        hyd_yr = dt.datetime.strptime(str(year-1) + '-10-01', '%Y-%m-%d')
        idx_start_ord = dt.datetime.toordinal(start_date)
        idx_end_ord = dt.datetime.toordinal(end_date)
        idx_start = idx_start_ord - start_date_ordinal
        idx_end = idx_end_ord - start_date_ordinal + 1
        
        # Accumulation array is on the form (time, Y1, X1) with accumulation in mm in each
        # cell that is part of the catchment area and zero otherwise. 
        # Crop accumulation array with start and end dates of hydrological
        # year.
        accumulation_hyd = accumulation[idx_start:idx_end,:,:]

        # Set cells to zero that are not part of the glacierized area. 
        accumulation_hyd = np.multiply(accumulation_hyd, mask_glacier_1_gridded)
        #accumulation_hyd = ne.evaluate("""accumulation_hyd * mask_glacier_1_gridded""".replace(" ",""))

        # Winter balance is max accumulation over the current hydrological 
        # year minus the minimum value during the previous years ablation
        # period. mbw_c is gridded winter balance with dimension (Y1,X1).
        mbw_c = np.amax(accumulation_hyd[-365:None,:,:], axis=0) - np.amin(accumulation_hyd[None:(395-90),:,:], axis=0)

        # Summer balance is the minimum acccumulation over this years  
        # ablation period minus the maximum value during the current 
        # hydrological year. mbs_c is gridded summer balance with dimension (Y1,X1).
        mbs_c = np.amin(accumulation_hyd[365:None,:], axis=0) - np.amax(accumulation_hyd[-365:None,:], axis=0)

        def get_mb_regression(mb, h, gl_mask_1, r_win, c_win):
        
            """
            Find regression coefficients a and b in the 
            equation y = a + bx. First find b by regression to 
            current cell and and adjoining cells (3-9 cells in total).
            Then find a by using y and h in current cell and
            a = y - bx.

            Parameters
            ----------
            mb : np.array
                Array with values on which to perform regression (y).
            h : np.array
                Array with x variable.
            gl_mask_1: np.array
                Array with glacier mask to determine cells part of 
                glacierized area.
            
            Returns
            -------
            a : np.array
                Array of regression coefficient a.
            b : np.array
                Array of regression coefficient b.
            """
            rows, cols = gl_mask_1.shape
            indices = np.arange(0, rows * cols).reshape(rows, cols)
            gl_mask_nonan = gl_mask_1.copy() # Added from downscale_smb.py 29.09.23
            gl_mask_nonan[np.isnan(gl_mask_1)] = 0 # Added from downscale_smb.py 29.09.23
            indices[gl_mask_nonan < 1] = 0 # Added from downscale_smb.py 29.09.23
            #indices[gl_mask_1 < 1] = 0 # Added from downscale_smb.py 29.09.23
            indices_flat = indices[indices != 0]
            mb_flat = mb.flatten()
            h_flat = h.flatten()
            b = np.empty((mb_flat.shape))
            b.fill(np.nan)
            a = np.empty((mb_flat.shape))
            a.fill(np.nan)

            for i in range(0, len(indices_flat)):
                idx = indices_flat[i]
                indices_sub = np.array([idx-1, idx, idx+1, (idx+cols-1),(idx+cols),(idx+cols+1), (idx-cols-1),(idx-cols),(idx-cols+1)])
                indices_sub = indices_sub[np.isin(indices_sub, indices_flat)]
                mb_sub = mb_flat[indices_sub]
                h_sub = h_flat[indices_sub]
                res = stats.linregress(h_sub, mb_sub)
                b[idx] = res.slope
                a[idx] = mb_flat[idx] - b[idx] * h_flat[idx]

            b = b.reshape(rows,cols)
            a = a.reshape(rows,cols)
            # rows, cols = gl_mask_1.shape
            # #shape_win = (3, 3)
            # #r_win = np.floor(shape_win[0] / 2).astype(int)
            # #c_win = np.floor(shape_win[1] / 2).astype(int)
            # b = np.zeros((rows,cols))
            # a = np.zeros((rows,cols))
            # b[gl_mask_1==0] = np.nan
            # a[gl_mask_1==0] = np.nan

            # #regr = LinearRegression()

            # for y in range(0, rows):
            #     ymin = max(0, y - r_win)
            #     ymax = min(rows, y + r_win + 1)
            #     for x in range(0, cols):
            #         xmin = max(0, x - c_win)
            #         xmax = min(cols, x + c_win + 1)

            #         check_mask = gl_mask_1[y, x]

            #         if check_mask != 0:
            #             mb_sub = mb[ymin:ymax, xmin:xmax]
            #             h_sub = h[ymin:ymax, xmin:xmax]
            #             gl_mask_1_sub = gl_mask_1[ymin:ymax, xmin:xmax]
            #             mb_reg = mb_sub[gl_mask_1_sub>0]
            #             h_reg = h_sub[gl_mask_1_sub>0]
            #             #h_reg = h_sub[gl_mask_1_sub>0].reshape(-1,1)

            #             res = stats.linregress(h_reg, mb_reg)
            #             #res = regr.fit(h_reg, mb_reg)
            #             b[y, x] = res.slope
            #             #b[y, x] = res.coef_
            #             a[y, x] = mb[y, x] - b[y, x]*h[y, x]
            return a, b

        #@jit(nopython=True, cache=True)
        def pad_regression(arr, r_win, c_win):
            rows, cols = arr.shape
            #shape_win = (3, 3)
            #r_win = np.floor(shape_win[0] / 2).astype(int)
            #c_win = np.floor(shape_win[1] / 2).astype(int)
            
            for i in range(2):
                for y in range(0, rows):
                    ymin = max(0, y - r_win)
                    ymax = min(rows, y + r_win + 1)
                    for x in range(0, cols):
                        xmin = max(0, x - c_win)
                        xmax = min(cols, x + c_win + 1)

                        check_mask = arr[y, x]

                        if np.isnan(check_mask):
                            arr_sub = arr[ymin:ymax, xmin:xmax]
                            if np.sum(~np.isnan(arr_sub)) >= 3:
                                #arr[y,x] = np.nanmean(arr_sub)
                                arr[y,x] = np.mean(arr_sub[~np.isnan(arr_sub)])
            return arr
        
        def pad_regression_multiple(arr1, arr2, arr3, arr4, r_win, c_win):
            
            rows, cols = arr1.shape
     
            for i in range(2):
                for y in range(0, rows):
                    ymin = max(0, y - r_win)
                    ymax = min(rows, y + r_win + 1)
                    for x in range(0, cols):
                        xmin = max(0, x - c_win)
                        xmax = min(cols, x + c_win + 1)

                        check_mask = arr1[y, x]

                        if np.isnan(check_mask):
                            arr1_sub = arr1[ymin:ymax, xmin:xmax]
                            arr2_sub = arr2[ymin:ymax, xmin:xmax]
                            arr3_sub = arr3[ymin:ymax, xmin:xmax]
                            arr4_sub = arr4[ymin:ymax, xmin:xmax]

                            if np.sum(~np.isnan(arr1_sub)) >= 3:
                                arr1[y,x] = np.mean(arr1_sub[~np.isnan(arr1_sub)])
                                arr2[y,x] = np.mean(arr2_sub[~np.isnan(arr2_sub)])
                                arr3[y,x] = np.mean(arr3_sub[~np.isnan(arr3_sub)])
                                arr4[y,x] = np.mean(arr4_sub[~np.isnan(arr4_sub)])
            
            return arr1, arr2, arr3, arr4

        # Set window shape for padding and regression.
        shape_win = (3, 3)
        row_win = np.floor(shape_win[0] / 2).astype(int)
        col_win = np.floor(shape_win[1] / 2).astype(int)

        # We have a coarse resolution elevation grid (Y1,X1). 
        # For each array mbs_c and mbw_c, together with elevation (Y1,X1) we compute the
        # regression slope b_1km and intercept a_1km in each cell that is glacier covered, 
        # such that we have b_mbw_1km, a_mbw_1km and b_mbs_1km, a_mbs_1km.
        # Arrays contain regression parameters in cells that are glacier-covered,
        # otherwise nan. 
        a_mbw_c, b_mbw_c = get_mb_regression(mbw_c, elev, mask_glacier_1_gridded, row_win, col_win)
        a_mbs_c, b_mbs_c = get_mb_regression(mbs_c, elev, mask_glacier_1_gridded, row_win, col_win)

        # Pad regression parameters 
        # Padded twice.
        #a_mbw_c = pad_regression(a_mbw_c, row_win, col_win)
        #b_mbw_c = pad_regression(b_mbw_c, row_win, col_win)
        #a_mbs_c = pad_regression(a_mbs_c, row_win, col_win)
        #b_mbs_c = pad_regression(b_mbs_c, row_win, col_win)
        a_mbw_c, b_mbw_c, a_mbs_c, b_mbs_c = pad_regression_multiple(a_mbw_c, b_mbw_c, a_mbs_c, b_mbs_c, row_win, col_win)

        # Does not work with PYMC3!
        # def reproject_regression(source, Y_src, X_src, Y_dest, X_dest, cellsize_src, cellsize_dest):
        #     """
        #     Reprojection of source array with coordinates (Y_src, X_src)
        #     and resolution cellsize_src to destination array with 
        #     coordinates (Y_dest, X_dest) and resolution cellsize_dest.

        #     Parameters
        #     ----------
        #     source : np.array
        #         Array with values to reproject.
        #     Y_src : np.array
        #         Y-coordinates (rows) of source array.
        #     X_src: np.array
        #         X-coordinates (columns) of source array.
        #     Y_dest : np.array
        #         Y-coordinates (rows) of destination array.
        #     X_dest : np.array
        #         X-coordinates (columns) of destination array.
        #     cellsize_src : int
        #         Resolution of source.
        #     cellsize_dest : int
        #         Resolution of destination. 
            
        #     Returns
        #     -------
        #     destination : np.array
        #         Array of reprojected values.
        #     """

        #     dst_crs = {'init': 'EPSG:32633'}
        #     src_crs = {'init': 'EPSG:32633'}

        #     dst_transform = A.translation((X_dest[0] - cellsize_dest/2), 
        #                       (Y_dest[0] + cellsize_dest/2)) * A.scale(cellsize_dest, -cellsize_dest)
        #     src_transform = A.translation((X_src[0] - cellsize_src/2), 
        #                       (Y_src[0] + cellsize_src/2)) * A.scale(cellsize_src, -cellsize_src)

        #     destination = np.empty((len(Y_dest), len(X_dest)))
        #     destination[:] = np.nan

        #     reproject(source,
        #               destination,
        #               src_transform = src_transform,
        #               src_crs = src_crs,
        #               dst_transform = dst_transform,
        #               dst_crs = dst_crs,
        #               src_nodata = np.nan,
        #               dst_nodata = np.nan,
        #               resampling = Resampling.bilinear)
            
        #     return destination

        # Resample regression parameters to high-resolution grid. 
        elev_h = np.array(da_elev_hr.sel(time=hyd_yr).values)
        gl_frac_h = np.array(da_gl_spec_frac_hr.sel(time = hyd_yr).values)

        # Get source and destination coordinates.
        Y_c = da_gl_spec_frac.Y.values
        X_c = da_gl_spec_frac.X.values
        Y_h = da_gl_spec_frac_hr.Y.values
        X_h = da_gl_spec_frac_hr.X.values

        # Get source and destination cellsize.
        cellsize_c = da_gl_spec_frac.res
        cellsize_h = da_gl_spec_frac_hr.res
        coarseness = cellsize_h/cellsize_c

        # Reproject regression coefficients a and b from coarse resolution
        # DEM to high resolution DEM.
        # Function does not work with PYMC3!
        #a_mbw_h = reproject_regression(a_mbw_c, Y_c, X_c, Y_h, X_h, cellsize_c, cellsize_h)
        #b_mbw_h = reproject_regression(b_mbw_c, Y_c, X_c, Y_h, X_h, cellsize_c, cellsize_h)
        #a_mbs_h = reproject_regression(a_mbs_c, Y_c, X_c, Y_h, X_h, cellsize_c, cellsize_h)
        #b_mbs_h = reproject_regression(b_mbs_c, Y_c, X_c, Y_h, X_h, cellsize_c, cellsize_h)

        # Replace Nan with zeros.
        a_mbw_c[np.isnan(a_mbw_c)] = 0
        b_mbw_c[np.isnan(b_mbw_c)] = 0
        a_mbs_c[np.isnan(a_mbs_c)] = 0
        b_mbs_c[np.isnan(b_mbs_c)] = 0

        def downscale_coeff(arr, Y_src, X_src, Y_dest, X_dest, scaling):
            
            x_src = np.arange(0, len(X_src), 1)
            y_src = np.arange(0, len(Y_src), 1)
            x_dest = np.arange(0, len(X_src), coarseness)
            y_dest = np.arange(0, len(Y_src), coarseness)

            f = RectBivariateSpline(y_src, x_src, arr) # Added from downscale_smb.py 29.09.23
            # f = interp2d(x_src, y_src, arr, kind='linear')

            arr_interp = f(x_dest, y_dest)

            return(arr_interp)
        
        a_mbw_h = downscale_coeff(a_mbw_c, Y_c, X_c, Y_h, X_h, coarseness)
        b_mbw_h = downscale_coeff(b_mbw_c, Y_c, X_c, Y_h, X_h, coarseness)
        a_mbs_h = downscale_coeff(a_mbs_c, Y_c, X_c, Y_h, X_h, coarseness)
        b_mbs_h = downscale_coeff(b_mbs_c, Y_c, X_c, Y_h, X_h, coarseness)

        #if year == 2015:
        #    plt.imshow(a_mbw_h)
        #    plt.imshow(b_mbw_h)
        #    plt.imshow(a_mbs_h)
        #    plt.imshow(b_mbs_h)

        # Calculate mass balance in each cell of the high resolution DEM.
        mbw_h = a_mbw_h + b_mbw_h * elev_h
        mbs_h = a_mbs_h + b_mbs_h * elev_h
        mba_h = mbw_h + mbs_h

        #mba_distr = np.multiply(np.sum(gl_frac_h, axis=0), mba_h)/1e3

        # Calculate distributed mass balance.
        mbw_gl = np.multiply(gl_frac_h, mbw_h)
        mbs_gl = np.multiply(gl_frac_h, mbs_h)
        mba_gl = np.multiply(gl_frac_h, mba_h)

        # Total mass balance for each glacier in mwe.
        mbw_spec = mbw_gl.sum(axis=(1,2)) / (gl_frac_h.sum(axis=(1,2))*1e3)
        mbs_spec = mbs_gl.sum(axis=(1,2)) / (gl_frac_h.sum(axis=(1,2))*1e3)
        mba_spec = mba_gl.sum(axis=(1,2)) / (gl_frac_h.sum(axis=(1,2))*1e3)

        mb[:,0] = mbw_spec
        mb[:,1] = mbs_spec
        mb[:,2] = mba_spec

    # Return array of mass balances. Will be nan for first year.
    return mb#, mba_distr

# End of function get_ds_glacier_mass_balance()

#%% Function get_ds_glacier_mass_balance()

def get_ds_glacier_mass_balance_upd(accumulation, mask_glacier_1_gridded, elev, da_elev_hr, da_gl_spec_frac, da_gl_spec_frac_hr, gl_id, year: int, first_yr: int, start_date_ordinal: int):   

    """
    Calculates and downscales mass balance from coarse to fine resolution
    DEM and glacier mask for a given year and set of glaciers. 
    
    Mass balance (winter and summer) is calculated in each cell of the coarse 
    glacier grid.

    Winter balance is calculated as the difference between the maximum 
    accumulation over the specific glacier area during the current 
    hydrological year (01.10. in year-1 until 30.09. in year) and the minimum 
    accumulation over the specific glacier during the past ablation season 
    (01.04. in year-1 until 30.09. in year-1).
    
    Summer balance is calculated as the difference between the minimum 
    accumulation over the specific glacier area during the current ablation 
    season (01.04. in year until 30.09. in year) and the maximum accumulation 
    over the specific glacier area during the current hydrological year 
    (01.10. in year-1 until 30.09. in year). 

    Uses the statistical downscaling method by Noel et al. (2016). Regression parameters 
    are determined for the equation:

    mb_c = a_c + b_c * h_c
    
    Where mb_c is the mass balance in the coarse grid cell, and h_c is the elevation
    in the coarse grid cell. First b_c is determined by regression to each ice-covered
    cell using the current cell and available neighboring cells (from 3-9 cells in total).
    The a_c is determined by using the b_c together with mb_c in the current grid cell.

    When arrays of a_c and b_c are determined, these are bilinearly interpolated
    to the high-resolution grid to get arrays a_h and b_h. The regression parameters
    a_h and b_h are then used together with the elevation, h_h, in the high-resolution
    DEM to calculate the high-resolution mass balance, mb_h.  
    
    Calculations are done for each glacier specified in gl_id.
    
    If year is equal to the first year of simulation, balances are set to nan.
    
    Parameters
    ----------
    tot_accumulation (time,Y,X) : numpy.array (float)
        Numpy array containing total daily accumulation in each cell. 
        Coarse resolution.
    mask_glacier_1_gridded (Y,X) : numpy.array (float)
        Numpy array containing ones in all cells that have glacier cover,
        zero otherwise. Coarse resolution.
    elev (Y,X) : numpy.array (float)
        Numpy array containing elevation of coarse grid.
    da_elev_hr (time,Y,X) : xarray.DataArray (float)
        DataArray with elevation for high-resolution grid with time.
        Coordinates:
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.  
    da_gl_spec_frac (BREID,time,Y,X) : xarray.DataArray (float)
        DataArray with fraction of cell inside glacier boundary.
        Coarse grid.
        Coordinates:
        BREID : int
            Glacier IDs.
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.   
    da_gl_spec_frac_hr (BREID,time,Y,X) : xarray.DataArray (float)
        DataArray with fraction of cell inside glacier boundary.
        High-resolution grid.
        Coordinates:
        BREID : int
            Glacier IDs.
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.  
    gl_id : list (int)
        List of glacier IDs (BREID).
    year : int
        Year for which to calculate mass balance.
    first_year : int
        First year of simulation.
    start_date_ordinal : dt.datetime
        Ordinal date for start of simulations.

    Returns
    -------
    mb(BREID,3) : numpy.array (float)
        Array of balances (columns: winter, summer, annual) for each glacier
        (rows) in BREID for the given year.
    """
    
    # Initialize array of balances for the given year, for each glacier in 
    # gl_id. Fill with nan because the first year will always be nan as the
    # balance cannot be computed. 
    mb = np.empty((len(gl_id),3), dtype=np.float32)
    mb.fill(np.nan)
    
    # If year is not the first year of simulation, compute balances. If 
    # year is first year of simulation, return array of nan.
    if year != first_yr:

        # Get start and end of period for calculation.
        start_date = dt.datetime.strptime(str(year-1) + '-03-31', '%Y-%m-%d')
        end_date = dt.datetime.strptime(str(year) + '-10-31', '%Y-%m-%d')
        hyd_yr = dt.datetime.strptime(str(year-1) + '-10-01', '%Y-%m-%d')
        idx_start_ord = dt.datetime.toordinal(start_date)
        idx_end_ord = dt.datetime.toordinal(end_date)
        idx_start = idx_start_ord - start_date_ordinal
        idx_end = idx_end_ord - start_date_ordinal + 1
        
        # Accumulation array is on the form (time, Y1, X1) with accumulation in mm in each
        # cell that is part of the catchment area and zero otherwise. 
        # Crop accumulation array with start and end dates of hydrological
        # year.
        accumulation_hyd = accumulation[idx_start:idx_end,:,:]

        # Set cells to zero that are not part of the glacierized area. 
        accumulation_hyd = np.multiply(accumulation_hyd, mask_glacier_1_gridded)
        #accumulation_hyd = ne.evaluate("""accumulation_hyd * mask_glacier_1_gridded""".replace(" ",""))
            
        w_max = np.argmax(np.sum(accumulation_hyd[-365:None,:,:], axis=(1,2)))
        s_min_prev = np.argmin(np.sum(accumulation_hyd[None:(395-90),:,:], axis=(1,2)))
        s_min_curr = np.argmin(np.sum(accumulation_hyd[365:None,:,:], axis=(1,2)))
            
        # Winter balance is max accumulation over the current hydrological 
        # year minus the minimum value during the previous years ablation
        # period. mbw_c is gridded winter balance with dimension (Y1,X1).
        mbw_c = accumulation_hyd[-365 + w_max,:,:] - accumulation_hyd[s_min_prev,:,:]
        #mbw_c = np.amax(accumulation_hyd[-365:None,:,:], axis=0) - np.amin(accumulation_hyd[None:(395-90),:,:], axis=0)

        # Summer balance is the minimum acccumulation over this years  
        # ablation period minus the maximum value during the current 
        # hydrological year. mbs_c is gridded summer balance with dimension (Y1,X1).
        mbs_c = accumulation_hyd[365 + s_min_curr,:,:] - accumulation_hyd[-365 + w_max,:,:]
        #mbs_c = np.amin(accumulation_hyd[365:None,:,:], axis=0) - np.amax(accumulation_hyd[-365:None,:,:], axis=0)
        
        def get_mb_regression(mb, h, gl_mask_1, r_win, c_win):
        
            """
            Find regression coefficients a and b in the 
            equation y = a + bx. First find b by regression to 
            current cell and and adjoining cells (3-9 cells in total).
            Then find a by using y and h in current cell and
            a = y - bx.

            Parameters
            ----------
            mb : np.array
                Array with values on which to perform regression (y).
            h : np.array
                Array with x variable.
            gl_mask_1: np.array
                Array with glacier mask to determine cells part of 
                glacierized area.
            
            Returns
            -------
            a : np.array
                Array of regression coefficient a.
            b : np.array
                Array of regression coefficient b.
            """
            rows, cols = gl_mask_1.shape
            indices = np.arange(0, rows * cols).reshape(rows, cols)
            gl_mask_nonan = gl_mask_1.copy() # Added from downscale_smb.py 29.09.23
            gl_mask_nonan[np.isnan(gl_mask_1)] = 0 # Added from downscale_smb.py 29.09.23
            indices[gl_mask_nonan < 1] = 0 # Added from downscale_smb.py 29.09.23
            #indices[gl_mask_1 < 1] = 0 # Added from downscale_smb.py 29.09.23
            indices_flat = indices[indices != 0]
            mb_flat = mb.flatten()
            h_flat = h.flatten()
            b = np.empty((mb_flat.shape))
            b.fill(np.nan)
            a = np.empty((mb_flat.shape))
            a.fill(np.nan)

            for i in range(0, len(indices_flat)):
                idx = indices_flat[i]
                indices_sub = np.array([idx-1, idx, idx+1, (idx+cols-1),(idx+cols),(idx+cols+1), (idx-cols-1),(idx-cols),(idx-cols+1)])
                indices_sub = indices_sub[np.isin(indices_sub, indices_flat)]
                mb_sub = mb_flat[indices_sub]
                h_sub = h_flat[indices_sub]
                res = stats.linregress(h_sub, mb_sub)
                b[idx] = res.slope
                a[idx] = mb_flat[idx] - b[idx] * h_flat[idx]

            b = b.reshape(rows,cols)
            a = a.reshape(rows,cols)
            # rows, cols = gl_mask_1.shape
            # #shape_win = (3, 3)
            # #r_win = np.floor(shape_win[0] / 2).astype(int)
            # #c_win = np.floor(shape_win[1] / 2).astype(int)
            # b = np.zeros((rows,cols))
            # a = np.zeros((rows,cols))
            # b[gl_mask_1==0] = np.nan
            # a[gl_mask_1==0] = np.nan

            # #regr = LinearRegression()

            # for y in range(0, rows):
            #     ymin = max(0, y - r_win)
            #     ymax = min(rows, y + r_win + 1)
            #     for x in range(0, cols):
            #         xmin = max(0, x - c_win)
            #         xmax = min(cols, x + c_win + 1)

            #         check_mask = gl_mask_1[y, x]

            #         if check_mask != 0:
            #             mb_sub = mb[ymin:ymax, xmin:xmax]
            #             h_sub = h[ymin:ymax, xmin:xmax]
            #             gl_mask_1_sub = gl_mask_1[ymin:ymax, xmin:xmax]
            #             mb_reg = mb_sub[gl_mask_1_sub>0]
            #             h_reg = h_sub[gl_mask_1_sub>0]
            #             #h_reg = h_sub[gl_mask_1_sub>0].reshape(-1,1)

            #             res = stats.linregress(h_reg, mb_reg)
            #             #res = regr.fit(h_reg, mb_reg)
            #             b[y, x] = res.slope
            #             #b[y, x] = res.coef_
            #             a[y, x] = mb[y, x] - b[y, x]*h[y, x]
            return a, b

        #@jit(nopython=True, cache=True)
        def pad_regression(arr, r_win, c_win):
            rows, cols = arr.shape
            #shape_win = (3, 3)
            #r_win = np.floor(shape_win[0] / 2).astype(int)
            #c_win = np.floor(shape_win[1] / 2).astype(int)
            
            for i in range(2):
                for y in range(0, rows):
                    ymin = max(0, y - r_win)
                    ymax = min(rows, y + r_win + 1)
                    for x in range(0, cols):
                        xmin = max(0, x - c_win)
                        xmax = min(cols, x + c_win + 1)

                        check_mask = arr[y, x]

                        if np.isnan(check_mask):
                            arr_sub = arr[ymin:ymax, xmin:xmax]
                            if np.sum(~np.isnan(arr_sub)) >= 3:
                                #arr[y,x] = np.nanmean(arr_sub)
                                arr[y,x] = np.mean(arr_sub[~np.isnan(arr_sub)])
            return arr
        
        def pad_regression_multiple(arr1, arr2, arr3, arr4, r_win, c_win):
            
            rows, cols = arr1.shape
     
            for i in range(2):
                for y in range(0, rows):
                    ymin = max(0, y - r_win)
                    ymax = min(rows, y + r_win + 1)
                    for x in range(0, cols):
                        xmin = max(0, x - c_win)
                        xmax = min(cols, x + c_win + 1)

                        check_mask = arr1[y, x]

                        if np.isnan(check_mask):
                            arr1_sub = arr1[ymin:ymax, xmin:xmax]
                            arr2_sub = arr2[ymin:ymax, xmin:xmax]
                            arr3_sub = arr3[ymin:ymax, xmin:xmax]
                            arr4_sub = arr4[ymin:ymax, xmin:xmax]

                            if np.sum(~np.isnan(arr1_sub)) >= 3:
                                arr1[y,x] = np.mean(arr1_sub[~np.isnan(arr1_sub)])
                                arr2[y,x] = np.mean(arr2_sub[~np.isnan(arr2_sub)])
                                arr3[y,x] = np.mean(arr3_sub[~np.isnan(arr3_sub)])
                                arr4[y,x] = np.mean(arr4_sub[~np.isnan(arr4_sub)])
            
            return arr1, arr2, arr3, arr4

        # Set window shape for padding and regression.
        shape_win = (3, 3)
        row_win = np.floor(shape_win[0] / 2).astype(int)
        col_win = np.floor(shape_win[1] / 2).astype(int)

        # We have a coarse resolution elevation grid (Y1,X1). 
        # For each array mbs_c and mbw_c, together with elevation (Y1,X1) we compute the
        # regression slope b_1km and intercept a_1km in each cell that is glacier covered, 
        # such that we have b_mbw_1km, a_mbw_1km and b_mbs_1km, a_mbs_1km.
        # Arrays contain regression parameters in cells that are glacier-covered,
        # otherwise nan. 
        a_mbw_c, b_mbw_c = get_mb_regression(mbw_c, elev, mask_glacier_1_gridded, row_win, col_win)
        a_mbs_c, b_mbs_c = get_mb_regression(mbs_c, elev, mask_glacier_1_gridded, row_win, col_win)

        # Pad regression parameters 
        # Padded twice.
        #a_mbw_c = pad_regression(a_mbw_c, row_win, col_win)
        #b_mbw_c = pad_regression(b_mbw_c, row_win, col_win)
        #a_mbs_c = pad_regression(a_mbs_c, row_win, col_win)
        #b_mbs_c = pad_regression(b_mbs_c, row_win, col_win)
        a_mbw_c, b_mbw_c, a_mbs_c, b_mbs_c = pad_regression_multiple(a_mbw_c, b_mbw_c, a_mbs_c, b_mbs_c, row_win, col_win)

        # Does not work with PYMC3!
        # def reproject_regression(source, Y_src, X_src, Y_dest, X_dest, cellsize_src, cellsize_dest):
        #     """
        #     Reprojection of source array with coordinates (Y_src, X_src)
        #     and resolution cellsize_src to destination array with 
        #     coordinates (Y_dest, X_dest) and resolution cellsize_dest.

        #     Parameters
        #     ----------
        #     source : np.array
        #         Array with values to reproject.
        #     Y_src : np.array
        #         Y-coordinates (rows) of source array.
        #     X_src: np.array
        #         X-coordinates (columns) of source array.
        #     Y_dest : np.array
        #         Y-coordinates (rows) of destination array.
        #     X_dest : np.array
        #         X-coordinates (columns) of destination array.
        #     cellsize_src : int
        #         Resolution of source.
        #     cellsize_dest : int
        #         Resolution of destination. 
            
        #     Returns
        #     -------
        #     destination : np.array
        #         Array of reprojected values.
        #     """

        #     dst_crs = {'init': 'EPSG:32633'}
        #     src_crs = {'init': 'EPSG:32633'}

        #     dst_transform = A.translation((X_dest[0] - cellsize_dest/2), 
        #                       (Y_dest[0] + cellsize_dest/2)) * A.scale(cellsize_dest, -cellsize_dest)
        #     src_transform = A.translation((X_src[0] - cellsize_src/2), 
        #                       (Y_src[0] + cellsize_src/2)) * A.scale(cellsize_src, -cellsize_src)

        #     destination = np.empty((len(Y_dest), len(X_dest)))
        #     destination[:] = np.nan

        #     reproject(source,
        #               destination,
        #               src_transform = src_transform,
        #               src_crs = src_crs,
        #               dst_transform = dst_transform,
        #               dst_crs = dst_crs,
        #               src_nodata = np.nan,
        #               dst_nodata = np.nan,
        #               resampling = Resampling.bilinear)
            
        #     return destination

        # Resample regression parameters to high-resolution grid. 
        elev_h = np.array(da_elev_hr.sel(time=hyd_yr).values)
        gl_frac_h = np.array(da_gl_spec_frac_hr.sel(time = hyd_yr).values)

        # Get source and destination coordinates.
        Y_c = da_gl_spec_frac.Y.values
        X_c = da_gl_spec_frac.X.values
        Y_h = da_gl_spec_frac_hr.Y.values
        X_h = da_gl_spec_frac_hr.X.values

        # Get source and destination cellsize.
        cellsize_c = da_gl_spec_frac.res
        cellsize_h = da_gl_spec_frac_hr.res
        coarseness = cellsize_h/cellsize_c

        # Reproject regression coefficients a and b from coarse resolution
        # DEM to high resolution DEM.
        # Function does not work with PYMC3!
        #a_mbw_h = reproject_regression(a_mbw_c, Y_c, X_c, Y_h, X_h, cellsize_c, cellsize_h)
        #b_mbw_h = reproject_regression(b_mbw_c, Y_c, X_c, Y_h, X_h, cellsize_c, cellsize_h)
        #a_mbs_h = reproject_regression(a_mbs_c, Y_c, X_c, Y_h, X_h, cellsize_c, cellsize_h)
        #b_mbs_h = reproject_regression(b_mbs_c, Y_c, X_c, Y_h, X_h, cellsize_c, cellsize_h)

        # Replace Nan with zeros.
        a_mbw_c[np.isnan(a_mbw_c)] = 0
        b_mbw_c[np.isnan(b_mbw_c)] = 0
        a_mbs_c[np.isnan(a_mbs_c)] = 0
        b_mbs_c[np.isnan(b_mbs_c)] = 0

        def downscale_coeff(arr, Y_src, X_src, Y_dest, X_dest, scaling):
            
            x_src = np.arange(0, len(X_src), 1)
            y_src = np.arange(0, len(Y_src), 1)
            x_dest = np.arange(0, len(X_src), coarseness)
            y_dest = np.arange(0, len(Y_src), coarseness)

            f = RectBivariateSpline(y_src, x_src, arr) # Added from downscale_smb.py 29.09.23
            #f = interp2d(x_src, y_src, arr, kind='linear')

            arr_interp = f(x_dest, y_dest)

            return(arr_interp)
        
        a_mbw_h = downscale_coeff(a_mbw_c, Y_c, X_c, Y_h, X_h, coarseness)
        b_mbw_h = downscale_coeff(b_mbw_c, Y_c, X_c, Y_h, X_h, coarseness)
        a_mbs_h = downscale_coeff(a_mbs_c, Y_c, X_c, Y_h, X_h, coarseness)
        b_mbs_h = downscale_coeff(b_mbs_c, Y_c, X_c, Y_h, X_h, coarseness)

        #if year == 2015:
        #    plt.imshow(a_mbw_h)
        #    plt.imshow(b_mbw_h)
        #    plt.imshow(a_mbs_h)
        #    plt.imshow(b_mbs_h)

        # Calculate mass balance in each cell of the high resolution DEM.
        mbw_h = a_mbw_h + b_mbw_h * elev_h
        mbs_h = a_mbs_h + b_mbs_h * elev_h
        mba_h = mbw_h + mbs_h

        #mba_distr = np.multiply(np.sum(gl_frac_h, axis=0), mba_h)/1e3

        # Calculate distributed mass balance.
        mbw_gl = np.multiply(gl_frac_h, mbw_h)
        mbs_gl = np.multiply(gl_frac_h, mbs_h)
        mba_gl = np.multiply(gl_frac_h, mba_h)

        # Total mass balance for each glacier in mwe.
        mbw_spec = mbw_gl.sum(axis=(1,2)) / (gl_frac_h.sum(axis=(1,2))*1e3)
        mbs_spec = mbs_gl.sum(axis=(1,2)) / (gl_frac_h.sum(axis=(1,2))*1e3)
        mba_spec = mba_gl.sum(axis=(1,2)) / (gl_frac_h.sum(axis=(1,2))*1e3)

        mb[:,0] = mbw_spec
        mb[:,1] = mbs_spec
        mb[:,2] = mba_spec

    # Return array of mass balances. Will be nan for first year.
    return mb#, mba_distr

# End of function get_ds_glacier_mass_balance_upd()


#%% Function get_ds_glacier_mass_balance()

def get_ds_glacier_mass_balance(accumulation, mask_glacier_1_gridded, elev, da_elev_hr, da_gl_spec_frac, da_gl_spec_frac_hr, gl_id, year: int, first_yr: int, start_date_ordinal: int):   

    """
    Calculates and downscales mass balance from coarse to fine resolution
    DEM and glacier mask for a given year and set of glaciers. 
    
    Mass balance (winter and summer) is calculated in each cell of the coarse 
    glacier grid.

    Winter balance is calculated as the difference between the maximum 
    accumulation over the specific glacier area during the current 
    hydrological year (01.10. in year-1 until 30.09. in year) and the minimum 
    accumulation over the specific glacier during the past ablation season 
    (01.04. in year-1 until 30.09. in year-1).
    
    Summer balance is calculated as the difference between the minimum 
    accumulation over the specific glacier area during the current ablation 
    season (01.04. in year until 30.09. in year) and the maximum accumulation 
    over the specific glacier area during the current hydrological year 
    (01.10. in year-1 until 30.09. in year). 

    Uses the statistical downscaling method by Noel et al. (2016). Regression parameters 
    are determined for the equation:

    mb_c = a_c + b_c * h_c
    
    Where mb_c is the mass balance in the coarse grid cell, and h_c is the elevation
    in the coarse grid cell. First b_c is determined by regression to each ice-covered
    cell using the current cell and available neighboring cells (from 3-9 cells in total).
    The a_c is determined by using the b_c together with mb_c in the current grid cell.

    When arrays of a_c and b_c are determined, these are bilinearly interpolated
    to the high-resolution grid to get arrays a_h and b_h. The regression parameters
    a_h and b_h are then used together with the elevation, h_h, in the high-resolution
    DEM to calculate the high-resolution mass balance, mb_h.  
    
    Calculations are done for each glacier specified in gl_id.
    
    If year is equal to the first year of simulation, balances are set to nan.
    
    Parameters
    ----------
    tot_accumulation (time,Y,X) : numpy.array (float)
        Numpy array containing total daily accumulation in each cell. 
        Coarse resolution.
    mask_glacier_1_gridded (Y,X) : numpy.array (float)
        Numpy array containing ones in all cells that have glacier cover,
        zero otherwise. Coarse resolution.
    elev (Y,X) : numpy.array (float)
        Numpy array containing elevation of coarse grid.
    da_elev_hr (time,Y,X) : xarray.DataArray (float)
        DataArray with elevation for high-resolution grid with time.
        Coordinates:
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.  
    da_gl_spec_frac (BREID,time,Y,X) : xarray.DataArray (float)
        DataArray with fraction of cell inside glacier boundary.
        Coarse grid.
        Coordinates:
        BREID : int
            Glacier IDs.
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.   
    da_gl_spec_frac_hr (BREID,time,Y,X) : xarray.DataArray (float)
        DataArray with fraction of cell inside glacier boundary.
        High-resolution grid.
        Coordinates:
        BREID : int
            Glacier IDs.
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.  
    gl_id : list (int)
        List of glacier IDs (BREID).
    year : int
        Year for which to calculate mass balance.
    first_year : int
        First year of simulation.
    start_date_ordinal : dt.datetime
        Ordinal date for start of simulations.

    Returns
    -------
    mb(BREID,3) : numpy.array (float)
        Array of balances (columns: winter, summer, annual) for each glacier
        (rows) in BREID for the given year.
    """
    
    # Initialize array of balances for the given year, for each glacier in 
    # gl_id. Fill with nan because the first year will always be nan as the
    # balance cannot be computed. 
    mb = np.empty((len(gl_id),3), dtype=np.float32)
    mb.fill(np.nan)
    
    # If year is not the first year of simulation, compute balances. If 
    # year is first year of simulation, return array of nan.
    if year != first_yr:

        # Get start and end of period for calculation.
        start_date = dt.datetime.strptime(str(year-1) + '-03-31', '%Y-%m-%d')
        end_date = dt.datetime.strptime(str(year) + '-09-30', '%Y-%m-%d')
        hyd_yr = dt.datetime.strptime(str(year-1) + '-10-01', '%Y-%m-%d')
        idx_start_ord = dt.datetime.toordinal(start_date)
        idx_end_ord = dt.datetime.toordinal(end_date)
        idx_start = idx_start_ord - start_date_ordinal
        idx_end = idx_end_ord - start_date_ordinal + 1
        
        # Accumulation array is on the form (time, Y1, X1) with accumulation in mm in each
        # cell that is part of the catchment area and zero otherwise. 
        # Crop accumulation array with start and end dates of hydrological
        # year.
        accumulation_hyd = accumulation[idx_start:idx_end,:,:]

        # Set cells to zero that are not part of the glacierized area. 
        accumulation_hyd = np.multiply(accumulation_hyd, mask_glacier_1_gridded)
        #accumulation_hyd = ne.evaluate("""accumulation_hyd * mask_glacier_1_gridded""".replace(" ",""))

        # Winter balance is max accumulation over the current hydrological 
        # year minus the minimum value during the previous years ablation
        # period. mbw_c is gridded winter balance with dimension (Y1,X1).
        mbw_c = np.amax(accumulation_hyd[-365:None,:,:], axis=0) - np.amin(accumulation_hyd[None:(395-90),:,:], axis=0)

        # Summer balance is the minimum acccumulation over this years  
        # ablation period minus the maximum value during the current 
        # hydrological year. mbs_c is gridded summer balance with dimension (Y1,X1).
        mbs_c = np.amin(accumulation_hyd[365:None,:], axis=0) - np.amax(accumulation_hyd[-365:None,:], axis=0)
 
        def get_mb_regression(mb, h, gl_mask_1, r_win, c_win):
        
            """
            Find regression coefficients a and b in the 
            equation y = a + bx. First find b by regression to 
            current cell and and adjoining cells (3-9 cells in total).
            Then find a by using y and h in current cell and
            a = y - bx.

            Parameters
            ----------
            mb : np.array
                Array with values on which to perform regression (y).
            h : np.array
                Array with x variable.
            gl_mask_1: np.array
                Array with glacier mask to determine cells part of 
                glacierized area.
            
            Returns
            -------
            a : np.array
                Array of regression coefficient a.
            b : np.array
                Array of regression coefficient b.
            """
            rows, cols = gl_mask_1.shape
            indices = np.arange(0, rows * cols).reshape(rows, cols)
            gl_mask_nonan = gl_mask_1.copy() # Added from downscale_smb.py 29.09.23
            gl_mask_nonan[np.isnan(gl_mask_1)] = 0 # Added from downscale_smb.py 29.09.23
            indices[gl_mask_nonan < 1] = 0 # Added from downscale_smb.py 29.09.23
            #indices[gl_mask_1 < 1] = 0 # Added from downscale_smb.py 29.09.23
            indices_flat = indices[indices != 0]
            mb_flat = mb.flatten()
            h_flat = h.flatten()
            b = np.empty((mb_flat.shape))
            b.fill(np.nan)
            a = np.empty((mb_flat.shape))
            a.fill(np.nan)

            for i in range(0, len(indices_flat)):
                idx = indices_flat[i]
                indices_sub = np.array([idx-1, idx, idx+1, (idx+cols-1),(idx+cols),(idx+cols+1), (idx-cols-1),(idx-cols),(idx-cols+1)])
                indices_sub = indices_sub[np.isin(indices_sub, indices_flat)]
                mb_sub = mb_flat[indices_sub]
                h_sub = h_flat[indices_sub]
                res = stats.linregress(h_sub, mb_sub)
                b[idx] = res.slope
                a[idx] = mb_flat[idx] - b[idx] * h_flat[idx]

            b = b.reshape(rows,cols)
            a = a.reshape(rows,cols)
            # rows, cols = gl_mask_1.shape
            # #shape_win = (3, 3)
            # #r_win = np.floor(shape_win[0] / 2).astype(int)
            # #c_win = np.floor(shape_win[1] / 2).astype(int)
            # b = np.zeros((rows,cols))
            # a = np.zeros((rows,cols))
            # b[gl_mask_1==0] = np.nan
            # a[gl_mask_1==0] = np.nan

            # #regr = LinearRegression()

            # for y in range(0, rows):
            #     ymin = max(0, y - r_win)
            #     ymax = min(rows, y + r_win + 1)
            #     for x in range(0, cols):
            #         xmin = max(0, x - c_win)
            #         xmax = min(cols, x + c_win + 1)

            #         check_mask = gl_mask_1[y, x]

            #         if check_mask != 0:
            #             mb_sub = mb[ymin:ymax, xmin:xmax]
            #             h_sub = h[ymin:ymax, xmin:xmax]
            #             gl_mask_1_sub = gl_mask_1[ymin:ymax, xmin:xmax]
            #             mb_reg = mb_sub[gl_mask_1_sub>0]
            #             h_reg = h_sub[gl_mask_1_sub>0]
            #             #h_reg = h_sub[gl_mask_1_sub>0].reshape(-1,1)

            #             res = stats.linregress(h_reg, mb_reg)
            #             #res = regr.fit(h_reg, mb_reg)
            #             b[y, x] = res.slope
            #             #b[y, x] = res.coef_
            #             a[y, x] = mb[y, x] - b[y, x]*h[y, x]
            return a, b

        #@jit(nopython=True, cache=True)
        def pad_regression(arr, r_win, c_win):
            rows, cols = arr.shape
            #shape_win = (3, 3)
            #r_win = np.floor(shape_win[0] / 2).astype(int)
            #c_win = np.floor(shape_win[1] / 2).astype(int)
            
            for i in range(2):
                for y in range(0, rows):
                    ymin = max(0, y - r_win)
                    ymax = min(rows, y + r_win + 1)
                    for x in range(0, cols):
                        xmin = max(0, x - c_win)
                        xmax = min(cols, x + c_win + 1)

                        check_mask = arr[y, x]

                        if np.isnan(check_mask):
                            arr_sub = arr[ymin:ymax, xmin:xmax]
                            if np.sum(~np.isnan(arr_sub)) >= 3:
                                #arr[y,x] = np.nanmean(arr_sub)
                                arr[y,x] = np.mean(arr_sub[~np.isnan(arr_sub)])
            return arr
        
        def pad_regression_multiple(arr1, arr2, arr3, arr4, r_win, c_win):
            
            rows, cols = arr1.shape
     
            for i in range(2):
                for y in range(0, rows):
                    ymin = max(0, y - r_win)
                    ymax = min(rows, y + r_win + 1)
                    for x in range(0, cols):
                        xmin = max(0, x - c_win)
                        xmax = min(cols, x + c_win + 1)

                        check_mask = arr1[y, x]

                        if np.isnan(check_mask):
                            arr1_sub = arr1[ymin:ymax, xmin:xmax]
                            arr2_sub = arr2[ymin:ymax, xmin:xmax]
                            arr3_sub = arr3[ymin:ymax, xmin:xmax]
                            arr4_sub = arr4[ymin:ymax, xmin:xmax]

                            if np.sum(~np.isnan(arr1_sub)) >= 3:
                                arr1[y,x] = np.mean(arr1_sub[~np.isnan(arr1_sub)])
                                arr2[y,x] = np.mean(arr2_sub[~np.isnan(arr2_sub)])
                                arr3[y,x] = np.mean(arr3_sub[~np.isnan(arr3_sub)])
                                arr4[y,x] = np.mean(arr4_sub[~np.isnan(arr4_sub)])
            
            return arr1, arr2, arr3, arr4

        # Set window shape for padding and regression.
        shape_win = (3, 3)
        row_win = np.floor(shape_win[0] / 2).astype(int)
        col_win = np.floor(shape_win[1] / 2).astype(int)

        # We have a coarse resolution elevation grid (Y1,X1). 
        # For each array mbs_c and mbw_c, together with elevation (Y1,X1) we compute the
        # regression slope b_1km and intercept a_1km in each cell that is glacier covered, 
        # such that we have b_mbw_1km, a_mbw_1km and b_mbs_1km, a_mbs_1km.
        # Arrays contain regression parameters in cells that are glacier-covered,
        # otherwise nan. 
        a_mbw_c, b_mbw_c = get_mb_regression(mbw_c, elev, mask_glacier_1_gridded, row_win, col_win)
        a_mbs_c, b_mbs_c = get_mb_regression(mbs_c, elev, mask_glacier_1_gridded, row_win, col_win)

        # Pad regression parameters 
        # Padded twice.
        #a_mbw_c = pad_regression(a_mbw_c, row_win, col_win)
        #b_mbw_c = pad_regression(b_mbw_c, row_win, col_win)
        #a_mbs_c = pad_regression(a_mbs_c, row_win, col_win)
        #b_mbs_c = pad_regression(b_mbs_c, row_win, col_win)
        a_mbw_c, b_mbw_c, a_mbs_c, b_mbs_c = pad_regression_multiple(a_mbw_c, b_mbw_c, a_mbs_c, b_mbs_c, row_win, col_win)

        # Does not work with PYMC3!
        # def reproject_regression(source, Y_src, X_src, Y_dest, X_dest, cellsize_src, cellsize_dest):
        #     """
        #     Reprojection of source array with coordinates (Y_src, X_src)
        #     and resolution cellsize_src to destination array with 
        #     coordinates (Y_dest, X_dest) and resolution cellsize_dest.

        #     Parameters
        #     ----------
        #     source : np.array
        #         Array with values to reproject.
        #     Y_src : np.array
        #         Y-coordinates (rows) of source array.
        #     X_src: np.array
        #         X-coordinates (columns) of source array.
        #     Y_dest : np.array
        #         Y-coordinates (rows) of destination array.
        #     X_dest : np.array
        #         X-coordinates (columns) of destination array.
        #     cellsize_src : int
        #         Resolution of source.
        #     cellsize_dest : int
        #         Resolution of destination. 
            
        #     Returns
        #     -------
        #     destination : np.array
        #         Array of reprojected values.
        #     """

        #     dst_crs = {'init': 'EPSG:32633'}
        #     src_crs = {'init': 'EPSG:32633'}

        #     dst_transform = A.translation((X_dest[0] - cellsize_dest/2), 
        #                       (Y_dest[0] + cellsize_dest/2)) * A.scale(cellsize_dest, -cellsize_dest)
        #     src_transform = A.translation((X_src[0] - cellsize_src/2), 
        #                       (Y_src[0] + cellsize_src/2)) * A.scale(cellsize_src, -cellsize_src)

        #     destination = np.empty((len(Y_dest), len(X_dest)))
        #     destination[:] = np.nan

        #     reproject(source,
        #               destination,
        #               src_transform = src_transform,
        #               src_crs = src_crs,
        #               dst_transform = dst_transform,
        #               dst_crs = dst_crs,
        #               src_nodata = np.nan,
        #               dst_nodata = np.nan,
        #               resampling = Resampling.bilinear)
            
        #     return destination

        # Resample regression parameters to high-resolution grid. 
        elev_h = np.array(da_elev_hr.sel(time=hyd_yr).values)
        gl_frac_h = np.array(da_gl_spec_frac_hr.sel(time = hyd_yr).values)

        # Get source and destination coordinates.
        Y_c = da_gl_spec_frac.Y.values
        X_c = da_gl_spec_frac.X.values
        Y_h = da_gl_spec_frac_hr.Y.values
        X_h = da_gl_spec_frac_hr.X.values

        # Get source and destination cellsize.
        cellsize_c = da_gl_spec_frac.res
        cellsize_h = da_gl_spec_frac_hr.res
        coarseness = cellsize_h/cellsize_c

        # Reproject regression coefficients a and b from coarse resolution
        # DEM to high resolution DEM.
        # Function does not work with PYMC3!
        #a_mbw_h = reproject_regression(a_mbw_c, Y_c, X_c, Y_h, X_h, cellsize_c, cellsize_h)
        #b_mbw_h = reproject_regression(b_mbw_c, Y_c, X_c, Y_h, X_h, cellsize_c, cellsize_h)
        #a_mbs_h = reproject_regression(a_mbs_c, Y_c, X_c, Y_h, X_h, cellsize_c, cellsize_h)
        #b_mbs_h = reproject_regression(b_mbs_c, Y_c, X_c, Y_h, X_h, cellsize_c, cellsize_h)

        # Replace Nan with zeros.
        a_mbw_c[np.isnan(a_mbw_c)] = 0
        b_mbw_c[np.isnan(b_mbw_c)] = 0
        a_mbs_c[np.isnan(a_mbs_c)] = 0
        b_mbs_c[np.isnan(b_mbs_c)] = 0

        def downscale_coeff(arr, Y_src, X_src, Y_dest, X_dest, scaling):
            
            x_src = np.arange(0, len(X_src), 1)
            y_src = np.arange(0, len(Y_src), 1)
            x_dest = np.arange(0, len(X_src), coarseness)
            y_dest = np.arange(0, len(Y_src), coarseness)

            f = RectBivariateSpline(y_src, x_src, arr) # Added from downscale_smb.py 29.09.23
            #f = interp2d(x_src, y_src, arr, kind='linear')

            arr_interp = f(x_dest, y_dest)

            return(arr_interp)
        
        a_mbw_h = downscale_coeff(a_mbw_c, Y_c, X_c, Y_h, X_h, coarseness)
        b_mbw_h = downscale_coeff(b_mbw_c, Y_c, X_c, Y_h, X_h, coarseness)
        a_mbs_h = downscale_coeff(a_mbs_c, Y_c, X_c, Y_h, X_h, coarseness)
        b_mbs_h = downscale_coeff(b_mbs_c, Y_c, X_c, Y_h, X_h, coarseness)

        #if year == 2015:
        #    plt.imshow(a_mbw_h)
        #    plt.imshow(b_mbw_h)
        #    plt.imshow(a_mbs_h)
        #    plt.imshow(b_mbs_h)

        # Calculate mass balance in each cell of the high resolution DEM.
        mbw_h = a_mbw_h + b_mbw_h * elev_h
        mbs_h = a_mbs_h + b_mbs_h * elev_h
        mba_h = mbw_h + mbs_h

        #mba_distr = np.multiply(np.sum(gl_frac_h, axis=0), mba_h)/1e3

        # Calculate distributed mass balance.
        mbw_gl = np.multiply(gl_frac_h, mbw_h)
        mbs_gl = np.multiply(gl_frac_h, mbs_h)
        mba_gl = np.multiply(gl_frac_h, mba_h)

        # Total mass balance for each glacier in mwe.
        mbw_spec = mbw_gl.sum(axis=(1,2)) / (gl_frac_h.sum(axis=(1,2))*1e3)
        mbs_spec = mbs_gl.sum(axis=(1,2)) / (gl_frac_h.sum(axis=(1,2))*1e3)
        mba_spec = mba_gl.sum(axis=(1,2)) / (gl_frac_h.sum(axis=(1,2))*1e3)

        mb[:,0] = mbw_spec
        mb[:,1] = mbs_spec
        mb[:,2] = mba_spec

    # Return array of mass balances. Will be nan for first year.
    return mb#, mba_distr

# End of function get_ds_glacier_mass_balance()

#%% Function getInternalBalance()
# Single-glacier version

def getInternalBalance(old_prec, curr_prec, elevation, glacier_mask, glacier_mask_ones, gl_id):   

    # Latent heat of fusion.
    L_f = 334 * 1e3 # [J kg-1] = [kg m2 s-2 kg-1]

    # Acceleration of gravity.
    g = 9.81 # [m s-2]

    # Precipitation from the end of the previous hydrological year to the end of the previous year.
    prec_fall = old_prec[-92:,:].sum(axis=0) # 1. oct to 31. dec
    
    # Precipitation from the start of the current year to the end of the current hydrological year.
    prec_spring = curr_prec[:-92,:].sum(axis=0) # 1. jan to 31. sept
    
    # Total distributed precipitation in the hydrological year.
    prec_hyd = prec_fall + prec_spring # mm w.e.
    prec_hyd_m = prec_hyd / 1e3 # m w.e.

    # Minimum elevation of the glacier (elevation of glacier snout).
    min_elev = min(elevation * glacier_mask_ones)

    # Total area of glacier.
    A_tot = glacier_mask.sum()

    # Total dissipation related to mass throughput.
    Dis_tot = (g * prec_hyd_m * glacier_mask * (elevation - min_elev))

    # Total melt by dissipation of energy.
    B_int = Dis_tot.sum()/(A_tot * L_f)

    return B_int

# End of function getInternalBalance()

#%% Function get_internal_ablation()
# Multi-glacier version

def get_internal_ablation(old_prec, curr_prec, elevation, da_gl_spec_frac, mask_catchment_1, gl_id, year, first_yr):   

    """
    Calculates internal ablation for each glacier over a year using the 
    method by Oerlemans (2013).
    
    Calculations are done for each glacier specified in gl_id.
    
    If year is equal to the first year of simulation, internal ablation is set to nan.
    
    Parameters
    ----------
    old_prec (Y*X,) : numpy.array (float)
        Vector containing adjusted/corrected precipitation over the glacierized area
        for the previous year.
    curr_prec (Y*X,) : numpy.array (float)
        Vector containing adjusted/corrected precipitation over the glacierized area 
        for the current year.  
    elevation (Y,X) : numpy.array (float)
        Array of elevation over the area. 
    da_gl_spec_frac (BREID,time,Y,X) : xarray.DataArray (float)
        DataArray with fraction of cell inside glacier boundary.
        Coordinates:
        BREID : int
            Glacier IDs.
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers. 
    gl_id : list (int)
        List of glacier IDs (BREID).
    year : int
        Year for which to calculate mass balance.
    first_year : int
        First year of simulation.

    Returns
    -------
    mb_int(len(gl_id),) : numpy.array (float)
        Array of internal ablation for each glacier for the given year.
    """

    # Initialize array of internal ablation for the given year, for each glacier in 
    # gl_id. Fill with nan because the first year will always be nan as the
    # balance cannot be computed. 
    mb_int = np.empty((len(gl_id),1), dtype=np.float32)
    mb_int.fill(np.nan)
    
    # If year is not the first year of simulation, compute internal ablation. If 
    # year is first year of simulation, return array of nan.
    if year != first_yr:

        # Latent heat of fusion.
        L_f = 334 * 1e3 # [J kg-1] = [kg m2 s-2 kg-1]

        # Acceleration of gravity.
        g = 9.81 # [m s-2]

        # Precipitation from the end of the previous hydrological year to the end of the previous year.
        prec_fall = old_prec[-92:,:].sum(axis=0) # 1. oct to 31. dec
    
        # Precipitation from the start of the current year to the end of the current hydrological year.
        prec_spring = curr_prec[:-92,:].sum(axis=0) # 1. jan to 31. sept
    
        # Total distributed precipitation in the hydrological year.
        prec_hyd = prec_fall + prec_spring # mm w.e.
        #prec_hyd_m = (mask_glacier_1_vec * prec_hyd) / 1e3 # m w.e.
        prec_hyd_m = prec_hyd / 1e3

        # Project vector of precipitation to glacier grid.
        prec_hyd_m_grid = mask_catchment_1.copy()
        prec_hyd_m_grid[np.isnan(prec_hyd_m_grid)] = 0
        prec_hyd_m_grid[prec_hyd_m_grid>0] = prec_hyd_m

        if config['update_area_from_outline'] == True and config['update_area_type'] == 'manual':
        
            map_yrs = config['map_years']
        
            map_idx = [i for i, j in enumerate(map_yrs) if j[0] == (year-1)]
            upd_year = map_yrs[map_idx[0]][1]
            
            # Hydrological year.
            hyd_yr = dt.datetime.strptime(str(upd_year) + '-10-01', '%Y-%m-%d')

        else:
            # Hydrological year.
            hyd_yr = dt.datetime.strptime(str(year-1) + '-10-01', '%Y-%m-%d')

        # Get array of glacier masks corresponding to the hydrological year.
        mask_glacier_all = np.array(da_gl_spec_frac.sel(time = hyd_yr).values)

        # Get array of glacier masks with ones and zeros. 
        mask_glacier_all_1 = np.copy(mask_glacier_all)
        mask_glacier_all_1[mask_glacier_all_1>0] = 1 # (82,47,54)
        
        # Get array of elevation masked by glacier masks.
        mask_elevation = np.multiply(elevation, mask_glacier_all_1) # (82,47,54)

        # Minimum elevation for each glacier.
        min_elev = np.min(np.where(mask_elevation==0, mask_elevation.max(), mask_elevation), axis=(1,2)) #(82,)

        # Area of each glacier.
        area = mask_glacier_all.sum(axis=(1,2)) # (82,)

        # Expand dimensions of min_elev to (n_glaciers,1,1).
        min_elev = np.expand_dims(min_elev, axis = (1,2))

        # Elevation difference for each glacier.
        elev_diff = mask_elevation - min_elev

        # Set cells to zero where elevation difference is negative.
        elev_diff[elev_diff<0] = 0
        
        # Expand dimensions of precipitaion grid to (1,X,Y).
        prec_hyd_m_grid = np.expand_dims(prec_hyd_m_grid, axis=0)

        # Dissipation related to mass throughput.
        dis = g * prec_hyd_m_grid * mask_glacier_all * elev_diff

        # Total dissipation across glacier.
        dis_tot = dis.sum(axis=(1,2)) # (82,)
 
        # Total melt by dissipation of energy.
        mb_int = dis_tot / (area * L_f)

    return mb_int

# End of function get_internal_ablation()


#%% Function get_discharge()

def get_discharge(tot_runoff, mask_catchment_all, cellsize):   
    
    """ 
    Function getDischarge() calculates the discharge from each catchment
    specified by vassdragNr for each day of the simulation period.
    
    The function uses the DataArray of daily runoff in each cell of the
    area and uses the arrays of the fraction of each cell inside the given 
    catchments to determine the total discharge from each catchment.
    
    The input DataArray da_runoff contains the portion of runoff in each cell
    that contributes to discharge at the catchment outlet on any given day.
    Runoff routing is performed in the function massBalance().

    Parameters
    ----------
    da_runoff (time,Y,X) : xarray.DataArray (float)
        DataArray containing daily runoff from each cell. 
        Coordinates:
        time : datetime64[ns]
            Time index (daily).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.
        Attributes:
        res : float
            Resolution of DEM (cellsize).
    da_ca_spec_frac (vassdragNr,Y,X) : xarray.DataArray (float)
        DataArray with fraction of cell inside catchment boundary.
        Coordinates:
        vassdragNr : str
            Catchment IDs.
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.    

    Returns
    -------
    discharge (vassdragNr, time) : xarray.DataArray (float)
        DataArray containing daily discharge from each catchment. 
        Coordinates:
        vassdragNr : str
            Catchment IDs.
        time : datetime64[ns]
            Time index (daily).
    """
    
    # Calculate area of grid cell [m2].
    cell_area = cellsize * cellsize
    
    # For conversion of unit day to second [s/day].
    day_to_sec = 60*60*24
    
    # For conversion of unit m to mm [mm/m].
    m_to_mm = 1e3    
    
    # Get runoff from dataArray and expand dimensions for multiplication.
    runoff = np.expand_dims(tot_runoff, axis=1)
    
    # Get all catchment masks and expand dimensions for multiplication.
    mask_catchment_all = np.expand_dims(mask_catchment_all, axis=0)
    
    # Get runoff for each specific catchment and each day by multiplying 
    # specific catchment masks with the runoff grid.
    #ca_spec_runoff = np.multiply(runoff, mask_catchment_all)
    ca_spec_runoff = ne.evaluate("""runoff * mask_catchment_all""".replace(" ",""))

    # REMOVED this line from original code. Did not have any impact on results.
    #ca_spec_runoff[ca_spec_runoff<1e-10] = 0
    
    # Calculate the total discharge over the whole catchment for each day.
    runoff_sum = ca_spec_runoff.sum(axis=(2,3))
    
    # Get discharge in m3/day for each catchment.
    discharge = (runoff_sum * cell_area)/(day_to_sec * m_to_mm) # m3/s
    
    return discharge

# End of function get_discharge()

#%% Function get_total_mass_balance()
def get_total_mass_balance(da_acc, ds_dem):    

    """
    The function get_total_mass_balance() calculates the total (glacier-wide) 
    surface mass balance (winter, summer and annual balance) of an entire 
    glacierized area based on a DataArray of distributed accumulation and a 
    DataArray of the glacier fraction in each cell.
    
    The winter balance is calculated as the difference between the maximum 
    accumulation over the glacierized area during the current hydrological 
    year (01.10. in year-1 until 30.09. in year) and the minimum accumulation 
    over the glacierized area during the past ablation season (01.04. in 
    year-1 until 30.09. in year-1).
    
    The summer balance is calculated as the difference between the minimum 
    accumulation over the glacierized area during the current ablation season 
    (01.04. in year until 30.09. in year) and the maximum accumulation over 
    the glacierized area during the current hydrological year (01.10. in 
    year-1 until 30.09. in year). 
    
    The function uses the glacier fraction variable in ds_dem to determine
    the fraction of glacier in each cell. The total (glacier_wide) surface
    mass balance is the sum of accumulation and ablation over the entire 
    glacierized area over the course of a hydrological year.
    
    Parameters
    ----------
    da_acc (time,Y,X) : xarray.DataArray (float)
        DataArray containing total daily accumulation in each cell. 
        Coordinates:
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.
        Attributes:
        res : float
            Resolution of DEM (cellsize).
    ds_dem (time,Y,X) : xarray.Dataset
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

    Returns
    -------
    df_mb(year,3) : pandas.Dataframe
        Dataframe of surface mass balances (columns: winter, summer, annual) 
        for each year (rows).
    """ 
    
    # Get accumulation array from file.
    #ds_acc = xr.open_dataset('C:/Users/kasj/mass_balance_model/results/test_JOB_all_sn1_210406/da_acc4.nc')
    #da_acc = ds_acc.__xarray_dataarray_variable__
    
    # Get start year, end year and the length of the simulation period from 
    # the accumulation array.
    start_year = pd.Timestamp(da_acc.time[0].values).year
    end_year = pd.Timestamp(da_acc.time[-1].values).year
    length_period = end_year-start_year+1
    
    # Initialize array of balances (winter, summer, annual) for each year.
    mb = np.empty((length_period,3))
    mb.fill(np.nan)
    
    # Initialize counter.
    i = 1
    
    # Go through each year and calculate the total mass balance for the 
    # glacierized area. 
    for year in range(start_year+1, end_year+1):
        
        # Get start and end of period for calculation.
        start_date = dt.datetime.strptime(str(year-1) + '-03-31', '%Y-%m-%d')
        end_date = dt.datetime.strptime(str(year) + '-09-30', '%Y-%m-%d')
        hyd_yr = dt.datetime.strptime(str(year-1) + '-10-01', '%Y-%m-%d')
        
        # Get array of glacier fraction for the given mass balance year.
        mask_glacierized_area = np.array(ds_dem.glacier_fraction.sel(time = 
                                                                     hyd_yr), dtype=np.float32)
         
        # Crop accumulation DataArray with start and end dates of hydrological 
        # year and convert to numpy array. 
        glacier_acc = np.array(da_acc.sel(time = slice(start_date, 
                                                           end_date)).values)
        
        # Get accumulation over the specific glacier area.
        gl_spec_acc = np.multiply(glacier_acc,mask_glacierized_area)

        # Total accumulation over the specific glacier area for each day.
        glacier_acc_d = (np.nansum(gl_spec_acc, axis = (1,2)) 
                         / mask_glacierized_area.sum())
        
        # Winter balance is max accumulation over the current hydrological 
        # year minus the minimum value during the previous years ablation
        # period.
        mb[i,0] = (glacier_acc_d[-365:None].max() 
                      - glacier_acc_d[None:(365-90)].min())
        
        # Summer balance is the minimum acccumulation over this years  
        # ablation period minus the maximum value during the current 
        # hydrological year.
        mb[i,1] = (glacier_acc_d[365:None].min() 
                      - glacier_acc_d[-365:None].max())
        
        # Annual balance is sum of winter and summer balance.
        mb[i,2] = mb[i,0] + mb[i,1]
        
        # Increment counter.
        i = i+1
        
    # Convert balances to m w.e.
    mb = mb/1e3
    
    # Convert to Pandas dataframe and set index equal to range of years and
    # column names 'Bw', 'Bs' and 'Ba' for winter balance, summer balance
    # and annual balance. 
    df_mb = pd.DataFrame(mb, index = range(start_year, end_year+1), columns=['Bw','Bs','Ba'])
    
    # Save dataframe as csv file.
    #df_mb.to_csv('C:/Users/kasj/mass_balance_model/results/test_JOB_all_sn1_210406/mb_all9.csv')
    
    # Return array of mass balances. Will be nan for first year.
    return df_mb

# End of function get_total_mass_balance()

#%% Function get_monthly_mb_grid()
def get_monthly_mb_grid(da_acc, ds_dem):   

    """
    The function get_distributed_mass_balance() calculates the gridded 
    (in each cell) monthly surface mass balance in a glacierized area based on 
    a DataArray of distributed accumulation and a DataArray of the glacier 
    fraction in each cell.
    
    The function uses the glacier fraction variable in ds_dem to determine
    the fraction of glacier in each cell. The gridded monthly surface mass
    balance is calculated for each cell that is part of the glacierized area
    as the sum of accumulation and ablation in each individual cell over the 
    course of each month.
    
    NB! Does not handle start dates other than 01-01-XXXX and end dates other
    than 31-12-XXXX.
    
    NB! Does not take into account a changing glacier area.
    
    Parameters
    ----------
    da_acc (time,Y,X) : xarray.DataArray (float)
        DataArray containing total daily accumulation in each cell. 
        Coordinates:
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.
        Attributes:
        res : float
            Resolution of DEM (cellsize).
    ds_dem (time,Y,X) : xarray.Dataset
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

    Returns
    -------
    da_monthly_mb (year_month,Y,X) : xarray.DataArray (float)
        DataArray of gridded monthly surface mass balance for the 
        glacierized area. 
        Coordinates:
        year_month : object
            Time index, month & year. 
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.    
    """

    # Get accumulation array from file.
    #ds_acc = xr.open_dataset('C:/Users/kasj/mass_balance_model/results/test_JOB_all_sn1_210406/da_acc4.nc')
    #da_acc = ds_acc.__xarray_dataarray_variable__
    
    # Get start and end year and month from accumulation array.
    start_year = pd.Timestamp(da_acc.time[0].values).year
    end_year = pd.Timestamp(da_acc.time[-1].values).year
    
    # If the start year is before 1960, set to 1960.
    start_year = 1960 if start_year < 1960 else start_year
    
    # Make list of years.
    #years = np.arange(start_year, end_year+1, dtype=int)
    
    # Leap years and days in month of leap year and normal year. 
    #leap_years = np.arange(1960, 2024, 4, dtype=int)   
    #days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    #days_in_month_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] # If leap year
    
    # Map years, including leap years, with days in each month. 
    #map_years_monthdays = np.tile(np.array(days_in_month),(len(years),1))
    #mask_leap_years = np.isin(years, leap_years)
    #map_years_monthdays[mask_leap_years,:] = days_in_month_leap
    
    # Make time index of year and month.
    time_range = pd.period_range(str(start_year)+'-01', str(end_year)+'-12', freq ='M').astype(str)
    
    # Get glacier mask.
    mask_glacierized_area = np.array(ds_dem.glacier_fraction)
    
    # Get mask of ones for cells part of glacier, zero otherwise.
    mask_gl_area_ones = mask_glacierized_area.copy()
    mask_gl_area_ones[mask_gl_area_ones>0]=1
    mask_gl_area_ones[mask_gl_area_ones<1]=np.nan
    
    # Crop accumlation array with start and end dates. Include last day before start date.
    da_acc = da_acc.sel(time = slice(str(start_year-1) + '-12-31', str(end_year) + '-12-31'))

    # Get the datetime object for the last day of the month by subtracting from
    # the first of each month. Add last date.
    first_of_months = da_acc.time.sel(time=(da_acc.time.dt.day == 1)).values
    last_of_months = first_of_months - np.timedelta64(1,'D')
    last_of_months = np.append(last_of_months, np.datetime64(str(end_year)+'-12-31'))

    # Get accumulation on the last day of each month as numpy array from 
    # accumulation dataarray.
    acc = da_acc.sel(time = da_acc.time.isin(last_of_months)).values

    # Multiply accumulation array with glacier mask to get nan in cells that 
    # are not part of the glacier area. 
    acc = np.multiply(acc, mask_gl_area_ones)
    
    # Subtract accumulation at last date of the month from accumulation at
    # last date of previous month to get the grid of monthly mass balance.   
    monthly_mb = acc[1:,:,:] - acc[:-1,:,:]
    
    # Convert to DataArray.
    da_monthly_mb = xr.DataArray(monthly_mb,
                                 coords= {'year_month': time_range,
                                          'Y': da_acc.Y.values,
                                          'X': da_acc.X.values},
                                 dims=["year_month", "Y", "X"],
                                 attrs={'Name': 'Gridded monthly mass balance',
                                        'res': da_acc.res,
                                        'dim': 'mm w.e.'})
    
    # Example plot 
    # da_monthly_mb[501,:,:].plot(cmap=plt.cm.RdBu, #vmin = -1000, vmax = 1000,
    #                           cbar_kwargs={'label':'Monthly mass balance (mm w.e.)'})
    # plt.xlabel('$UTM_{east}$')
    # plt.ylabel('$UTM_{north}$')
    # plt.title('Oct 2001')
    
    # Return DataArray of gridded monthly mass balance.
    return da_monthly_mb

# End of function get_monthly_mb_grid()

#%% Function downscale_mb_grid

def downscale_mb_grid(da_monthly_mb, ds_dem, ds_dem_hr):
        
#accumulation, mask_glacier_1_gridded, elev, da_elev_hr, da_gl_spec_frac, da_gl_spec_frac_hr, gl_id, year: int, first_yr: int, start_date_ordinal: int):   

    """
    Downscales mass balance grid from coarse to fine resolution.

    Uses the statistical downscaling method by Noel et al. (2016). Regression parameters 
    are determined for the equation:

    mb_c = a_c + b_c * h_c
    
    Where mb_c is the mass balance in the coarse grid cell, and h_c is the elevation
    in the coarse grid cell. First b_c is determined by regression to each ice-covered
    cell using the current cell and available neighboring cells (from 3-9 cells in total).
    The a_c is determined by using the b_c together with mb_c in the current grid cell.

    When arrays of a_c and b_c are determined, these are bilinearly interpolated
    to the high-resolution grid to get arrays a_h and b_h. The regression parameters
    a_h and b_h are then used together with the elevation, h_h, in the high-resolution
    DEM to calculate the high-resolution mass balance, mb_h.      
    
    Parameters
    ----------
    da_monthly_mb (year_month,Y,X) : xarray.DataArray (float)
        DataArray of gridded monthly surface mass balance for the 
        glacierized area. Coarse resolution mass balance.
        Coordinates:
        year_month : object
            Time index, month & year. 
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers. 
    ds_dem (Y,X) : xarray.Dataset
        Dataset for the catchment/glacier with coarse resolution containing:
        Coordinates:
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.
        Data variables:
        elevation (Y,X) : xarray.DataArray (float)
            Elevation in each cell in bounding box.
        glacier_fraction (Y,X) : xarray.DataArray (float)
            Fraction of cell inside glacier boundary.
        Attributes:
        res : float
            Resolution of DEM (cellsize).  
    ds_dem_hr (Y,X) : xarray.Dataset
        Dataset for the catchment/glacier with high resolution containing:
        Coordinates:
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.
        Data variables:
        elevation (Y,X) : xarray.DataArray (float)
            Elevation in each cell in bounding box.
        glacier_fraction (Y,X) : xarray.DataArray (float)
            Fraction of cell inside glacier boundary.
        Attributes:
        res : float
            Resolution of DEM (cellsize).  

    Returns
    -------
    da_monthly_mb_hr (year_month,Y,X) : xarray.DataArray (float)
        DataArray of gridded monthly surface mass balance for the 
        glacierized area. High resolution mass balance.
        Coordinates:
        year_month : object
            Time index, month & year. 
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers. 
    """

    def get_mb_regression(mb, h, gl_mask_1, r_win, c_win):
        
        """
        Find regression coefficients a and b in the 
        equation y = a + bx. First find b by regression to 
        current cell and and adjoining cells (3-9 cells in total).
        Then find a by using y and h in current cell and
        a = y - bx.

        Parameters
        ----------
        mb : np.array
            Array with values on which to perform regression (y).
        h : np.array
            Array with x variable.
        gl_mask_1: np.array
            Array with glacier mask to determine cells part of 
            glacierized area.
            
        Returns
        -------
        a : np.array
            Array of regression coefficient a.
        b : np.array
            Array of regression coefficient b.
        """
        rows, cols = gl_mask_1.shape
        indices = np.arange(0, rows * cols).reshape(rows, cols)
        indices[gl_mask_1 < 1] = 0
        indices_flat = indices[indices != 0]
        mb_flat = mb.flatten()
        h_flat = h.flatten()
        b = np.empty((mb_flat.shape))
        b.fill(np.nan)
        a = np.empty((mb_flat.shape))
        a.fill(np.nan)

        for i in range(0, len(indices_flat)):
            idx = indices_flat[i]
            indices_sub = np.array([idx-1, idx, idx+1, (idx+cols-1),(idx+cols),(idx+cols+1), (idx-cols-1),(idx-cols),(idx-cols+1)])
            indices_sub = indices_sub[np.isin(indices_sub, indices_flat)]
            mb_sub = mb_flat[indices_sub]
            h_sub = h_flat[indices_sub]
            res = stats.linregress(h_sub, mb_sub)
            b[idx] = res.slope
            a[idx] = mb_flat[idx] - b[idx] * h_flat[idx]

        b = b.reshape(rows,cols)
        a = a.reshape(rows,cols)

        return a, b

    def pad_regression(arr, r_win, c_win):
        rows, cols = arr.shape
            
        for i in range(2):
            for y in range(0, rows):
                
                ymin = max(0, y - r_win)
                ymax = min(rows, y + r_win + 1)
                
                for x in range(0, cols):
                    
                    xmin = max(0, x - c_win)
                    xmax = min(cols, x + c_win + 1)
                    check_mask = arr[y, x]
                    
                    if np.isnan(check_mask):
                        
                        arr_sub = arr[ymin:ymax, xmin:xmax]
                        
                        if np.sum(~np.isnan(arr_sub)) >= 3:
                            
                            arr[y,x] = np.mean(arr_sub[~np.isnan(arr_sub)])
        return arr
    
    def downscale_coeff(arr, Y_src, X_src, Y_dest, X_dest, scaling):
            
        x_src = np.arange(0, len(X_src), 1)
        y_src = np.arange(0, len(Y_src), 1)
        x_dest = np.arange(0, len(X_src), coarseness)
        y_dest = np.arange(0, len(Y_src), coarseness)

        f = interp2d(x_src, y_src, arr, kind='linear')

        arr_interp = f(x_dest, y_dest)

        return(arr_interp)    
    
    print('starting_downscaling')
    
    # Get coarse elevation.
    elev = ds_dem.elevation.values    
    #elev[elev==0] = np.nan
    
    # Get coarse glacier mask.
    mask_glacierized_area = np.array(ds_dem.glacier_fraction)
    
    # Get coarse mask of ones for cells part of glacier, zero otherwise.
    mask_glacier_1_gridded = mask_glacierized_area.copy()
    mask_glacier_1_gridded[mask_glacier_1_gridded>0]=1
    
    # Get high resolution elevation. 
    elev_h = np.array(ds_dem_hr.elevation.values)
    gl_frac_h = np.array(ds_dem_hr.glacier_fraction.values)
    gl_frac_h[gl_frac_h == 0] = np.nan
    
    # Get source and destination coordinates.
    Y_c = ds_dem.Y.values
    X_c = ds_dem.X.values
    Y_h = ds_dem_hr.Y.values
    X_h = ds_dem_hr.X.values
    
    # Get source and destination cellsize.
    cellsize_c = ds_dem.res
    cellsize_h = ds_dem_hr.res
    coarseness = cellsize_h/cellsize_c
    
    # Set window shape for padding and regression.
    shape_win = (3, 3)
    row_win = np.floor(shape_win[0] / 2).astype(int)
    col_win = np.floor(shape_win[1] / 2).astype(int)
    
    # Create empty array to fill high resolution monthly mass balance.
    monthly_mb_h = np.empty((len(da_monthly_mb.year_month), len(Y_h), len(X_h)))
    monthly_mb_h.fill(np.nan)
    
    for i in range(0, len(da_monthly_mb.year_month)):
        
        # Get coarse resolution gridded monthly mass balance.
        mb_c = da_monthly_mb[i,:,:].values
        
        # We have a coarse resolution elevation grid (Y1,X1). 
        # For array mb_c together with elevation (Y1,X1) we compute the
        # regression slope b_mb_c and intercept a_mb_c in each cell that is glacier covered
        # Arrays contain regression parameters in cells that are glacier-covered,
        # otherwise nan. 
        a_mb_c, b_mb_c = get_mb_regression(mb_c, elev, mask_glacier_1_gridded, row_win, col_win)
        
        # Pad regression parameters 
        # Padded twice.
        a_mb_c = pad_regression(a_mb_c, row_win, col_win)
        b_mb_c = pad_regression(b_mb_c, row_win, col_win)
    
        # Replace Nan with zeros.
        a_mb_c[np.isnan(a_mb_c)] = 0
        b_mb_c[np.isnan(b_mb_c)] = 0
        
        # Downscale coefficients to high-resolution grid.    
        a_mb_h = downscale_coeff(a_mb_c, Y_c, X_c, Y_h, X_h, coarseness)
        b_mb_h = downscale_coeff(b_mb_c, Y_c, X_c, Y_h, X_h, coarseness)
    
        # Calculate high resolution mass balance in each cell.
        mb_h = a_mb_h + b_mb_h * elev_h
    
        # Multiply with glacier mask to mask cells not part of domain with nan.
        mb_h = mb_h * gl_frac_h
        
        # Store in array.
        monthly_mb_h[i,:,:] = mb_h
    
    # Convert to DataArray.
    da_monthly_mb_h = xr.DataArray(monthly_mb_h,
                                 coords= {'year_month': da_monthly_mb.year_month.values,
                                          'Y': Y_h,
                                          'X': X_h},
                                 dims=["year_month", "Y", "X"],
                                 attrs={'Name': 'Gridded monthly mass balance',
                                        'res': cellsize_h,
                                        'dim': 'mm w.e.'})
    
    # Example plot 
    # da_monthly_mb_h[0,:,:].plot(cmap=plt.cm.RdBu, #vmin = -1000, vmax = 1000,
    #                           cbar_kwargs={'label':'Monthly mass balance (mm w.e.)'})
    # plt.xlabel('$UTM_{east}$')
    # plt.ylabel('$UTM_{north}$')
    # plt.title('Jan 1960')
    
    return da_monthly_mb_h

# End of function downscale_mb_grid()

#%% Function get_distributed_mass_balance()
def get_distributed_mass_balance(da_acc, ds_dem):   

    """
    The function get_distributed_mass_balance() calculates the distributed 
    (in each cell) annual surface mass balance in a glacierized area based on 
    a DataArray of distributed accumulation and a DataArray of the glacier 
    fraction in each cell.
    
    Winter balance is calculated as the difference between the maximum 
    accumulation over the specific glacier area during the current 
    hydrological year (01.10. in year-1 until 30.09. in year) and the minimum 
    accumulation over the specific glacier during the past ablation season 
    (01.04. in year-1 until 30.09. in year-1).
    
    Summer balance is calculated as the difference between the minimum 
    accumulation over the specific glacier area during the current ablation 
    season (01.04. in year until 30.09. in year) and the maximum accumulation 
    over the specific glacier area during the current hydrological year 
    (01.10. in year-1 until 30.09. in year). 
    
    The function uses the glacier fraction variable in ds_dem to determine
    the fraction of glacier in each cell. The distributed annual surface mass
    balance is calculated for each cell that is part of the glacierized area
    as the sum of accumulation and ablation in each individual cell over the 
    course of a hydrological year.
    
    Parameters
    ----------
    da_acc (time,Y,X) : xarray.DataArray (float)
        DataArray containing total daily accumulation in each cell. 
        Coordinates:
        time : datetime64[ns]
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.
        Attributes:
        res : float
            Resolution of DEM (cellsize).
    ds_dem (time,Y,X) : xarray.Dataset
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

    Returns
    -------
    da_distr_ba (year,Y,X) : xarray.DataArray (float)
        DataArray of distributed annual surface mass balance for the 
        glacierized area. 
        Coordinates:
        year : int
            Time index (yearly).
        Y : float
            Y-coordinate of cell centers.
        X : float
            X-coordinate of cell centers.    
    """
    
    # Get accumulation array from file.
    #ds_acc = xr.open_dataset('C:/Users/kasj/mass_balance_model/results/test_JOB_all_sn1_210406/da_acc4.nc')
    #da_acc = ds_acc.__xarray_dataarray_variable__
    
    # Get start year, end year and the length of the simulation period from 
    # the accumulation array.
    start_year = pd.Timestamp(da_acc.time[0].values).year
    end_year = pd.Timestamp(da_acc.time[-1].values).year
    length_period = end_year-start_year+1
    
    # Empty DataArray of daily total grid accumulation.
    da_distr_ba = xr.DataArray(
        coords= {'year': range(start_year+1, end_year+1),
                 'Y': da_acc.Y,
                 'X': da_acc.X},
        dims=["year", "Y", "X"],
        attrs={'Name': 'Distributed annual mass balance',
               'res': da_acc.res})
    
    # Initialize array of balances.
    mb_ba = np.empty((length_period,1))
    mb_ba.fill(np.nan)
    
    i = 1
    
    # For each year get annual balance in each cell of the accumulation DataArray.
    for year in range(start_year+1, end_year+1):
        
        # Get start and end of period for calculation.
        start_date = dt.datetime.strptime(str(year-1) + '-03-31', '%Y-%m-%d')
        end_date = dt.datetime.strptime(str(year) + '-09-30', '%Y-%m-%d')
        hyd_yr = dt.datetime.strptime(str(year-1) + '-10-01', '%Y-%m-%d')
        
        # Get array of glacier fraction for the given mass balance year.
        mask_glacierized_area = np.array(ds_dem.glacier_fraction.sel(time = 
                                                                     hyd_yr))
        
        # Get mask of ones for cells part of glacier, zero otherwise.
        mask_gl_area_ones = mask_glacierized_area.copy()
        mask_gl_area_ones[mask_gl_area_ones>0]=1
        mask_gl_area_ones[mask_gl_area_ones<1]=np.nan

        # Crop accumulation DataArray with start and end dates of hydrological 
        # year and convert to numpy array. 
        glacier_acc = np.array(da_acc.sel(time = slice(start_date, 
                                                           end_date)).values)
        
        # Get accumulation over the specific glacier area.
        ###gl_spec_acc = np.multiply(glacier_acc,mask_glacierized_area)
        gl_spec_acc = np.multiply(glacier_acc, mask_gl_area_ones)

        # Total accumulation over the specific glacier area for each day.
        # Only used to find indices for max and min accumulation!
        glacier_acc_d = (np.nansum(gl_spec_acc, axis = (1,2)) 
                         / mask_glacierized_area.sum())
        
        # Winter balance is max accumulation over the current hydrological 
        # year minus the minimum value during the previous years ablation
        # period. Find the index of glacier_acc_d with the maximum and
        # minimum values.
        max_idx_bw = ((len(glacier_acc_d)-365) 
                      + np.argmax(glacier_acc_d[-365:None]))
        min_idx_bw = np.argmin(glacier_acc_d[None:(365-90)])
       
        # Get accumulation array for winter mass balance.
        gl_spec_acc_bw = gl_spec_acc[max_idx_bw] - gl_spec_acc[min_idx_bw]

        # Summer balance is the minimum acccumulation over this years  
        # ablation period minus the maximum value during the current 
        # hydrological year. Find the index of glacier_acc_d with the maximum
        # and minimum values.
        min_idx_bs = 365 + np.argmin(glacier_acc_d[365:None])
        max_idx_bs = ((len(glacier_acc_d)-365)
                      + np.argmax(glacier_acc_d[-365:None]))
        
        # Get accumulation array for summer mass balance.
        gl_spec_acc_bs = gl_spec_acc[min_idx_bs] - gl_spec_acc[max_idx_bs]
        
        # Annual balance is the sum of winter and summer balance.
        gl_spec_acc_ba = gl_spec_acc_bw + gl_spec_acc_bs
        
        # Annual balance over the whole glacier. 
        #mb_ba[i] = (np.nansum(gl_spec_acc_ba, axis = (0,1)) 
        #                 / mask_glacierized_area.sum())/1e3
        
        # Annual balance in m w.e.
        gl_spec_acc_ba_m = np.multiply(gl_spec_acc_ba, mask_gl_area_ones) /1e3
        
        # Add the distributed annual mass balance for the given year to the 
        # output DataArray.
        da_distr_ba.loc[year]=gl_spec_acc_ba_m
        
        # Increment counter.
        i = i+1
    
    # Return DataArray of distributed annual mass balance. The first year of 
    # simulation is not included.
    return da_distr_ba

# End of function get_distributed_mass_balance()

#%% Function combineIceThickness()

def combineIceThickness(da_ice_thickness, da_gl_spec_frac):
    
    """
    Combines ice thickness for each glacier in one array
    Interpolates between ice thickness in overlapping cells.
    """

    #%% Check if da_gl_spec_frac and ds_dem.frac give the same values.
    
    # gl_frac_tot = np.array([da_gl_spec_frac.Y.values, da_gl_spec_frac.X.values])
    # time_check = '1957'
    # year_1_frac = da_gl_spec_frac.sel(dict(time=time_check)).values
    # sum_fractions = year_1_frac.sum(axis=0).squeeze()
    # sum_total_fractions = ds_dem_mod.glacier_fraction.sel(dict(time=time_check)).values.squeeze()
    # check_sum = np.nansum(sum_fractions-sum_total_fractions) 
    # sum_total_fractions = ds_dem.glacier_fraction.sel(dict(time=time_check)).values.squeeze()

    #%% Check difference in ice thickness at start and end of modelling
    
    import numpy as np
    
    ice_thickness_start = da_ice_thickness.sel(dict(time='1957')).values
    ice_thickness_start = np.nansum(ice_thickness_start, axis=0).squeeze()
    ice_thickness_end = da_ice_thickness.sel(dict(time='2020')).values
    ice_thickness_end = np.nansum(ice_thickness_end, axis=0).squeeze()
    ice_thickness_diff = ice_thickness_end - ice_thickness_start

    x = da_ice_thickness.X.values
    y = da_ice_thickness.Y.values
    X,Y = np.meshgrid(x,y)
    
    import matplotlib.pyplot as plt
    
    figure1 = plt.figure(dpi=500)
    ax = figure1.gca()
    #cmap = plt.get_cmap('bwr')
    #figure7.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=-500, vmax=500), 
    #                                   cmap=cmap), ax=ax)
    #plt.imshow(ice_thickness_diff)
    plt.pcolormesh(X,Y,ice_thickness_diff, shading='auto', cmap='coolwarm', vmin=-50, vmax=50)
    #plt.scatter(x_utm_meas, y_utm_meas, s=1, c=point_meas, edgecolors='k', linewidths=0.1)
    cbar = plt.colorbar()
    cbar.set_label('Ice thickness difference [m]')
    
    figure2 = plt.figure(dpi=500)
    ax = figure2.gca()
    #cmap = plt.get_cmap('bwr')
    #figure7.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=-500, vmax=500), 
    #                                   cmap=cmap), ax=ax)
    #plt.imshow(ice_thickness_diff)
    plt.pcolormesh(X,Y,ice_thickness_end, shading='auto') #, cmap='coolwarm', vmin=-500, vmax=500)
    #plt.scatter(x_utm_meas, y_utm_meas, s=1, c=point_meas, edgecolors='k', linewidths=0.1)
    cbar = plt.colorbar()
    cbar.set_label('Ice thickness at end of simulation [m]')
    
# End of function combineIceThickness()

#%% End of postprocessing.py
