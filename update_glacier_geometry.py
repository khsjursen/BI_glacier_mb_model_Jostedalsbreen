# -*- coding: utf-8 -*-
"""
Created on Thu May 20 09:42:39 2021

@author: kasj

-------------------------------------------------------------------
Mass-balance model
-------------------------------------------------------------------

SCRIPT DESCRIPTION - UPDATE!
Script to calculate glacier geometry chagnes from mass changes.

"""

#%% Libraries

# Standard libraries

# External libraries
import numpy as np
import xarray as xr
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.plot import show
from rasterio import Affine as A

# Internal libraries

#%% Function deltah_parameterization()

def deltah_parameterization(ds_dem, da_gl_spec_frac, da_ice_thickness, 
                           da_mass_balance, glacier_id, yr, rho_water, 
                           rho_ice):
    
    """
    The function deltah_parameterization() converts glacier mass change at the
    end of a hydrological year to a change in glacier geometry following the 
    delta-h parameterization of Huss et al. (2010).
    
    Glacier mass change for each glacier ID is converted to a distributed ice 
    thickness change according to the redistribution curves by Huss et al. 
    (2010),
        
    delta_h = (h_r + a)^sigma + b*(h_r + a) + c,
    
    where delta_h is the normalized surface elevation change and h_r is the 
    normalized elevation range,

    h_r = (h_max − h)/(h_max − h_min).
    
    The variables h_max and h_min are the maximum and minimum glacier surface 
    elevation, retreived from the surface DEM in combination with a glacier 
    mask. The parameters of the redistribution curve is retreived from Huss et 
    al.(2010) according to three different glacier size classes:
        
    1) Large valley glaciers (A > 20km2)    
        delta_h = (h_r - 0.02)^6 + 0.12*(h_r - 0.02)
    2) Medium-sized valley glaciers (5km2 < A < 20km2)
        delta_h = (h_r - 0.05)^4 + 0.19*(h_r - 0.05) + 0.01
    3) Small glaciers (A < 5km2)
        delta_h = (h_r - 0.30)^2 + 0.60(h_r - 0.30) + 0.09
    
    h_r and delta_h is calculated for each cell that is part of the glacier.
    These are used to find the scaling factor that scales the normalized ice
    thickness change to the volume change,
    
    f_s = ∆V_tot / SUM(delta_h_cell ∙ A_cell).
    
    The volume change in each cell is then calculated as,
    
    ∆V_cell = f_s ∙ delta_h_cell ∙ A_cell
    
    The new ice thickness is determined from the current ice volume in each
    cell and the change in volume in each cell. In cells where the new ice 
    thickness is negative, the ice thickness is set to zero and the excess 
    volume is recorded. The surface elevation is updated according to the 
    change in ice thickness and the fraction of glacier coverage in cells 
    with zero ice thickness is set to zero.
    
    Any excess volume as a result of negative ice thickness is redistributed
    over all remaining glacier cells according to the procedure described
    above, until the excess volume is zero. 
    
    Note that the function currently only calculates glacier retreat, not
    glacier advance!
    
    Possible improvements:
        - Implement glacier advance. How to set criteria for advance? Advance
        in an entire cell or fraction? 
        - Problem with initializing ice thickness from Farinotti ice thickness
        and hoydedata DEM.
        - See also Huss & Hock (2015) and Rounce et al. (2020) for 
        calculations, including a method for determining glacier advance.
        - Calibrate parameters in redistribution curves to ice cap outlet
        glaciers of different sizes?
    
    Parameters
    ----------
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
    da_ice_thickness (BREID,time,Y,X) : xarray.DataArray
        DataArray of ice thickness for each glacier specified by BREID
        (glacier ID).
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
    da_mass_balance (BREID,year,balance) : xarray.DataArray (float)
        DataArray of summer, winter and annual surface mass balance (given by 
        coordinate 'balance') for each glacier specified by BREID (glacier ID)
        and year. Mass balances are calculated at the end of each hydrological
        year
        Coordinates:
        BREID : int
            Glacier IDs.
        year : int
        balance : str
            Summer (Bs), winter (Bw) and annual (Ba) for the given glacier and
            year. 
    yr : int
        Current year. New glacier geometry calculated for the hydrological
        year starting at 01-10-yr.

    Returns
    -------
    gl_frac_updated (BREID,Y,X) : numpy array 
        Updated glacier fraction for each glacier ID.
    gl_ice_thickness_updated (BREID,Y,X) : numpy array 
        Updated ice thickness for each glacier ID. 
    elev_updated (Y,X) : numpy array    
        Updated surface elevation of the area after change in ice thickness.
    """
    
    # Get resolution of grid.
    cellsize = ds_dem.res
    
    # Get the current glacier fraction, surface elevation and ice thickness 
    # (determined at the end of the previous hydrological year). 
    old_gl_frac = da_gl_spec_frac.sel(time = str(yr-1) + '-10-01')
    old_elev = ds_dem.elevation.sel(time = str(yr-1) + '-10-01')
    old_ice_thickness = da_ice_thickness.sel(time = str(yr-1) + '-10-01')
    old_tot_gl_frac = ds_dem.glacier_fraction.sel(time = str(yr-1) + '-10-01')
    
    # Get the mass balance at the end of the current hydrological year for all
    # glaciers.
    ###mb_mwe = da_mass_balance.sel(dict(year = yr, balance = 'Ba'))
    mb_mwe = da_mass_balance.loc[da_mass_balance['Year'] == yr, ['BREID','Ba']]
        
    # Intitalize arrays for updated glacier fraction and ice thickness.
    gl_frac_updated = np.zeros(old_gl_frac.shape)
    gl_ice_thickness_updated = np.zeros(old_gl_frac.shape)
    
    # Initialize array of updated surface elevation and total glacier fraction
    # as the old surface elevation and glacier fraction.
    elev_updated = np.array(old_elev.values)
    tot_gl_frac_updated = np.array(old_tot_gl_frac.values)

    for i in range(0, len(glacier_id)):
        
        # Get current glacier specific fraction for the given glacier ID.
        gl_frac = np.array(old_gl_frac.loc[dict(BREID=glacier_id[i])].values)
        
        # Subtract the glacier fraction for the given glacier ID from the
        # total glacier fraction. Each cell in tot_frac_updated contains the
        # sum of fractions for each glacier and for each glacier the fraction
        # of this glacier needs to be subtracted from the total glacier
        # fraction and then added again after the new glacier fraction of each
        # individual glacier is updated.
        tot_gl_frac_updated = tot_gl_frac_updated - gl_frac
        
        # Create mask of ones in cells part of the glacier, nan in cells that
        # are not part of the glacier. Cells that are not part of the glacier
        # need to be nan, otherwise the minimum elevation will be outside the 
        # glacier area.
        gl_mask_ones = gl_frac.copy()
        gl_mask_ones[gl_mask_ones == 0] = np.nan
        with np.errstate(invalid='ignore'): # Ignore warning when comparing nan
            gl_mask_ones[gl_mask_ones > 0] = 1
        
        # Get elevation of cells part of the current glacier.
        gl_elev = np.array(old_elev.values * gl_mask_ones)
        
        # Get current glacier ice thickness [m ice]
        gl_ice_thickness = np.array(old_ice_thickness.loc[dict(BREID=glacier_id[i])].values)

        # Get current glacier area [km3].
        gl_area = np.sum(gl_frac * cellsize**2) / 1e6
        
        # Glacier-wide volume change.
        gl_mb_mwe = mb_mwe.loc[mb_mwe['BREID'] == glacier_id[i],'Ba'].values # meter water eq
        #gl_mb_mwe = mb_mwe.at[glacier_id[i],'Ba'] # meter water eq

        ###gl_mb_mwe = mb_mwe.loc[dict(BREID=glacier_id[i])]
        gl_mb_mie = gl_mb_mwe * (rho_water / rho_ice) # meter ice eq
        delta_V_tot = gl_mb_mie * (float(gl_area) * 1e6) # m3 ice
        
        # Glacier retreat:
        # If the new ice thickness is negative in any cells, set the new ice
        # thickness to zero in these cells and redistribute the remaining 
        # volume across the remaining cells of the glacier. 
        while abs(delta_V_tot) > 0:
            
            print(glacier_id[i])
            
            # Normalize elevation by elevation range: 
            # h_r = (h_max − h)/(h_max − h_min)
            # Cell with maximum elevation will have value 0, and cell with 
            # minimum elevation will have value 1.
            h_r = ((np.nanmax(gl_elev) - gl_elev) 
                   / (np.nanmax(gl_elev) - np.nanmin(gl_elev)))
        
            # Get normalized surface elevation change (delta_h) for the given
            # glacier size. Parameters of the delta_h parameterization are from 
            # Huss et al. (2010). 
            if gl_area >= 20: # Large valley glaciers
                print('large')
                delta_h = (h_r - 0.02)**6 + 0.12*(h_r - 0.02)
            elif gl_area < 20 and gl_area >= 5: # Medium-sized valley glaciers
                print('medium')
                delta_h = (h_r - 0.05)**4 + 0.19*(h_r - 0.05) + 0.01
            else: # Small glaciers
                print('small')
                delta_h = (h_r - 0.30)**2 + 0.60*(h_r - 0.30) + 0.09
                # Note! delta_h is 1 where h_r is 1 and 0 where h_r is 0... 
    
            # Compute the scaling factor:
            # f_s = ∆Vtot / SUM(∆hcell ∙ Acell)
            f_s = delta_V_tot / np.nansum(delta_h * gl_frac * cellsize**2)
    
            # Calculate the volume change in each cell:
            # ∆Vcell = f_s ∙ ∆hcell ∙ Acell
            delta_V = f_s * delta_h * gl_frac * cellsize**2 # m3 ice
    
            # Calculate new ice thickness in each cell from the volume change 
            # in each cell.
            gl_ice_thickness_new = (gl_ice_thickness 
                                    + delta_V / (gl_frac * cellsize**2)) # m3 ice

            # Get the remaining volume from any cells that have a negative ice
            # thickness after redistribution.
            V_new = gl_ice_thickness_new * gl_frac * cellsize**2 # m3 ice
            with np.errstate(invalid='ignore'): # Remove warning when comparing nan
                V_excess_tot = -np.sum(V_new[V_new<0]) # m3 ice
            
            # Set glacier area to zero in cells where all ice has melted.
            gl_frac_new = gl_frac.copy()
            with np.errstate(invalid='ignore'): # Ignore warning when comparing nan
                gl_frac_new[gl_ice_thickness_new <= 0] = 0
            
            # Set ice thickness to zero in cells where all ice has melted.
            with np.errstate(invalid='ignore'): # Ignore warning when comparing nan
                gl_ice_thickness_new[gl_ice_thickness_new <= 0] = 0
        
            # Calculate the updated elevation before redistribution as the
            # sum of the old surface elevation and the difference between the
            # new and old ice thickness.
            gl_elev_new = gl_elev + (gl_ice_thickness_new - gl_ice_thickness)
            
            # Update array of overall surface elevation with new elevation in 
            # cells that are part of the glacier (including cells where ice 
            # thickness now is zero).
            elev_updated[gl_mask_ones == 1] = gl_elev_new[gl_mask_ones == 1]
            
            # Update glacier mask with nan in cells that are no longer part of
            # the glacier.
            gl_mask_ones[gl_frac_new == 0] = np.nan
            
            # Mask out cells in elevation array that are now not part of the 
            # glacier. This is necessary to find the minimum and maximum 
            # elevation of the glacier in the next iteration.
            gl_elev_new = np.multiply(gl_elev_new, gl_mask_ones)
            
            # The total volume to be redistributed is the excess volume. 
            delta_V_tot = V_excess_tot
            
            # Update values if ice is to be redistributed.
            if abs(delta_V_tot) > 0:
                
                gl_ice_thickness = gl_ice_thickness_new
                gl_elev = gl_elev_new
                gl_frac = gl_frac_new
                gl_area = np.sum(gl_frac * cellsize**2) / 1e6

        # End of while loop.                 

        # Add new glacier thickness and fraction for glaicer i.
        gl_frac_updated[i,:,:] = gl_frac_new
        gl_ice_thickness_updated[i,:,:] = gl_ice_thickness_new
        
        # Update the total glacier fraction. 
        tot_gl_frac_updated = tot_gl_frac_updated + gl_frac_new
    
    # Return the new glacier fraction for each glacier, the new ice thickness
    # for each glacier and the updated surface elevation for the entire area.
    return gl_frac_updated, gl_ice_thickness_updated, elev_updated, tot_gl_frac_updated

#%% End of function deltah_parameterization()

#%% Function deltah_w_downscaling()

def deltah_w_downscaling(ds_dem, ds_dem_hres, da_gl_spec_frac_hres, da_ice_thickness, 
                         mb, glacier_id, yr, rho_water, rho_ice):
    
    """
    The function deltah_parameterization() converts glacier mass change at the
    end of a hydrological year to a change in glacier geometry following the 
    delta-h parameterization of Huss et al. (2010).
    
    Glacier mass change for each glacier ID is converted to a distributed ice 
    thickness change according to the redistribution curves by Huss et al. 
    (2010),
        
    delta_h = (h_r + a)^sigma + b*(h_r + a) + c,
    
    where delta_h is the normalized surface elevation change and h_r is the 
    normalized elevation range,

    h_r = (h_max − h)/(h_max − h_min).
    
    The variables h_max and h_min are the maximum and minimum glacier surface 
    elevation, retreived from the surface DEM in combination with a glacier 
    mask. The parameters of the redistribution curve is retreived from Huss et 
    al.(2010) according to three different glacier size classes:
        
    1) Large valley glaciers (A > 20km2)    
        delta_h = (h_r - 0.02)^6 + 0.12*(h_r - 0.02)
    2) Medium-sized valley glaciers (5km2 < A < 20km2)
        delta_h = (h_r - 0.05)^4 + 0.19*(h_r - 0.05) + 0.01
    3) Small glaciers (A < 5km2)
        delta_h = (h_r - 0.30)^2 + 0.60(h_r - 0.30) + 0.09
    
    h_r and delta_h is calculated for each cell that is part of the glacier.
    These are used to find the scaling factor that scales the normalized ice
    thickness change to the volume change,
    
    f_s = ∆V_tot / SUM(delta_h_cell ∙ A_cell).
    
    The volume change in each cell is then calculated as,
    
    ∆V_cell = f_s ∙ delta_h_cell ∙ A_cell
    
    The new ice thickness is determined from the current ice volume in each
    cell and the change in volume in each cell. In cells where the new ice 
    thickness is negative, the ice thickness is set to zero and the excess 
    volume is recorded. The surface elevation is updated according to the 
    change in ice thickness and the fraction of glacier coverage in cells 
    with zero ice thickness is set to zero.
    
    Any excess volume as a result of negative ice thickness is redistributed
    over all remaining glacier cells according to the procedure described
    above, until the excess volume is zero. 
    
    Note that the function currently only calculates glacier retreat, not
    glacier advance!
    
    Possible improvements:
        - Implement glacier advance. How to set criteria for advance? Advance
        in an entire cell or fraction? 
        - Problem with initializing ice thickness from Farinotti ice thickness
        and hoydedata DEM.
        - See also Huss & Hock (2015) and Rounce et al. (2020) for 
        calculations, including a method for determining glacier advance.
        - Calibrate parameters in redistribution curves to ice cap outlet
        glaciers of different sizes?
    
    Parameters
    ----------
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
    da_ice_thickness (BREID,time,Y,X) : xarray.DataArray
        DataArray of ice thickness for each glacier specified by BREID
        (glacier ID).
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
    da_mass_balance (BREID,year,balance) : xarray.DataArray (float)
        DataArray of summer, winter and annual surface mass balance (given by 
        coordinate 'balance') for each glacier specified by BREID (glacier ID)
        and year. Mass balances are calculated at the end of each hydrological
        year
        Coordinates:
        BREID : int
            Glacier IDs.
        year : int
        balance : str
            Summer (Bs), winter (Bw) and annual (Ba) for the given glacier and
            year. 
    yr : int
        Current year. New glacier geometry calculated for the hydrological
        year starting at 01-10-yr.

    Returns
    -------
    gl_frac_updated (BREID,Y,X) : numpy array 
        Updated glacier fraction for each glacier ID.
    gl_ice_thickness_updated (BREID,Y,X) : numpy array 
        Updated ice thickness for each glacier ID. 
    elev_updated (Y,X) : numpy array    
        Updated surface elevation of the area after change in ice thickness.
    """
    
    # Get resolution of grid.
    #cellsize = ds_dem.res
    cellsize = ds_dem_hres.res
    cellsize_c = ds_dem.res
    coarseness = int(cellsize_c/cellsize)
    
    # Get the current glacier fraction and surface elevation for the coarse grid. 
    # (determined at the end of the previous hydrological year). 
    #old_gl_frac = np.array(da_gl_spec_frac.sel(time = str(yr-1) + '-10-01').values)
    old_elev = np.array(ds_dem.elevation.sel(time = str(yr-1) + '-10-01').values)
    #old_tot_gl_frac = np.array(ds_dem.glacier_fraction.sel(time = str(yr-1) + '-10-01').values)
    
    # Get the current glacier fraction and surface elevation for the high resolution grid. 
    # (determined at the end of the previous hydrological year). 
    old_gl_frac_hres = da_gl_spec_frac_hres.sel(time = str(yr-1) + '-10-01')
    old_elev_hres = np.array(ds_dem_hres.elevation.sel(time = str(yr-1) + '-10-01').values)
    old_ice_thickness = da_ice_thickness.sel(time = str(yr-1) + '-10-01')
    old_tot_gl_frac_hres = np.array(ds_dem_hres.glacier_fraction.sel(time = str(yr-1) + '-10-01').values)
    
    # Get the mass balance at the end of the current hydrological year for all
    # glaciers.
    mb_mwe = mb.loc[mb['Year'] == yr, ['BREID','Ba']]
        
    # Intitalize arrays for updated glacier fraction and ice thickness.
    gl_frac_hres_upd = np.zeros(old_gl_frac_hres.shape)
    gl_ice_thickness_upd = np.zeros(old_gl_frac_hres.shape)
    
    # Initialize array of updated surface elevation and total glacier fraction
    # as the old surface elevation and glacier fraction.
    elev_hres_upd = old_elev_hres.copy()
    tot_gl_frac_hres_upd = old_tot_gl_frac_hres.copy()

    for i in range(0, len(glacier_id)):
        
        # Get current glacier specific fraction for the given glacier ID.
        gl_frac = np.array(old_gl_frac_hres.loc[dict(BREID=glacier_id[i])].values)
        
        # Subtract the glacier fraction for the given glacier ID from the
        # total glacier fraction. Each cell in tot_frac_updated contains the
        # sum of fractions for each glacier and for each glacier the fraction
        # of this glacier needs to be subtracted from the total glacier
        # fraction and then added again after the new glacier fraction of each
        # individual glacier is updated.
        tot_gl_frac_hres_upd = tot_gl_frac_hres_upd - gl_frac
        
        # Create mask of ones in cells part of the glacier, nan in cells that
        # are not part of the glacier. Cells that are not part of the glacier
        # need to be nan, otherwise the minimum elevation will be outside the 
        # glacier area.
        gl_mask_ones = gl_frac.copy()
        gl_mask_ones[gl_mask_ones == 0] = np.nan
        with np.errstate(invalid='ignore'): # Ignore warning when comparing nan
            gl_mask_ones[gl_mask_ones > 0] = 1
        
        # Get elevation of cells part of the current glacier.
        gl_elev = old_elev_hres * gl_mask_ones
        
        # Get current glacier ice thickness [m ice]
        gl_ice_thickness = np.array(old_ice_thickness.loc[dict(BREID=glacier_id[i])].values)

        # Get current glacier area [km3].
        gl_area = np.sum(gl_frac * cellsize**2) / 1e6
        
        # Glacier-wide volume change.
        gl_mb_mwe = mb_mwe.loc[mb_mwe['BREID'] == glacier_id[i],'Ba'].values # meter water eq
        gl_mb_mie = gl_mb_mwe * (rho_water / rho_ice) # meter ice eq
        delta_V_tot = gl_mb_mie * (float(gl_area) * 1e6) # m3 ice
        
        # Glacier retreat:
        # If the new ice thickness is negative in any cells, set the new ice
        # thickness to zero in these cells and redistribute the remaining 
        # volume across the remaining cells of the glacier. 
        while abs(delta_V_tot) > 0:
            
            print(glacier_id[i])
            
            # Normalize elevation by elevation range: 
            # h_r = (h_max − h)/(h_max − h_min)
            # Cell with maximum elevation will have value 0, and cell with 
            # minimum elevation will have value 1.
            h_r = ((np.nanmax(gl_elev) - gl_elev) 
                   / (np.nanmax(gl_elev) - np.nanmin(gl_elev)))
        
            # Get normalized surface elevation change (delta_h) for the given
            # glacier size. Parameters of the delta_h parameterization are from 
            # Huss et al. (2010). 
            if gl_area >= 20: # Large valley glaciers
                print('large')
                delta_h = (h_r - 0.02)**6 + 0.12*(h_r - 0.02)
            elif gl_area < 20 and gl_area >= 5: # Medium-sized valley glaciers
                print('medium')
                delta_h = (h_r - 0.05)**4 + 0.19*(h_r - 0.05) + 0.01
            else: # Small glaciers
                print('small')
                delta_h = (h_r - 0.30)**2 + 0.60*(h_r - 0.30) + 0.09
                # Note! delta_h is 1 where h_r is 1 and 0 where h_r is 0... 
    
            # Compute the scaling factor:
            # f_s = ∆Vtot / SUM(∆hcell ∙ Acell)
            f_s = delta_V_tot / np.nansum(delta_h * gl_frac * cellsize**2)
    
            # Calculate the volume change in each cell:
            # ∆Vcell = f_s ∙ ∆hcell ∙ Acell
            delta_V = f_s * delta_h * gl_frac * cellsize**2 # m3 ice
    
            # Calculate new ice thickness in each cell from the volume change 
            # in each cell.
            gl_ice_thickness_new = (gl_ice_thickness 
                                    + delta_V / (gl_frac * cellsize**2)) # m3 ice

            # Get the remaining volume from any cells that have a negative ice
            # thickness after redistribution.
            V_new = gl_ice_thickness_new * gl_frac * cellsize**2 # m3 ice
            with np.errstate(invalid='ignore'): # Remove warning when comparing nan
                V_excess_tot = -np.sum(V_new[V_new<0]) # m3 ice
            
            # Set glacier area to zero in cells where all ice has melted.
            gl_frac_new = gl_frac.copy()
            with np.errstate(invalid='ignore'): # Ignore warning when comparing nan
                gl_frac_new[gl_ice_thickness_new <= 0] = 0
            
            # Set ice thickness to zero in cells where all ice has melted.
            with np.errstate(invalid='ignore'): # Ignore warning when comparing nan
                gl_ice_thickness_new[gl_ice_thickness_new <= 0] = 0
        
            # Calculate the updated elevation before redistribution as the
            # sum of the old surface elevation and the difference between the
            # new and old ice thickness.
            gl_elev_new = gl_elev + (gl_ice_thickness_new - gl_ice_thickness)
            
            # Update array of overall surface elevation with new elevation in 
            # cells that are part of the glacier (including cells where ice 
            # thickness now is zero).
            elev_hres_upd[gl_mask_ones == 1] = gl_elev_new[gl_mask_ones == 1]
            
            # Update glacier mask with nan in cells that are no longer part of
            # the glacier.
            gl_mask_ones[gl_frac_new == 0] = np.nan
            
            # Mask out cells in elevation array that are now not part of the 
            # glacier. This is necessary to find the minimum and maximum 
            # elevation of the glacier in the next iteration.
            gl_elev_new = np.multiply(gl_elev_new, gl_mask_ones)
            
            # The total volume to be redistributed is the excess volume. 
            delta_V_tot = V_excess_tot
            
            # Update values if ice is to be redistributed.
            if abs(delta_V_tot) > 0:
                
                gl_ice_thickness = gl_ice_thickness_new
                gl_elev = gl_elev_new
                gl_frac = gl_frac_new
                gl_area = np.sum(gl_frac * cellsize**2) / 1e6

        # End of while loop.                 

        # Add new glacier thickness and fraction for glaicer i.
        gl_frac_hres_upd[i,:,:] = gl_frac_new
        gl_ice_thickness_upd[i,:,:] = gl_ice_thickness_new
        
        # Update the total glacier fraction. 
        tot_gl_frac_hres_upd = tot_gl_frac_hres_upd + gl_frac_new

    # Get elevation difference between old and new high resolution elevation.
    elev_diff_hres = elev_hres_upd - old_elev_hres

    # Get mean elevation difference in coarse resolution grid and update
    # coarse resolution elevation.
    da_elev_diff_coarse = xr.DataArray(elev_diff_hres, dims=['Y','X'])
    elev_diff = da_elev_diff_coarse.coarsen(Y=coarseness,X=coarseness).mean().values
    elev_upd = old_elev + elev_diff

    # Get coarse resolution grid glacier specific fraction by summing high 
    # resolution grid fractions across the coarse grid.
    da_gl_frac_coarse = xr.DataArray(gl_frac_hres_upd, dims=['BREID','Y','X'])
    gl_frac_upd = da_gl_frac_coarse.coarsen(Y=coarseness, X=coarseness).sum().values / (coarseness**2)
    gl_frac_upd = np.around(gl_frac_upd, decimals=6)

    # Get the total glacier fraction in each cell of the coarse resolution grid.
    tot_gl_frac_upd = np.sum(gl_frac_upd, axis=0)

    # Return the new glacier fraction for each glacier, the new ice thickness
    # for each glacier and the updated surface elevation for the entire area.
    return gl_frac_upd, elev_upd, tot_gl_frac_upd, gl_frac_hres_upd, elev_hres_upd, tot_gl_frac_hres_upd, gl_ice_thickness_upd

#%% End of function deltah_w_downscaling()

    
    
    
