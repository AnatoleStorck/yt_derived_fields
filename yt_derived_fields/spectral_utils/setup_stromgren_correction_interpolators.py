

import tqdm
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from pathlib import Path
from typing import Optional
from functools import cache

@cache
def _resolve_cloudy_dir(data_dir: Optional[str]) -> Path:
    """
    Resolve directory containing CLOUDY tables.
    Tries:
      - explicit data_dir if provided
      - known fallbacks
    """
    candidates: list[Path] = []
    if data_dir:
        candidates.append(Path(data_dir))
    # Fallback onto known paths (glamdring, infinity)
    candidates.append(Path("/mnt/glacier/DATA/CLOUDY_UPDATE_APR11"))
    candidates.append(Path("/data122/cadiou/Megatron/DATA/CLOUDY_UPDATE_APR11"))

    for base in candidates:
        test_dir_1 = base / "BPASS"
        test_dir_2 = base / "MEGATRON_cloudy_models_popIII_nip"
        if test_dir_1.exists() and test_dir_2.exists():
            return base

    raise FileNotFoundError(
        "Could not locate CLOUDY tables directory. "
        "Pass data_dir=..., or place files under one of the known paths."
    )


path_cloudy_update = _resolve_cloudy_dir(None)

path_cloudy_interp_pop2 = f"{path_cloudy_update}/BPASS"
path_cloudy_interp_pop3 = f"{path_cloudy_update}/MEGATRON_cloudy_models_popIII_nip"




def load_SED_from_sim(top_dir, ngroups=8, SED_isEgy=True):
    """
    Function to load in the SED tables from a ramses simulation

    Note that when we use SED_isEgy, these files store eV/s and not phot/s
    We need to divide by the mean energy of the bin
    """
    sed_dat = np.zeros((252,8,ngroups))
    for i in range(ngroups):
        dat = np.loadtxt(f"{top_dir}/SEDtable{i+1}.list",skiprows=1)
        for j in range(8):
            if SED_isEgy:
                sed_dat[:,j,i] = dat[252*j:252*(j+1),2] / dat[252*j:252*(j+1),4]
            else:
                sed_dat[:,j,i] = dat[252*j:252*(j+1),2] 
    ages = dat[:252,0]
    # convert to Myr
    ages *= 1e3
    ages[0] += 0.1
    metal_rebin = np.log10(np.unique(dat[:,1]))
    mif = RegularGridInterpolator((np.log10(ages),metal_rebin), np.log10(sed_dat), bounds_error=False, fill_value=0)
    return ages, metal_rebin, mif







def get_cloudy_el_interpolator():
    """
    Returns a regular grid interpolator for unresolved stromgren spheres
    this run includes induced processes
    """
    import json

    top_dirs = ["zem5", "zem4", "z001", "z002", "z003", "z004",
                "z006", "z008", "z010", "z014", "z020", "z030"]
    metals = np.array([1e-5, 1e-4, 0.001, 0.002, 0.003, 0.004,
                       0.006, 0.008, 0.010, 0.014, 0.020, 0.030])

    # List of all gas phase metallicities (w/ respect to the stellar metallicity)
    O_grid = np.arange(-3, 4.1, 1.0)

    # List of all ionizing luminosities
    Q_grid = np.arange(46.5, 54.6, 1.0)

    # List of all gas densities
    D_grid = np.arange(1.0, 6.1, 1.0)

    # List of all carbon fractions (w/ respect to the O abundance)
    C_grid = np.arange(-3.0, 1.1, 1.0)

    # List of all ages
    A_grid = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6,
                       6.7, 6.8, 6.9, 7.0, 7.176, 7.3])

    # Load in the list of lines
    with open(f"{path_cloudy_interp_pop2}/line_list.dat") as f:
        lines = f.readlines()
    all_lines = [l.strip() for l in lines  if (len(l.strip()) > 0)]
    header = ["iteration"] + all_lines
    col_rename = {i: header[i] for i in range(len(header))}

    cloudy_grid = -50.0 + np.zeros((len(top_dirs),len(O_grid),len(D_grid),len(A_grid),len(Q_grid),len(C_grid),len(header)-1))

    for j,td in enumerate(top_dirs):
        for i in range(len(O_grid)):
            try:
                df = pd.read_csv(f"{path_cloudy_interp_pop2}/{td}/lines_list_{str(i).zfill(5)}.list",header=None,delimiter="\t",comment="#")
            except (NameError,pd.errors.EmptyDataError,FileNotFoundError):
                continue
            df = df.rename(columns=col_rename)
            tmp = np.log10(np.array(df[header[1:]]).reshape(len(D_grid),len(A_grid),len(Q_grid),len(C_grid),len(header)-1)+1e-50)

            # Now the array should have columns of
            # Density, Age, Q, C/O, Emission line luminosities
            cloudy_grid[j,i,:,:,:,:,:] = tmp

    # Now handle the broken models
    # Load in the dictionary of broken models
    with open(f"{path_cloudy_interp_pop2}/broken_models.json","r") as bmj:
        broken_models = json.load(bmj)

    for j,td in enumerate(top_dirs):
        for i in range(len(O_grid)):
            bm_list = broken_models[f"metal_{td}"][str(i)]
            broken_counter = 0
            for kk in range(len(D_grid)):
                for ll in range(len(A_grid)):
                    for mm in range(len(Q_grid)):
                        for nn in range(len(C_grid)):
                            # NAN out the broken models
                            if broken_counter in bm_list:
                                cloudy_grid[j,i,kk,ll,mm,nn,:] = np.nan
                            # Increment the index
                            broken_counter += 1

    # Fixes for single point failures
    cloudy_grid[cloudy_grid < -49.0] = np.nan

    mif = RegularGridInterpolator(
        (
            np.log10(metals),
            O_grid, D_grid,
            A_grid, Q_grid,
            C_grid,
        ),
        cloudy_grid, bounds_error=False, fill_value=np.nan
    )

    return mif, header[1:]


def get_cloudy_el_p3_interpolator():
    """
    Returns a regular grid interpolator for unresolved stromgren spheres
    for the Pop III calculation
    """
    # Get the densities
    dens = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # Get the corresponding masses for the models
    mod_params  = np.loadtxt(f"{path_cloudy_interp_pop3}/model_params.dat",delimiter=",",skiprows=1)
    masses = mod_params[:,0]

    # Load in the emission line table
    lines = np.load(f"{path_cloudy_interp_pop3}/popIII_emission_lines.npy")

    mif = RegularGridInterpolator((dens,masses), lines, bounds_error=False, fill_value=0.0)

    return mif













def initialize_cloudy_nebc_unresolved_p3(downsample=False,ds_nwv=5):
    """
    Returns a regular grid interpolator for unresolved stromgren spheres
    for the Pop III calculation
    """

    # Set up the wavelength grid
    lmin = 1150
    lmax = 10000
    wvls = np.arange(lmin,lmax+0.1)

    # Get the densities
    dens = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # Get the corresponding masses for the models
    mod_params  = np.loadtxt(f"{path_cloudy_update}/MEGATRON_cloudy_models_popIII_nip/model_params.dat",delimiter=",",skiprows=1)
    masses = mod_params[:,0]

    # Load in the nebular continuum
    cloudy_grid = np.load(f"{path_cloudy_update}/MEGATRON_cloudy_models_popIII_nip/reduced_neb_continuum.npy")

    # If we need to downsample
    if downsample:
        wvls_ds = pd.Series(wvls).rolling(window=ds_nwv, min_periods=1, center=True).mean()[::ds_nwv]

        # Initialize the cloudy grid
        cloudy_grid_ds = np.zeros((len(dens),len(masses),len(wvls_ds)))

        for ii in range(len(dens)):
            for jj in range(len(masses)):
                cloudy_grid_ds[ii,jj] = pd.Series(cloudy_grid[ii,jj]).rolling(window=ds_nwv, min_periods=1, center=True).mean()[::ds_nwv]

        mif_ds = RegularGridInterpolator((dens,masses), cloudy_grid_ds, bounds_error=False, fill_value=0.0)

        return mif_ds

    mif = RegularGridInterpolator((dens,masses), cloudy_grid, bounds_error=False, fill_value=0.0)

    return mif

def initialize_cloudy_nebc_unresolved(downsample=False,ds_nwv=5):
    """
    Create the interpolating function for the nebular continuum for unresolved stromgren spheres
    based on cloudy calculations
    """
    import json

    top_dirs = ["zem5", "zem4", "z001", "z002", "z003", "z004", "z006", "z008", "z010", "z014", "z020", "z030"]
    metals = np.array([1e-5, 1e-4, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010, 0.014, 0.020, 0.030])

    # List of all gas phase metallicities (w/ respect to the stellar metallicity)
    O_grid = np.arange(-3,4.1,1.0)

    # List of all ionizing luminosities
    Q_grid = np.arange(46.5,54.6,1.0)

    # List of all gas densities
    D_grid = np.arange(1.0,6.1,1.0)

    # List of all carbon fractions (w/ respect to the O abundance)
    C_grid = np.arange(-3.0,1.1,1.0)

    # List of all ages
    A_grid = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.176, 7.3])
    
    # Set up the wavelength grid
    lmin = 1150
    lmax = 10000
    wvls = np.arange(lmin,lmax+0.1)

    # Initialize the cloudy grid
    cloudy_grid = np.zeros((len(top_dirs),len(O_grid),len(D_grid),len(A_grid),len(Q_grid),len(C_grid),len(wvls)))

    # Load in the cloudy models
    print("Initializing cloudy nebular continuum grid")
    for j,td in tqdm(enumerate(top_dirs)):
        for i in range(len(O_grid)):
            try:
                df = np.load(f"{path_cloudy_update}/BPASS/{td}/reduced_neb_continuum_{str(i).zfill(5)}.npy")
            except (NameError,pd.errors.EmptyDataError,FileNotFoundError):
                continue
            tmp = df.reshape(len(D_grid),len(A_grid),len(Q_grid),len(C_grid),len(wvls))

            # Now the array should have columns of
            # stellar metallicity, gas metallicity, density, age, Q, C/O, and nebular continuum 
            cloudy_grid[j,i,:,:,:,:,:] = np.log10(tmp+1e-50)

    # Now handle the broken models
    # Load in the dictionary of broken models
    with open(f"{path_cloudy_update}/BPASS/broken_models.json","r") as bmj:
        broken_models = json.load(bmj)

    for j,td in enumerate(top_dirs):
        for i in range(len(O_grid)):
            bm_list = broken_models[f"metal_{td}"][str(i)]
            broken_counter = 0
            for kk in range(len(D_grid)):
                for ll in range(len(A_grid)):
                    for mm in range(len(Q_grid)):
                        for nn in range(len(C_grid)):
                            # NAN out the broken models
                            if broken_counter in bm_list:
                                cloudy_grid[j,i,kk,ll,mm,nn,:] = np.nan
                            # Increment the index
                            broken_counter += 1

    # Fixes for single point failures
    cloudy_grid[cloudy_grid < -49.0] = np.nan

    mif = RegularGridInterpolator((np.log10(metals),O_grid,D_grid,A_grid,Q_grid,C_grid), cloudy_grid, bounds_error=False, fill_value=0.0)

    # If we need to downsample
    if downsample:
        wvls_ds = pd.Series(wvls).rolling(window=ds_nwv, min_periods=1, center=True).mean()[::ds_nwv]

        # Initialize the cloudy grid
        cloudy_grid_ds = np.zeros((len(top_dirs),len(O_grid),len(D_grid),len(A_grid),len(Q_grid),len(C_grid),len(wvls_ds)))

        for ii in tqdm(range(len(top_dirs))):
            for jj in range(len(O_grid)):
                for kk in range(len(D_grid)):
                    for ll in range(len(A_grid)):
                        for mm in range(len(Q_grid)):
                            for nn in range(len(C_grid)):
                                cloudy_grid_ds[ii,jj,kk,ll,mm,nn] = pd.Series(cloudy_grid[ii,jj,kk,ll,mm,nn]).rolling(window=ds_nwv, min_periods=1, center=True).mean()[::ds_nwv]

        mif_ds = RegularGridInterpolator((np.log10(metals),O_grid,D_grid,A_grid,Q_grid,C_grid), cloudy_grid_ds, bounds_error=False, fill_value=0.0)

        return mif_ds

    return mif



def format_cloudy_interpolator_p3(df):
    """
    Format the data for cloudy tabular interpolation
    --> this is specifically for pop III stars
    """

    # Initialize the interpolation array
    to_interpolate = np.zeros((len(df),2))

    # Gas density
    to_interpolate[:,0] = np.array(df["nH"])
    to_interpolate[:,0][to_interpolate[:,0] < 1.0] = 1.0
    to_interpolate[:,0][to_interpolate[:,0] > 6.0] = 6.0

    # Stellar mass
    to_interpolate[:,1] = np.array(df["initial_mass"])
    to_interpolate[:,1][to_interpolate[:,1] < 1.0] = 1.0
    to_interpolate[:,1][to_interpolate[:,1] > 820.0] = 820.0

    return to_interpolate


def format_cloudy_interpolator(
        nH,
        gas_C_over_H,
        gas_O_over_H,
        gas_O_depletion,
        star_age,
        star_metal,
        star_ionLum,
):
    """
    Format the data for cloudy tabular interpolation
    """

    # Initialize the interpolation array
    to_interpolate = np.zeros((len(nH),6))

    # Stellar metallicity
    to_interpolate[:,0] = np.array(np.log10(star_metal))
    to_interpolate[:,0][to_interpolate[:,0] < -5.0] = -5.0
    to_interpolate[:,0][to_interpolate[:,0] > np.log10(0.030)] = np.log10(0.030)

    # Gas metallicity -- with respect to the stellar metallicity
    # (remember to account for depletion)
    to_interpolate[:,1] = (np.log10(gas_O_over_H) +
                           np.log10(gas_O_depletion) -
                           np.log10(star_metal/0.014))
    to_interpolate[:,1][to_interpolate[:,1] < -3.0] = -3.0
    to_interpolate[:,1][to_interpolate[:,1] > 4.0] = 4.0

    # Due to limitations with cloudy, we require stellar metallicity + gas metallicity < 0.75
    stellar_plus_gas_metal = to_interpolate[:,0] + to_interpolate[:,1]
    cloudy_crash_flag = stellar_plus_gas_metal > 0.75
    to_subtract = stellar_plus_gas_metal - 0.75
    if cloudy_crash_flag.sum() > 0:
        print(f"There are {cloudy_crash_flag.sum()} unphysical metallicities")
        to_interpolate[:,1][cloudy_crash_flag] -= to_subtract[cloudy_crash_flag] 

    # Gas density
    to_interpolate[:,2] = np.log10(nH)
    to_interpolate[:,2][to_interpolate[:,2] < 1.0] = 1.0
    to_interpolate[:,2][to_interpolate[:,2] > 6.0] = 6.0

    # Stellar age
    to_interpolate[:,3] = np.log10(np.maximum(star_age*1e6,1.0)) # Max prevents negative ages
    to_interpolate[:,3][to_interpolate[:,3] < 6.0] = 6.0
    to_interpolate[:,3][to_interpolate[:,3] > 7.3] = 7.3

    # Ionizing luminosity
    to_interpolate[:,4] = np.log10(star_ionLum)
    to_interpolate[:,4][to_interpolate[:,4] < 46.5] = 46.5
    to_interpolate[:,4][to_interpolate[:,4] > 54.5] = 54.5

    # C/O --> this is ok because we account for depletion later
    to_interpolate[:,5] = np.log10(gas_C_over_H) - np.log10(gas_O_over_H)
    to_interpolate[:,5][to_interpolate[:,5] < -3.0] = -3.0
    to_interpolate[:,5][to_interpolate[:,5] > 1.0] = 1.0

    return to_interpolate