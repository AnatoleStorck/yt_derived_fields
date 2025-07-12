
import tqdm
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator

import yt_derived_fields.megatron_derived_fields.stars_derived_fields as stars_derived_fields

lmin=1150, lmax=10000, downsample=False, ds_nwv=5


def wavelength_space(lmin=lmin, lmax=lmax, downsample=downsample, ds_nwv=ds_nwv):
    wvls = np.arange(lmin,lmax+0.1)
    if downsample:
        wvls = pd.Series(wvls).rolling(window=ds_nwv, min_periods=1, center=True).mean()[::ds_nwv]
    return wvls

# BPASS (v2.2.1) spectra (interpolated over metallicity and age)
def generate_pop_II_spec_interp(lmin=lmin, lmax=lmax, downsample=downsample, ds_nwv=ds_nwv):
    """
    Function that loads and returns an interpolating function for the pop II spectra
    --> these are the BPASS models
    units are erg/s/A
    """
    # Set up the wavelength grid
    wvls = wavelength_space(lmin=lmin, lmax=lmax, downsample=False, ds_nwv=ds_nwv)

    metal_names = ["zem5","zem4","z001","z002","z003","z004","z006","z008","z010","z014","z020","z030","z040"]
    metal_vals = np.array([1e-5,1e-4,1e-3,2e-3,3e-3,4e-3,6e-3,8e-3,1e-2,1.4e-2,2e-2,3e-2,4e-2])

    ages = 10.**(6. + 0.1 * np.arange(51))

    # Load in the first array
    dat = np.load(f"/mnt/glacier/DATA/bpass_v2.2.1_imf_chab300/reduced_spectra-bin-imf_chab300.{metal_names[0]}.dat.npy").T

    # Setup the array that contains all of the spectral data
    n_metals = len(metal_names)
    n_ages = len(ages)
    n_wvl = len(wvls)
    all_spec = np.zeros((n_metals,n_ages,n_wvl))

    all_spec[0,:,:] = dat

    for i,m in enumerate(metal_names[1:]):
        dat = np.load(f"/mnt/glacier/DATA/bpass_v2.2.1_imf_chab300/reduced_spectra-bin-imf_chab300.{m}.dat.npy").T
        all_spec[i+1,:,:] = dat

    popII_interp = RegularGridInterpolator((metal_vals,ages),all_spec)

    # Downsample
    if downsample:
        wvls_ds = wavelength_space(lmin=lmin, lmax=lmax, downsample=downsample, ds_nwv=ds_nwv)

        # Initialize the downsampled grid
        all_spec_ds = np.zeros((all_spec.shape[0],all_spec.shape[1],len(wvls_ds)))

        for ii in range(len(metal_vals)):
            for jj in range(len(ages)):
                all_spec_ds[ii,jj] = pd.Series(all_spec[ii,jj]).rolling(window=ds_nwv, min_periods=1, center=True).mean()[::ds_nwv]

        # Interpolate the spectra over mass
        popII_interp_ds = RegularGridInterpolator((metal_vals,ages),all_spec_ds)

        return metal_vals, ages, popII_interp_ds 

    return metal_vals, ages, popII_interp


def get_pop_2_spectrum(data, n_batch=5000, ncpu_max=10):
    """
    Calculates the Population 2 spectrum
    units of erg/s/A
    """
    
    
    N_pop2 = np.sum(data["pop2", "particle_ones"])

    if N_pop2 < 1:
        return None
    
    spec_interp_p2, metals_p2, ages_p2 = generate_pop_II_spec_interp(lmin=lmin, lmax=lmax, downsample=downsample, ds_nwv=ds_nwv)

    # Get data for interpolation
    to_interp = np.zeros((N_pop2,2))
    met_O = data["pop2", "particle_metallicity_002"]
    met_Fe = data["pop2", "particle_metallicity_001"]
    to_interp[:,0] = (2.09 * met_O + 1.06 * met_Fe)
    to_interp[:,1] = data["pop2", "age"].to("yr").value

    # Enforce bounds
    to_interp[:,0][to_interp[:,0] < metals_p2.min()] = metals_p2.min()
    to_interp[:,0][to_interp[:,0] > metals_p2.max()] = metals_p2.max()

    to_interp[:,1][to_interp[:,1] < ages_p2.min()] = ages_p2.min()
    to_interp[:,1][to_interp[:,1] > ages_p2.max()] = ages_p2.max()

    # Get a list of initial masses
    initial_masses = data["pop2", "initial_mass"].to("Msun").value

    def parallel_interp(spec_interp_p2,to_interp,initial_masses):
        return (spec_interp_p2(to_interp)*initial_masses[:,None]).sum(axis=0)

    # Chunk the data for efficient parallelization
    all_c1 = [i * n_batch for i in range(1+len(to_interp)//n_batch)]
    all_c2 = [c1 + n_batch for c1 in all_c1]
    all_c2[-1] = len(to_interp)

    # Calculate the number of CPUs to use
    n_cpus = min(len(all_c1),ncpu_max) # Set the maximum number of CPUs to ncpu_max (but no more than the number of batches)

    # Get the results in parallel
    print(f"Interpolating stellar continuum for {len(to_interp)} stars")
    with tqdm(total=len(all_c1)) as progress_bar:
        def update_progress(*args):
            progress_bar.update()

        results = Parallel(n_jobs=n_cpus)(
                      delayed(parallel_interp)(
                          spec_interp_p2,
                          to_interp[all_c1[i]:all_c2[i],:],
                          initial_masses[all_c1[i]:all_c2[i]]
                      ) for i in range(len(all_c1)) if not update_progress() )


    # Convert to a numpy array
    p2_spec = np.array(results).sum(axis=0)

    return p2_spec
