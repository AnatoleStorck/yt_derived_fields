import numpy as np
import pandas as pd

from scipy.interpolate import RegularGridInterpolator

import unyt as u


def wavelength_space(lmin, lmax, downsample, ds_nwv):
    wvls = np.arange(lmin, lmax + 0.1)
    if downsample:
        wvls = pd.Series(wvls).rolling(window=ds_nwv, min_periods=1, center=True).mean()[::ds_nwv]
    return wvls


# Larkin et al. (2023) spectra (interpolated over mass)
def generate_pop_III_spec_interp(lmin, lmax, downsample, ds_nwv):
    """
    Function that loads and returns an interpolating function for the pop III spectra
    Units are erg/s/A
    """
    # Load in the data and spectrum properties
    dat = np.load("/mnt/glacier/DATA/Pop_III_spectra/reduced_popiii_spec.npy")
    props = pd.read_csv("/mnt/glacier/DATA/Pop_III_spectra/model_params.dat")

    # Interpolate the spectra over mass
    popIII_interp = RegularGridInterpolator((np.array(props["Mass_Msol"]),), dat)

    if downsample:
        wvls_ds = wavelength_space(lmin, lmax, downsample, ds_nwv)

        # Initialize the downsampled grid
        dat_ds = np.zeros((len(dat), len(wvls_ds)))

        for ii in range(len(dat)):
            dat_ds[ii] = pd.Series(dat[ii]).rolling(window=ds_nwv, min_periods=1, center=True).mean()[::ds_nwv]

        # Interpolate the spectra over mass
        popIII_interp_ds = RegularGridInterpolator((np.array(props["Mass_Msol"]),), dat_ds)

        return popIII_interp_ds

    return popIII_interp


def get_pop_3_spectrum(data, combined=True, lmin=1150, lmax=10000, downsample=True, ds_nwv=5):
    """
    Calculates the Population 3 spectrum
    units of erg/s/A
    """

    pop3_alive_status = data["pop3", "isAlive"]

    if not np.any(pop3_alive_status):
        return np.zeros_like(wavelength_space(lmin, lmax, downsample, ds_nwv))

    # Get a list of active Pop III masses
    active_popIII_masses = data["pop3", "particle_initial_mass"][pop3_alive_status].to("Msun").value

    # Enforce bounds
    active_popIII_masses[active_popIII_masses < 1.0] = 1.0
    active_popIII_masses[active_popIII_masses > 820.2] = 820.2

    spec_interp_p3 = generate_pop_III_spec_interp(lmin, lmax, downsample, ds_nwv)

    p3_spec = spec_interp_p3(active_popIII_masses)
    if combined:
        p3_spec = p3_spec.sum(axis=0)

    return p3_spec * u.erg / u.s
