# Generating derived fields for the nebular continuum lines module
# NOTE: for the MEGATRON simulations ran with RAMSES-RTZ

# This module will follow closely the approach of
# Harley Katz for computing nebular continuum

# Everything normally needs to be in cgs units

# TODO: Add corections for unresolved stromgren spheres
# TODO: remove some hardcoded constants and definitions

# Author: Anatole Storck

from yt import units as u
from yt.fields.field_detector import FieldDetector

from yt_derived_fields.megatron_derived_fields import chemistry_derived_fields as chem_fields

import numpy as np
import pandas as pd

import pyneb as pn

from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator


""" Physical Constants """
clight = 2.99792458e10  # [cm/s] light speed
kb = 1.38064852e-16  # [erg/K] Boltzman constant
planck = 6.626070040e-27  # [erg s] Planck's constant


# Collisional ionization rate H I
def coll_ion_H(T):
    return 5.85e-11 * np.sqrt(T) * ((1.0 + np.sqrt(T / 1e5)) ** -1) * np.exp(-157809.1 / T)


# Recombination rate H II
def recomb_ion_H(T):
    lam_HI = 315614.0 / T
    alphab = 1.269e-13 * (lam_HI**1.503) / (1.0 + (lam_HI / 0.522) ** 0.47) ** 1.923  # cm^3 s^-1
    return alphab


def get_nebular_continuum(ds, lmin=1150, lmax=10000, downsample=True, ds_nwv=5, n_batch=1000, ncpu_max=12):
    def wavelength_space(lmin=lmin, lmax=lmax, downsample=downsample, ds_nwv=ds_nwv):
        wvls = np.arange(lmin, lmax + 0.1)
        if downsample:
            wvls = pd.Series(wvls).rolling(window=ds_nwv, min_periods=1, center=True).mean()[::ds_nwv]
        return wvls

    def generate_pyneb_nebc_interp(downsample, ds_nwv):
        """
        Loads and interpolates the pyneb nebular continuum over (T, n_e, HeI_frac, HeII_frac)
        units --> erg*cm^3/s/A
        note that this only considers recombination
        """
        # Load in the precomputed tables
        try: # glamdring
            dat = np.load("/mnt/glacier/DATA/pyneb_nebc_tables/neb_continuum.npy")
        except: # infinity
            dat = np.load("/data100/cadiou/Megatron/DATA/pyneb_nebc_tables/neb_continuum.npy")

        # Set up the parameters
        temp = 10.0 ** np.arange(3.5, 4.477, 0.02)
        den = 10.0 ** np.arange(0.0, 7.0, 0.2)
        HeI_frac = np.arange(0.0, 0.081, 0.01)
        HeII_frac = np.arange(0.0, 0.021, 0.01)

        # Create the interpolatin function
        nebc_interp = RegularGridInterpolator((temp, den, HeI_frac, HeII_frac), dat)

        # downsample?
        # The pyneb data is at 1A resolution by default
        if downsample:
            wvls_ds = wavelength_space(downsample=downsample, ds_nwv=ds_nwv)

            # Initialize the downsampled grid
            dat_ds = np.zeros((dat.shape[0], dat.shape[1], dat.shape[2], dat.shape[3], len(wvls_ds)))

            for ii in range(len(temp)):
                for jj in range(len(den)):
                    for kk in range(len(HeI_frac)):
                        for ll in range(len(HeII_frac)):
                            dat_ds[ii, jj, kk, ll] = (
                                pd.Series(dat[ii, jj, kk, ll])
                                .rolling(window=ds_nwv, min_periods=1, center=True)
                                .mean()[::ds_nwv]
                            )

            # Create the interpolatin function
            nebc_interp_ds = RegularGridInterpolator((temp, den, HeI_frac, HeII_frac), dat_ds)

            return nebc_interp_ds

        return nebc_interp

    def nebc_resolved_recomb(ds):
        def _get_nebc_resolved_recomb(field, data):
            """
            Calculates the nebular continuum for each cell (note that we ignore cells with unresolved stromgren spheres)
            """

            nH = data["gas", "hydrogen_number_density"].to("cm**-3").value
            nHe = data["gas", "helium_number_density"].to("cm**-3").value

            # Get a list of cell volumes
            cell_volumes = data["gas", "volume"].to("cm**3").value

            # Get the electron number density
            electron_density = data["gas", "electron_number_density"].to("cm**-3").value

            # Get the ionized hydrogen density
            HII_density = (nH * data["gas", "hydrogen_02"]).value
            HeII_density = (nHe * data["gas", "helium_02"]).value
            HeIII_density = (nHe * data["gas", "helium_03"]).value

            # Downweight temperatures that are too high
            temperatures = data["gas", "temperature"].to("K").value
            T_rescale = np.ones(temperatures.shape)
            T_filt = temperatures > 10.0**4.46
            # Note that the Ha emissivity decreases nearly linearly with temperature emis~T^-0.94
            # So we simply scale things linearly inversely with temperature
            # At low temperatures the emissivity rises but this is likely unphysical in the simulation
            # either due to unresolved stromgren spheres or numerical diffusion of electrons
            T_rescale[T_filt] = (10.0**4.46) / temperatures[T_filt]

            if isinstance(data, FieldDetector):
                return np.zeros(nH.shape) * u.erg / u.s

            print("Note: the nebular continuum is not a single-value field, but a spectrum for each cell.")

            # Get the CIE HII fraction --> needed below to make sure cooling isn't too strong
            CIE_HII = coll_ion_H(temperatures) / (coll_ion_H(temperatures) + recomb_ion_H(temperatures))
            HI_density_alt = nH * np.minimum(nH, 1.0 - CIE_HII)

            # Get data for interpolation
            to_interpolate = np.zeros((len(nH), 4))

            to_interpolate[:, 0] = temperatures
            to_interpolate[:, 1] = electron_density
            to_interpolate[:, 2] = HeII_density / HII_density
            to_interpolate[:, 3] = HeIII_density / HII_density

            # Enforce Bounds
            # Temperature
            to_interpolate[:, 0][to_interpolate[:, 0] < 10.0**3.5] = 10.0**3.5
            to_interpolate[:, 0][to_interpolate[:, 0] > 10.0**4.46] = 10.0**4.46

            # Density
            to_interpolate[:, 1][to_interpolate[:, 1] < 1.0] = 1.0
            to_interpolate[:, 1][to_interpolate[:, 1] > 1e7] = 1e7

            # He_II
            to_interpolate[:, 2][to_interpolate[:, 2] < 0.0] = 0.0
            to_interpolate[:, 2][to_interpolate[:, 2] > 0.08] = 0.08

            # He_III --> He_III > 0.02 is unphysical for the sources we consider
            # Unless there is a SN event --> these should never dominate except in very
            # rare circumstances so we will ignore
            to_interpolate[:, 3][to_interpolate[:, 3] < 0.0] = 0.0
            to_interpolate[:, 3][to_interpolate[:, 3] > 0.02] = 0.02

            nebc_interp = generate_pyneb_nebc_interp(downsample, ds_nwv)


            ### Parallelize the interpolation step ###

            # Determine the number of wavelengths from the interpolator output
            test_shape = nebc_interp(to_interpolate[:1]).shape[-1]
            results = np.zeros((len(nH), test_shape))  # shape: (N_cells, N_wavelengths)
            print(f"Interpolating nebular continuum for {len(nH)} cells over {test_shape} wavelengths...")

            # Chunk the data for efficient parallelization
            all_c1 = [i * n_batch for i in range(1 + len(to_interpolate) // n_batch)]
            all_c2 = [c1 + n_batch for c1 in all_c1]
            all_c2[-1] = len(to_interpolate)
            # Calculate the number of CPUs to use
            n_cpus = min(
                len(all_c1), ncpu_max
            )  # Set the maximum number of CPUs to 10 (but no more than the number of batches)
            # Define the batch interpolation function
            def batch_interp(c1, c2):
                return (
                    nebc_interp(to_interpolate[c1:c2, :])
                    * cell_volumes[c1:c2, None]
                    * electron_density[c1:c2, None]
                    * HII_density[c1:c2, None]
                    * T_rescale[c1:c2, None]
                )
            from tqdm import tqdm
            # Parallelize over batches
            batch_results = Parallel(n_jobs=n_cpus)(
                delayed(batch_interp)(all_c1[i], all_c2[i]) for i in tqdm(range(len(all_c1)))
            )
            print("Interpolation complete.")
            # Fill results array
            for i, batch_result in enumerate(batch_results):
                results[all_c1[i] : all_c2[i], :] = batch_result

            # NOTE: THIS COMBINES ALL THE CELLS TOGETHER
            # nebc_spec = np.array(results).sum(axis=0)

            # RETURN THE SPECTRUM PER CELL
            nebc_spec = np.array(results) * u.erg / u.s

            return nebc_spec

        ds.add_field(
            name=("gas", "nebc_resolved_recomb"),
            function=_get_nebc_resolved_recomb,
            units="erg/s",  # NOTE: check if this or "erg/s/A"
            sampling_type="cell",
            display_name="Nebular Continuum Resolved Recombination",
        )

    def nebc_resolved_two_photon(ds):
        def _get_nebc_resolved_two_photon(field, data):
            nH = data["gas", "hydrogen_number_density"].to("cm**-3").value
            temperatures = data["gas", "temperature"].to("K").value

            cell_volumes = data["gas", "volume"].to("cm**3").value
            electron_density = data["gas", "electron_number_density"].to("cm**-3").value

            nebc = pn.Continuum()
            wvls = wavelength_space(downsample=downsample, ds_nwv=ds_nwv)
            two_photon_generic = nebc.two_photon(1e4, 1, wvls)
            two_phot_erg_s = np.trapz(two_photon_generic, wvls)

            # Get the CIE HII fraction --> needed below to make sure cooling isn't too strong
            CIE_HII = coll_ion_H(temperatures) / (coll_ion_H(temperatures) + recomb_ion_H(temperatures))
            HI_density_alt = nH * np.minimum(nH, 1.0 - CIE_HII)

            if isinstance(data, FieldDetector):
                return np.zeros(nH.shape) * u.erg / u.s

            print("Note: the two-photon continuum is not a single-value field, but a spectrum for each cell.")

            # Note that the previous code only considered recombination
            # Here we now calculate the collisional contribution to the two-photon emission
            # only from resolved cells

            # Note that these polynomial fits come from
            # https://ui.adsabs.harvard.edu/abs/2022MNRAS.517....1S/abstract
            a0 = 0.267486
            a1 = 1.57257
            a2 = -6.44026
            a3 = 11.5401
            omega_1 = 2.0
            T6 = temperatures / 1e6
            Gamma_x = a0 + (a1 * T6) + (a2 * T6 * T6) + (a3 * T6 * T6 * T6)

            # Double check whether we should use nu_Lya here
            nu_Lya = clight * 1e8 / 1215.67  # Freq of Lya
            T12 = planck * nu_Lya / kb

            prefac = (8.63e-6) / (omega_1 * np.sqrt(temperatures))
            phi_two_phot = prefac * planck * nu_Lya * Gamma_x * np.exp(-T12 / temperatures)  # erg * cm / s
            cool_two_phot = (
                phi_two_phot * HI_density_alt * electron_density * cell_volumes
            )  # Total two photon cooling in erg/s

            two_phot_rescale = cool_two_phot / two_phot_erg_s

            # Return both the recombination spectrum and the two-photon cooling spectrum
            return np.array(two_phot_rescale)[:, None] * np.array(two_photon_generic) * u.erg / u.s

        ds.add_field(
            name=("gas", "nebc_resolved_two_photon"),
            function=_get_nebc_resolved_two_photon,
            units="erg/s",  # NOTE: check if this or "erg/s/A"
            sampling_type="cell",
            display_name="Nebular Continuum Resolved Two Photon",
        )

    # generate chemistry fields if not already present
    if not ("gas", "electron_number_density") in ds.derived_field_list:
        chem_fields.create_chemistry_derived_fields(ds)
    # Add the fields to the dataset
    nebc_resolved_recomb(ds)
    nebc_resolved_two_photon(ds)
