# First attempt to create a "replacement" scheme for the emission of gas cells which
# host stars whose stromgren spheres are not resolved.

# The point is that unresolved stromgren spheres will have their emission
# over/underestimated because the gas cell will not have the correct ionization states.
# So we want to replace the emission of these cells with the emission from a cloudy model
# with the same ionization parameter and metallicity as the star(s) in the cell.

import yt
import numpy as np

from yt import units as u
from yt.fields.field_detector import FieldDetector
from yt.funcs import mylog

from yt_derived_fields.spectral_utils.setup_stromgren_correction_interpolators import (
    load_SED_from_sim,
)

from pathlib import Path
from typing import Optional
from functools import cache

@cache
def _resolve_SED_dir(data_dir: Optional[str]) -> Path:
    """
    Resolve directory containing the sim SEDs.
    Tries:
      - explicit data_dir if provided
      - known fallbacks
    """
    candidates: list[Path] = []
    if data_dir:
        candidates.append(Path(data_dir))
    # Fallback onto known paths (glamdring, infinity)
    candidates.append(Path("/mnt/glacier/DATA/SEDtables"))
    candidates.append(Path("/data122/cadiou/Megatron/DATA/SEDtables"))

    for base in candidates:
        test_file = base / "SEDtable1.list"
        if test_file.exists():
            return base

    raise FileNotFoundError(
        "Could not locate SED sim directory. "
        "Pass data_dir=..., or place files under one of the known paths."
    )


def stromgren_correction_pipeline(ds):
    """
    This is the main function which will run the stromgren correction pipeline.
    It will return a dataframe with the corrected emission line luminosities for
    each gas cell.
    """

    def young_pop2_filter(field, data):
        
        ages = data["pop2", "age"].in_units("Myr")

        return ages < 10.0 * u.Myr
    yt.add_particle_filter(
        "young_pop2",
        function=young_pop2_filter,
        requires=["age"],
        filtered_type="star",
    )
    if "young_pop2" not in ds.filtered_particle_types:
        ds.add_particle_filter("young_pop2")

    def star_ion_lums(field, data):

        Nstars = data["young_pop2", "particle_ones"].sum().d

        ages = data["young_pop2", "age"].in_units("Myr").d
        masses = data["young_pop2", "particle_initial_mass"].in_units("Msun").d
        metal = data["young_pop2", "metallicity"]

        if isinstance(data, FieldDetector):
            return np.zeros(metal.shape) * u.erg / u.s

        # get the interpolation matrix
        # (age, metal) --> ionizing luminosity (erg/s) # TODO: check units
        sed_path = _resolve_SED_dir(None)
        age_bins, metal_bins, mif = load_SED_from_sim(
            top_dir=sed_path, ngroups=8, SED_isEgy=True
        )

        pp = np.zeros((int(Nstars), 2))
        loc_ages = ages
        loc_ages[loc_ages < 1.e-3] = 1e-3           # floor the ages
        pp[:,0] = np.log10(loc_ages)                # Note that this prevents nans
        pp[:,1] = np.log10(metal + 1.e-40)          # eqn in rt_spectra 

        # Enforce bounds
        pp[:,0][pp[:,0] < np.log10(age_bins)[0]]  = np.log10(age_bins)[0]
        pp[:,0][pp[:,0] > np.log10(age_bins)[-1]] = np.log10(age_bins)[-1]
        pp[:,1][pp[:,1] < metal_bins[0]]  = metal_bins[0]
        pp[:,1][pp[:,1] > metal_bins[-1]] = metal_bins[-1]

        ion_lums = 10.0**mif(pp) * masses[:, np.newaxis]

        # Take group 4 (13.6-24.6 eV) which is the relevant one for H ionization
        ion_lums = ion_lums[:, 4]

        return ion_lums * u.erg / u.s # TODO: check units

    ds.add_field(
        name=("young_pop2", "ionizing_luminosity"),
        function=star_ion_lums,
        units="erg/s",
        sampling_type="particle",
        display_name="Young Pop. II Star Ionizing Luminosity"
    )

    deposit_ionLum = ds.add_deposited_particle_field(
       ("young_pop2", "ionizing_luminosity"), method="sum"
   )


    def is_stromgren_unresolved(field, data):

        ion_lums = data[deposit_ionLum].in_units("erg/s").d

        nH = data["gas", "hydrogen_number_density"].in_units("cm**-3").d
        xHI = data["gas", "hydrogen_01"].d

        dx = data["gas", "dx"].in_units("pc")

        # Recombination rate
        HII_temp = 1e4
        lam_HI = 315614.0/HII_temp
        alphab = (1.269e-13 * (lam_HI**1.503) /
                  (1. + (lam_HI/0.522)**0.47)**1.923) # cm^3 s^-1

        nHI = nH * xHI

        r_strom = np.cbrt((3.0 * ion_lums) / (4.0 * np.pi * nHI**2 * alphab)) * u.cm

        bool_arr = 2 * r_strom.to("pc") < dx

        return bool_arr

    ds.add_field(
        name=("gas", "unresolved_stromgren"),
        function=is_stromgren_unresolved,
        units="dimensionless",
        sampling_type="cell",
        display_name="Stromgren Sphere is Unresolved"
    )

    deposit_age_wIonLum = ds.add_deposited_particle_field(
       ("young_pop2", "age"),
       method="weighted_mean", weight_field="ionizing_luminosity"
   )
    deposit_metallicity_wIonLum = ds.add_deposited_particle_field(
       ("young_pop2", "metallicity"),
       method="weighted_mean", weight_field="ionizing_luminosity"
   )


