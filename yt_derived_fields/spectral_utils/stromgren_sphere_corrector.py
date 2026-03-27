# First attempt to create a "replacement" scheme for the emission of gas cells which host stars whose stromgren spheres are not resolved.

# The point is that unresolved stromgren spheres will have their emission over/underestimated because the gas cell will not have the correct ionization states. So we want to replace the emission of these cells with the emission from a cloudy model with the same ionization parameter and metallicity as the star(s) in the cell.

import numpy as np
import pandas as pd

from scipy import spatial
from scipy.interpolate import RegularGridInterpolator

from yt_derived_fields.spectral_utils.setup_stromgren_correction_interpolators import (
    load_SED_from_sim,
    initialize_cloudy_nebc_unresolved,
    initialize_cloudy_nebc_unresolved_p3,
    get_cloudy_el_interpolator,
    get_cloudy_el_p3_interpolator,
)
from yt_derived_fields.spectral_utils.setup_stromgren_correction_finder import (
    get_unresolved_stromgren_stars,
    group_unresolved_strom_stars,
)
from yt_derived_fields.spectral_utils.setup_stromgren_correction_replacer import (
    get_cloudy_emission_line_luminosities,
    replace_emission,
)


# Get the initial nebular continuum + emission and stellar spectra for each gas cell.
# .
# .
# .



# Whatever change this dogshit later
downsample = True
ds_nwv = 5


# --- --- --- --- --- -----Generate the Interpolators----- --- --- --- --- --- #

cloudy_nebc_interp = initialize_cloudy_nebc_unresolved(
    downsample=downsample,ds_nwv=ds_nwv
)
cloudy_nebc_interp_p3 = initialize_cloudy_nebc_unresolved_p3(
    downsample=downsample,ds_nwv=ds_nwv
)

mif_cloudy,line_list = get_cloudy_el_interpolator()
mif_cloudy_p3 = get_cloudy_el_p3_interpolator()

age_bins, metal_bins, mif = load_SED_from_sim(
    top_dir="/mnt/glacier/DATA/SEDtables", ngroups=8, SED_isEgy=True
)

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #

ds = None

df_gas = None # is just going to be some fields of ("gas", <field>)
df_stars = None # is just going to be some fields of ("stars", <field>)

df_emis = None # is just going to be a Ncells x spectra array.

df_strom_stars = get_unresolved_stromgren_stars(
    df_gas, df_stars,
    age_bins, metal_bins, mif,
    ionizing_group=4, verbose=False
)

df_strom_stars_new = group_unresolved_strom_stars(
    df_strom_stars, df_gas["redshift"].iloc[0],
    boxsize=70, rad_multiplier=0.75
)


replace_emission_lines = get_cloudy_emission_line_luminosities(
    df_strom_stars_new,
    mif_cloudy,
    line_list,
) # maybe replacement_emission_lines has nans, who cares for now


df_emis = replace_emission(
    df_emis, df_strom_stars_new,
    replace_emission_lines, line_list,
)
