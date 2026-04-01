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

        bool_arr = np.full_like(dx, False, dtype=bool)

        # two things to check: is r_strom > 0 (does it have stars), if not then don't flag it as unresolved.
        # If r_strom > 0, then check if it's smaller than dx/2 (i.e. unresolved)
        bool_arr[r_strom.to("pc") > 0] = r_strom.to("pc")[r_strom.to("pc") > 0] < (dx[r_strom.to("pc") > 0] / 2.0)

        return np.array(bool_arr)

    ds.add_field(
        name=("gas", "unresolved_stromgren"),
        function=is_stromgren_unresolved,
        units="",
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



from yt_derived_fields.spectral_utils.setup_stromgren_correction_interpolators import (
    format_cloudy_interpolator,
    get_cloudy_el_interpolator,
)


def get_all_dust_depletions(metal):

    elements = [
        "Iron","Oxygen","Nitrogen","Carbon","Magnesium","Neon","Silicon","Sulphur","Calcium"
    ]
    dep_solar = [
        0.01, 0.728, 0.603, 0.501, 0.163, 1.0, 0.1, 1.0, 0.003, 1.0
    ]

    # Broken powerlaw model from RR14 (consistent with Taysun's Lya feedback)
    # See Table 1 of: https://www.aanda.org/articles/aa/pdf/2014/03/aa22803-13.pdf
    # We use the XCO,Z case
    a  = 2.21
    aH = 1.00
    b  = 0.96
    aL = 3.10
    xt = 8.10
    xs = 8.69

    x = xs + metal
    x = min(x,xs+np.log10(3.0)) # Limit metallicity to 3x solar for the dust computation

    y = a + (aH * (xs - x))
    if (x <= xt):
        y = b + (aL * (xs - x))

    y = 10.0**y # This is the Gas to Dust mass ratio

    depletion_dict = {
            elements[i]: dep_solar[i] for i in range(len(elements))
            }

    for el in depletion_dict.keys():
        d = depletion_dict[el]
        depletion = 1.0 - ( (1.0 - d) * min(1.0,162.0/y) )
        depletion_dict[el] = depletion

    return depletion_dict


def rescale_emission(df, all_emission_lines, line_list):
    """
    Gets the emission line luminosities for a given cloudy interpolator
    """

    # Rescale in case we are above or below the gas metallicity bounds
    tmp = np.array(df["[O/H]"] + df["O_dep"] - np.array(np.log10(df["metallicity"]/0.014)))
    idx_oxygen = [i for i,j in enumerate(line_list) if j[0]=="O"]
    rescale = np.ones(len(df))
    rescale[tmp < -3.0] = (10.**tmp[tmp < -3.0]) / (10.**-3.0)
    rescale[tmp > 4.0] = (10.**tmp[tmp > 4.0]) / (10.**4.0)
    all_emission_lines[:,idx_oxygen] *= rescale[:, np.newaxis]

    # Rescale the carbon lines in case of bounds errors
    idx_carbon = [i for i,j in enumerate(line_list) if j[0]=="C"]
    C_over_H_cloudy = np.log10(np.array([get_all_dust_depletions(kk)["Carbon"] for kk in df["[O/H]"]+df["O_dep"]])) # Dust depletion (Carbon)
    tmp1 = np.array(df["[C/H]"] - df["[O/H]"])
    tmp1[tmp1 < -3.0] = -3.0
    tmp1[tmp1 > 1.0] = 1.0
    C_over_H_cloudy += np.log10(2.69E-04) + df["[O/H]"] + df["O_dep"] + tmp1 
    tmp = np.array(df["[C/H]"] + df["C_dep"] + np.log10(2.69E-04))
    rescale = np.array(tmp / C_over_H_cloudy)
    all_emission_lines[:,idx_carbon] *= rescale[:, np.newaxis]

    # Rescale nitrogen by deviation from solar
    idx_nitrogen = [i for i,j in enumerate(line_list) if j[0]=="N"]
    N_over_H_cloudy = np.log10(np.array([get_all_dust_depletions(kk)["Nitrogen"] for kk in df["[O/H]"]+df["O_dep"]])) # Dust depletion (Carbon)
    N_over_H_cloudy += np.log10(6.76E-05) + df["[O/H]"] + df["O_dep"]
    tmp = np.array(df["[N/H]"] + df["N_dep"] + np.log10(6.76E-05))
    rescale = np.array(10.**(tmp - N_over_H_cloudy))
    all_emission_lines[:,idx_nitrogen] *= rescale[:, np.newaxis]

    # Rescale neon by deviation from solar
    idx_neon = [i for i,j in enumerate(line_list) if j[:2]=="Ne"]
    Ne_over_H_cloudy = np.log10(np.array([get_all_dust_depletions(kk)["Neon"] for kk in df["[O/H]"]+df["O_dep"]])) # Dust depletion (Carbon)
    Ne_over_H_cloudy += np.log10(8.51E-05) + df["[O/H]"] + df["O_dep"]
    tmp = np.array(df["[Ne/H]"] + df["Ne_dep"] + np.log10(8.51E-05))
    rescale = np.array(10.**(tmp - Ne_over_H_cloudy))
    all_emission_lines[:,idx_neon] *= rescale[:, np.newaxis]

    # Rescale sulfur by deviation from solar
    idx_sulfur = [i for i,j in enumerate(line_list) if j[:2]=="S "]
    S_over_H_cloudy = np.log10(np.array([get_all_dust_depletions(kk)["Sulphur"] for kk in df["[O/H]"]+df["O_dep"]])) # Dust depletion (Carbon)
    S_over_H_cloudy += np.log10(1.32E-05) + df["[O/H]"] + df["O_dep"]
    tmp = np.array(df["[S/H]"] + df["S_dep"] + np.log10(1.32E-05))
    rescale = np.array(10.**(tmp - S_over_H_cloudy))
    all_emission_lines[:,idx_sulfur] *= rescale[:, np.newaxis]

    # Rescale by Q if out of bounds
    # Note that this isn't exactly correct but I feel uncomfortable with too high Q
    rescale = np.ones(len(df))
    tmp = df["ionizing_luminosity"] 
    rescale[tmp < 46.5] = 10.**(tmp[tmp < 46.5] - 46.5)
    rescale[tmp > 54.5] = 10.**(tmp[tmp > 54.5] - 54.5)
    all_emission_lines *= rescale[:, np.newaxis]

    return np.log10(all_emission_lines)

def apply_stromgren_correction(
        cells_to_replace,
        nH, nO, nC, nN, nNe, nS,
        O_over_H, C_over_H, N_over_H, Ne_over_H, S_over_H,
        O_depletion, C_depletion, N_depletion, Ne_depletion, S_depletion,
        star_age, star_metal, star_ion_lums,
):
    """
    Function to apply the stromgren correction to the emission lines.
    This will rescale the emission lines based on the deviation of the gas properties
    from the properties of the cloudy models.
    """

    # Get the interpolation matrix
    # (metal, O/H, nH, ages, ionLum, C/O) --> list of line luminosities (erg/s)
    mif_cloudy, line_list = get_cloudy_el_interpolator()

    # Format the input for harley's cloudy correction
    df_strom = {
        "nH": np.log10(nH[cells_to_replace]),
        "nO": np.log10(nO[cells_to_replace]),
        "nC": np.log10(nC[cells_to_replace]),
        "nN": np.log10(nN[cells_to_replace]),
        "nNe": np.log10(nNe[cells_to_replace]),
        "nS": np.log10(nS[cells_to_replace]),
        # THESE ARE DEFINED IN LOGARITHMIC UNITS ----
        "[O/H]": O_over_H[cells_to_replace],
        "[C/H]": C_over_H[cells_to_replace],
        "[N/H]": N_over_H[cells_to_replace],
        "[Ne/H]": Ne_over_H[cells_to_replace],
        "[S/H]": S_over_H[cells_to_replace],
        "O_dep": O_depletion[cells_to_replace],
        "C_dep": C_depletion[cells_to_replace],
        "N_dep": N_depletion[cells_to_replace],
        "Ne_dep": Ne_depletion[cells_to_replace],
        "S_dep": S_depletion[cells_to_replace],
        # -------------------------------------------
        "age": star_age[cells_to_replace],
        "metallicity": star_metal[cells_to_replace],
        "ionizing_luminosity": np.log10(star_ion_lums[cells_to_replace]),
    }

    print ("Applying stromgren correction to", len(df_strom), "cells")
    print (df_strom)

    to_interpolate = format_cloudy_interpolator(df_strom)
    replace_emission_lines = 10.**mif_cloudy(to_interpolate)
    replace_emission_lines = rescale_emission(df_strom, replace_emission_lines, line_list)

    df_nans = df_strom[np.isnan(replace_emission_lines)[:,0]]
    if len(df_nans) > 0:
        # First iteration
        df_nans.loc[:,"age"] += 1.0 # Try updating the age

        to_interpolate = format_cloudy_interpolator(df_nans)
        replace_emission_lines = 10.**mif_cloudy(to_interpolate)
        tmp = rescale_emission(df_nans, replace_emission_lines, line_list)

        # Second iteration
        if np.isnan(tmp)[:,0].sum() > 0:
            df_nans.loc[:,"age"][np.isnan(tmp)[:,0]] += 1.0 # Try updating the age again

            to_interpolate = format_cloudy_interpolator(df_nans)
            replace_emission_lines = 10.**mif_cloudy(to_interpolate)
            tmp = rescale_emission(df_nans, replace_emission_lines, line_list)

            # Third iteration
            if np.isnan(tmp)[:,0].sum() > 0:
                df_nans.loc[:,"age"][np.isnan(tmp)[:,0]] += 1.0 # Try updating the age again

                to_interpolate = format_cloudy_interpolator(df_nans)
                replace_emission_lines = 10.**mif_cloudy(to_interpolate)
                tmp = rescale_emission(df_nans, replace_emission_lines, line_list)

                # Fourth iteration, set the age back and lower Q
                if np.isnan(tmp)[:,0].sum() > 0:
                    rescale_flux = np.zeros((len(df_nans),1))
                    df_nans.loc[:,"age"][np.isnan(tmp)[:,0]] -= 3.0
                    df_nans.loc[:,"ionizing_luminosity"][np.isnan(tmp)[:,0]] -= 1.0
                    rescale_flux[np.isnan(tmp)[:,0],0] = 1.0

                    to_interpolate = format_cloudy_interpolator(df_nans)
                    replace_emission_lines = 10.**mif_cloudy(to_interpolate)
                    tmp = rescale_emission(df_nans, replace_emission_lines, line_list) + rescale_flux

        replace_emission_lines[np.isnan(replace_emission_lines)[:,0]] = tmp


    # Check for nans again
    # Note that there can occasionally be a nan for some high ionization lines
    # so first check for those
    replace_emission_lines[:,53][np.isnan(replace_emission_lines[:,53])] = -50.0
    replace_emission_lines[:,54][np.isnan(replace_emission_lines[:,54])] = -50.0
    if (np.isnan(replace_emission_lines).sum() > 0):
        print("!!!!!! OMG !!!!!! --> you broke cloudy")

    return 10**replace_emission_lines

def get_line_list_map_dict():
    # Mapper from the cloudy line names to the field names we want to use in yt
    line_list_map_dict = {
            'H  1 1215.67A': "Lya",
            'H  1 6562.80A': "Ha",
            'H  1 4861.32A': "Hb",
            'H  1 4340.46A': "Hg",
            'H  1 4101.73A': "Hd",
            'He 2 1640.41A': "He2-1640",
            'He 2 4685.68A': "He2-4686",
            'O  1 63.1679m': None,
            'O  1 145.495m': None,
            'O  1 6300.30A': "O1-6300",
            'O  1 6363.78A': "O1-6362",
            'O  2 3726.03A': "O2-3726",
            'O  2 3728.81A': "O2-3728",
            'O 2R 3726.00A': "O2-3726",
            'O 2R 3729.00A': "O2-3728",
            'O  2 7318.92A': "O2-7319",
            'O  2 7319.99A': "O2-7320",
            'O  2 7329.67A': "O2-7330",
            'O  2 7330.73A': "O2-7331",
            'O 2R 7332.00A': "O2-7331",
            'O 2R 7323.00A': "O2-7320",
            'O  3 1660.81A': "O3-1661",
            'O  3 1666.15A': "O3-1666",
            'O  3 4363.21A': "O3-4363",
            'O 3R 4363.00A': "O3-4363",
            'O 3C 4363.00A': "O3-4363",
            'O  3 4958.91A': "O3-4959",
            'O  3 5006.84A': "O3-5007",
            'O  3 51.8004m': None,
            'O  3 88.3323m': None,
            'O  4 25.8863m': None,
            'Ne 3 3868.76A': "Ne3-3869",
            'Ne 3 3967.47A': "Ne3-3967",
            'C  2 157.636m': None,
            'C  3 1906.68A': "C3-1906",
            'C  3 1908.73A': "C3-1908",
            'C  4 1548.19A': "C4-1548",
            'C  4 1550.77A': "C4-1550",
            'N  2 5754.59A': "N2-5755",
            'N 2R 5755.00A': "N2-5755",
            'N  2 6548.05A': "N2-6548",
            'N  2 6583.45A': "N2-6583",
            'N 2R 6584.00A': "N2-6583",
            'N  2 205.283m': None,
            'N  2 121.769m': None,
            'N  3 57.3238m': None,
            'N  3 1748.65A': "N3-1749",
            'N  3 1753.99A': "N3-1754",
            'N  3 1746.82A': "N3-1747",
            'N  3 1752.16A': "N3-1752",
            'N  3 1749.67A': "N3-1750",
            'N  4 1483.32A': "N4-1483",
            'N  4 1486.50A': "N4-1486",
            'N  5 1238.82A': "N5-1239",
            'N  5 1242.80A': "N5-1243",
            'S  2 6716.44A': "S2-6716",
            'S  2 6730.82A': "S2-6731",
            'S  2 4076.35A': "S2-4076",
            'S  2 4068.60A': "S2-4069",
            'S  3 6312.06A': None,
            'S  3 9068.62A': None,
            'S  3 9530.62A': None,
            }
    return line_list_map_dict