# Generating derived fields for the emission lines module
# NOTE: for the MEGATRON simulations ran with RAMSES-RTZ

# This module will follow closely the approach of
# Harley Katz for computing spectra and emission lines

# Everything normally needs to be in cgs units

# TODO: Add corections for unresolved stromgren spheres

# Author: Anatole Storck

from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from roman import fromRoman

from yt import units as u
from yt.fields.field_detector import FieldDetector
from yt.funcs import mylog

from yt_derived_fields.megatron_derived_fields import (
    chemistry_derived_fields as chem_fields,
    chemistry_data as chem_data,
)
from yt_derived_fields.spectral_utils.setup_stromgren_correction_interpolators import (
    format_cloudy_interpolator,
    get_cloudy_el_interpolator,
)

_spectral_data = Path(__file__).parent.parent / "spectral_utils"
met_data = chem_data.get_metal_data()
prim_data = chem_data.get_prim_data()


@cache
def get_coll_line_dict() -> dict[str, Any]:
    coll_line_path = Path(_spectral_data / "coll_line_dict.npy")
    if coll_line_path.exists():
        mylog.info("Loading collision line dictionary from disk.")
        _COLL_LINE_DICT = np.load(coll_line_path, allow_pickle=True).item()
        return _COLL_LINE_DICT
    else:
        raise FileNotFoundError(
            "Could not locate the collisional line dictionary. "
            "Please generate it using generate_atomic_grids.py "
            "in the spectral_utils folder."
        )


@cache
def get_rec_line_dict() -> dict[str, Any]:
    rec_line_path = Path(_spectral_data / "rec_line_dict.npy")
    if rec_line_path.exists():
        mylog.info("Loading recombination line dictionary from disk.")
        _REC_LINE_DICT = np.load(_spectral_data / "rec_line_dict.npy", allow_pickle=True).item()
        return _REC_LINE_DICT
    else:
        raise FileNotFoundError(
            "Could not locate the recombination line dictionary. "
            "Please generate it using generate_atomic_grids.py "
            "in the spectral_utils folder."
        )

def generate_dust_depletion_field(ds):

    elements = ["Fe", "O", "N", "C", "Mg", "Ne", "Si", "S"," Ca"," CO"]
    dep_solar = [0.01, 0.728, 0.603, 0.501, 0.163, 1.0, 0.1, 1.0, 0.003, 1.0]

    for el, dep in zip(elements, dep_solar):
        def _get_depletion(field, data):

            nH = data["gas", "hydrogen_number_density"].in_units("cm**-3").d
            nO = data["gas", "oxygen_number_density"].in_units("cm**-3").d
            T = data["gas", "temperature"].in_units("K").d

            if isinstance(data, FieldDetector):
                return np.zeros(nH.shape)

            depletion_table = np.ones(len(nH))
            filt = T < 6.0

            x = 12.0 + (nO - nH)

            # Broken powerlaw model from RR14 (consistent with Taysun's Lya feedback)
            # See Table 1 of: https://www.aanda.org/articles/aa/pdf/2014/03/aa22803-13.pdf
            # We use the XCO,Z case
            a  = 2.21
            aH = 1.00
            b  = 0.96
            aL = 3.10
            xt = 8.10
            xs = 8.69

            y = a + (aH * (xs - x))
            y[x<=xt] = b + (aL * (xs - x[x<=xt]))

            y = 10.0**y # This is the Gas to Dust mass ratio

            # Fill table with depletions
            depletion_table[filt] = 1.0 - ( (1.0 - dep) * np.minimum(1.0, 162.0/y[filt]) )

            return depletion_table

        ds.add_field(
            name=("gas", f"{el}_dep"),
            function=_get_depletion,
            units="",
            sampling_type="cell",
            display_name=f"{el} Depletion Factor",
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

def rescaling_interpolator(
        line_list, cells_to_replace, all_emission_lines,
        O_over_H_log, O_depletion_log,
        C_over_H_log, C_depletion_log,
        N_over_H_log, N_depletion_log,
        Ne_over_H_log, Ne_depletion_log,
        S_over_H_log, S_depletion_log,
        star_metal, star_ion_lums,
):

    # Rescale in case we are above or below the gas metallicity bounds
    tmp = np.array(O_over_H_log + O_depletion_log - np.log10(star_metal / 0.014))
    idx_oxygen = [i for i,j in enumerate(line_list) if j[0]=="O"]
    n_rows = all_emission_lines.shape[0]
    rescale = np.ones(n_rows)
    rescale[tmp < -3.0] = (10.**tmp[tmp < -3.0]) / (10.**-3.0)
    rescale[tmp > 4.0] = (10.**tmp[tmp > 4.0]) / (10.**4.0)
    all_emission_lines[:,idx_oxygen] *= rescale[:, np.newaxis]

    # Rescale the carbon lines in case of bounds errors
    idx_carbon = [i for i,j in enumerate(line_list) if j[0]=="C"]
    C_over_H_cloudy = np.log10(np.array([get_all_dust_depletions(kk)["Carbon"] for kk in O_over_H_log+O_depletion_log])) # Dust depletion (Carbon)
    tmp1 = np.array(C_over_H_log - O_over_H_log)
    tmp1[tmp1 < -3.0] = -3.0
    tmp1[tmp1 > 1.0] = 1.0
    C_over_H_cloudy += np.log10(2.69E-04) + O_over_H_log + O_depletion_log + tmp1 
    tmp = np.array(C_over_H_log + C_depletion_log + np.log10(2.69E-04))
    rescale = np.array(tmp / C_over_H_cloudy)
    all_emission_lines[:,idx_carbon] *= rescale[:, np.newaxis]

    # Rescale nitrogen by deviation from solar
    idx_nitrogen = [i for i,j in enumerate(line_list) if j[0]=="N"]
    N_over_H_cloudy = np.log10(np.array([get_all_dust_depletions(kk)["Nitrogen"] for kk in O_over_H_log+O_depletion_log])) # Dust depletion (Carbon)
    N_over_H_cloudy += np.log10(6.76E-05) + O_over_H_log + O_depletion_log
    tmp = np.array(N_over_H_log + N_depletion_log + np.log10(6.76E-05))
    rescale = np.array(10.**(tmp - N_over_H_cloudy))
    all_emission_lines[:,idx_nitrogen] *= rescale[:, np.newaxis]

    # Rescale neon by deviation from solar
    idx_neon = [i for i,j in enumerate(line_list) if j[:2]=="Ne"]
    Ne_over_H_cloudy = np.log10(np.array([get_all_dust_depletions(kk)["Neon"] for kk in O_over_H_log+O_depletion_log])) # Dust depletion (Carbon)
    Ne_over_H_cloudy += np.log10(8.51E-05) + O_over_H_log + O_depletion_log
    tmp = np.array(Ne_over_H_log + Ne_depletion_log + np.log10(8.51E-05))
    rescale = np.array(10.**(tmp - Ne_over_H_cloudy))
    all_emission_lines[:,idx_neon] *= rescale[:, np.newaxis]

    # Rescale sulfur by deviation from solar
    idx_sulfur = [i for i,j in enumerate(line_list) if j[:2]=="S "]
    S_over_H_cloudy = np.log10(np.array([get_all_dust_depletions(kk)["Sulphur"] for kk in O_over_H_log+O_depletion_log])) # Dust depletion (Carbon)
    S_over_H_cloudy += np.log10(1.32E-05) + O_over_H_log + O_depletion_log
    tmp = np.array(S_over_H_log + S_depletion_log + np.log10(1.32E-05))
    rescale = np.array(10.**(tmp - S_over_H_cloudy))
    all_emission_lines[:,idx_sulfur] *= rescale[:, np.newaxis]

    # Rescale by Q if out of bounds
    # Note that this isn't exactly correct but I feel uncomfortable with too high Q
    rescale = np.ones(n_rows)
    tmp = np.log10(star_ion_lums)
    rescale[tmp < 46.5] = 10.**(tmp[tmp < 46.5] - 46.5)
    rescale[tmp > 54.5] = 10.**(tmp[tmp > 54.5] - 54.5)
    all_emission_lines *= rescale[:, np.newaxis]

    return all_emission_lines

def get_emission_lines(
        ds,
        coll_lines=None,
        rec_lines=None,
        all_lines=False,
        fix_unres_stromgren=False,
):
    """
    Add emission line luminosity fields to the dataset, given line emissivity grids
    have been generated on disk.

    Args:
        ds (yt.Dataset): The dataset object.
        coll_lines (list, optional): A list of collision lines to consider.
                                     Defaults to None.
                                     Example: ["O3-5007", "S2-6731"]
        rec_lines (list, optional): A list of recombination lines to consider.
                                    Defaults to None.
                                    Example: ["Lya", "Hb", "He-1640"]
        fix_unres_stromgren (bool, optional): Replace emission from cells that have
                                             unresolved stromgren spheres with CLOUDY models.
    """

    # generate chemistry fields if not already present
    if ("gas", "electron_number_density") not in ds.derived_field_list:
        chem_fields.create_chemistry_derived_fields(ds, electron_number_density=True)

    # generate dust depletion fields if not already present (needed for stromgren correction)
    if fix_unres_stromgren and ("gas", "O_dep") not in ds.derived_field_list: # Check for one of the depletion fields
        generate_dust_depletion_field(ds)

    # These dictionaries are used to store the emission line metadata, along with interpolation grids
    # Can be generated using the generate_atomic_grids.py script (To contain more lines or finer interpolation)
    if coll_lines is not None or all_lines:
        coll_line_dict = get_coll_line_dict()
    if rec_lines is not None or all_lines:
        rec_line_dict = get_rec_line_dict()
    if fix_unres_stromgren:
        # Get the interpolation matrix
        # (metal, O/H, nH, ages, ionLum, C/O) --> list of line luminosities (erg/s)
        mif_cloudy, line_list = get_cloudy_el_interpolator()

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

        # Build a robust lookup from yt line names (e.g. "Ha") to CLOUDY line-list indices.
        # Some yt names can map to multiple CLOUDY aliases; keep the first valid entry.
        line_index_map = {}
        for cloudy_name, yt_name in line_list_map_dict.items():
            if yt_name is None or yt_name in line_index_map:
                continue
            if cloudy_name in line_list:
                line_index_map[yt_name] = line_list.index(cloudy_name)

        def get_cloudy_line_luminosity(all_emission_lines, line_idx, n_cells):
            arr = np.asarray(all_emission_lines)

            if line_idx is None or arr.size == 0:
                return np.zeros(n_cells)

            if arr.ndim == 1:
                if line_idx == 0 and arr.shape[0] == n_cells:
                    return arr
                return np.zeros(n_cells)

            # Preferred convention: (n_lines, n_cells)
            if arr.shape[0] > line_idx and arr.shape[1] == n_cells:
                return arr[line_idx, :]

            # Alternate convention: (n_cells, n_lines)
            if arr.shape[1] > line_idx and arr.shape[0] == n_cells:
                return arr[:, line_idx]

            # Last-resort fallbacks if dimensions are unexpected.
            if arr.shape[0] > line_idx:
                candidate = arr[line_idx, :]
                if candidate.shape[0] == n_cells:
                    return candidate
            if arr.shape[1] > line_idx:
                candidate = arr[:, line_idx]
                if candidate.shape[0] == n_cells:
                    return candidate

            return np.zeros(n_cells)

    # line in the form of, for example, "O3-5007"
    def coll_line(ds, line):
        def _get_coll_line_emissivity(field, data):

            rho = data["gas", "density"].to("g/cm**3").value
            Tgas = data["gas", "temperature"].to("K").value
            ne = data["gas", "electron_number_density"].to("cm**-3").value

            # ----------------------------------------------------------

            # Find the fields for stromgren sphere corrections if needed
            if fix_unres_stromgren:

                nH = data["gas", "hydrogen_number_density"].in_units("cm**-3").d
                nO = data["gas", "oxygen_number_density"].in_units("cm**-3").d
                nC = data["gas", "carbon_number_density"].in_units("cm**-3").d
                nN = data["gas", "nitrogen_number_density"].in_units("cm**-3").d
                nNe = data["gas", "neon_number_density"].in_units("cm**-3").d
                nS = data["gas", "sulfur_number_density"].in_units("cm**-3").d

                O_depletion = data["gas", "O_dep"]
                C_depletion = data["gas", "C_dep"]
                N_depletion = data["gas", "N_dep"]
                Ne_depletion = data["gas", "Ne_dep"]
                S_depletion = data["gas", "S_dep"]

                star_age = data["deposit", "young_pop2_avg_age"].in_units("Myr").d
                star_metal = data["deposit", "young_pop2_avg_metallicity"]
                star_ion_lums = data["deposit", "young_pop2_sum_ionizing_luminosity"].d

                cells_to_replace = data["gas", "unresolved_stromgren"]

            # ----------------------------------------------------------

            el = coll_line_dict[line]["ion"].split("_")[0]  # C,    O,      Fe
            ion_roman = coll_line_dict[line]["ion"].split("_")[1]  # II,   III,    VII

            nel = data["gas", f"{met_data[el]['name']}_number_density"].to("cm**-3").value
            xion = data["gas", f"{met_data[el]['name']}_{fromRoman(ion_roman):02d}"].value

            if isinstance(data, FieldDetector):
                return np.zeros(rho.shape) * u.erg / u.s

            # Set up the arrays to interpolate
            to_interp = np.zeros((len(rho), 2))
            to_interp[:, 0] = np.log10(Tgas)
            to_interp[:, 1] = np.log10(ne)

            # get emisitivity of cells based on T and ne
            loc_emis = coll_line_dict[line]["emis_grid"](to_interp)

            # Multiply by the electron density and ion density
            # n{el} = nO, ion = O_IV
            loc_lum = ne * nel * xion

            if fix_unres_stromgren:
                loc_lum_cloudy = np.zeros(int(cells_to_replace.sum()))

                nH = nH[cells_to_replace]
                nO = nO[cells_to_replace]
                nC = nC[cells_to_replace]
                nN = nN[cells_to_replace]
                nNe = nNe[cells_to_replace]
                nS = nS[cells_to_replace]

                O_depletion_log = np.log10(O_depletion[cells_to_replace])
                C_depletion_log = np.log10(C_depletion[cells_to_replace])
                N_depletion_log = np.log10(N_depletion[cells_to_replace])
                Ne_depletion_log = np.log10(Ne_depletion[cells_to_replace])
                S_depletion_log = np.log10(S_depletion[cells_to_replace])

                star_age = star_age[cells_to_replace]
                star_metal = star_metal[cells_to_replace]
                star_ion_lums = star_ion_lums[cells_to_replace]

                O_over_H_log = np.log10(nO) - np.log10(nH) - np.log10(4.90E-04)     # depletion
                C_over_H_log = np.log10(nC) - np.log10(nH) - np.log10(2.69E-04)     # depletion
                N_over_H_log = np.log10(nN) - np.log10(nH) - np.log10(6.76E-05)     # depletion
                Ne_over_H_log = np.log10(nNe) - np.log10(nH) - np.log10(8.51E-05)   # depletion
                S_over_H_log = np.log10(nS) - np.log10(nH) - np.log10(1.32E-05)     # depletion

                to_interp = format_cloudy_interpolator(
                    nH, C_over_H_log, O_over_H_log, O_depletion_log,
                    star_age, star_metal, star_ion_lums,
                )

                # TODO: This line does the interpolation for all lines, but we redo it for every line.
                # Can be sped up by only doing it once and storing the results.
                all_emission_lines = 10.**mif_cloudy(to_interp)

                print(all_emission_lines[:, 13])

                all_emission_lines = rescaling_interpolator(
                    line_list, cells_to_replace, all_emission_lines,
                    O_over_H_log, O_depletion_log,
                    C_over_H_log, C_depletion_log,
                    N_over_H_log, N_depletion_log,
                    Ne_over_H_log, Ne_depletion_log,
                    S_over_H_log, S_depletion_log,
                    star_metal, star_ion_lums,
                )

                print(all_emission_lines[:, 13])

                nan_mask = np.isnan(all_emission_lines).any(axis=1)
                if nan_mask.any():
                    idx_nan = np.where(nan_mask)[0]

                    # First iteration
                    tmp_star_age = star_age[nan_mask] * 10 # Try updating the age

                    to_interp = format_cloudy_interpolator(
                        nH[nan_mask], C_over_H_log[nan_mask], O_over_H_log[nan_mask], O_depletion_log[nan_mask],
                        tmp_star_age, star_metal[nan_mask], star_ion_lums[nan_mask],
                    )
                    all_emission_lines_tmp = 10.**mif_cloudy(to_interp)
                    print(all_emission_lines_tmp[:, 13])
                    all_emission_lines_tmp = rescaling_interpolator(
                        line_list, cells_to_replace, all_emission_lines_tmp,
                        O_over_H_log[nan_mask], O_depletion_log[nan_mask],
                        C_over_H_log[nan_mask], C_depletion_log[nan_mask],
                        N_over_H_log[nan_mask], N_depletion_log[nan_mask],
                        Ne_over_H_log[nan_mask], Ne_depletion_log[nan_mask],
                        S_over_H_log[nan_mask], S_depletion_log[nan_mask],
                        star_metal[nan_mask], star_ion_lums[nan_mask],
                    )
                    print(all_emission_lines_tmp[:, 13])

                    nan_mask_second = np.isnan(all_emission_lines_tmp).any(axis=1)
                    if nan_mask_second.any():
                        idx_nan_second = np.where(nan_mask_second)[0]

                        # Second iteration, updating the subset which are still NaN with older age
                        tmp_star_age_second = tmp_star_age[nan_mask_second] * 10
                        to_interp = format_cloudy_interpolator(
                            nH[nan_mask][nan_mask_second], C_over_H_log[nan_mask][nan_mask_second], O_over_H_log[nan_mask][nan_mask_second], O_depletion_log[nan_mask][nan_mask_second],
                            tmp_star_age_second, star_metal[nan_mask][nan_mask_second], star_ion_lums[nan_mask][nan_mask_second],
                        )
                        all_emission_lines_tmp_second = 10.**mif_cloudy(to_interp)
                        print(all_emission_lines_tmp_second[:, 13])
                        all_emission_lines_tmp_second = rescaling_interpolator(
                            line_list, cells_to_replace, all_emission_lines_tmp_second,
                            O_over_H_log[nan_mask][nan_mask_second], O_depletion_log[nan_mask][nan_mask_second],
                            C_over_H_log[nan_mask][nan_mask_second], C_depletion_log[nan_mask][nan_mask_second],
                            N_over_H_log[nan_mask][nan_mask_second], N_depletion_log[nan_mask][nan_mask_second],
                            Ne_over_H_log[nan_mask][nan_mask_second], Ne_depletion_log[nan_mask][nan_mask_second],
                            S_over_H_log[nan_mask][nan_mask_second], S_depletion_log[nan_mask][nan_mask_second],
                            star_metal[nan_mask][nan_mask_second], star_ion_lums[nan_mask][nan_mask_second],
                        )
                        print(all_emission_lines_tmp_second[:, 13])
                        all_emission_lines_tmp[nan_mask_second] = all_emission_lines_tmp_second

                        nan_mask_third = np.isnan(all_emission_lines_tmp).any(axis=1)
                        if nan_mask_third.any():
                            idx_nan_third = np.where(nan_mask_third)[0]

                            # Third iteration, updating the subset which are still NaN with older age
                            tmp_star_age_third = tmp_star_age_second[nan_mask_third] * 10
                            to_interp = format_cloudy_interpolator(
                                nH[nan_mask][nan_mask_second][nan_mask_third], C_over_H_log[nan_mask][nan_mask_second][nan_mask_third], O_over_H_log[nan_mask][nan_mask_second][nan_mask_third], O_depletion_log[nan_mask][nan_mask_second][nan_mask_third],
                                tmp_star_age_third, star_metal[nan_mask][nan_mask_second][nan_mask_third], star_ion_lums[nan_mask][nan_mask_second][nan_mask_third],
                            )
                            all_emission_lines_tmp_third = 10.**mif_cloudy(to_interp)
                            all_emission_lines_tmp_third = rescaling_interpolator(
                                line_list, cells_to_replace, all_emission_lines_tmp_third,
                                O_over_H_log[nan_mask][nan_mask_second][nan_mask_third], O_depletion_log[nan_mask][nan_mask_second][nan_mask_third],
                                C_over_H_log[nan_mask][nan_mask_second][nan_mask_third], C_depletion_log[nan_mask][nan_mask_second][nan_mask_third],
                                N_over_H_log[nan_mask][nan_mask_second][nan_mask_third], N_depletion_log[nan_mask][nan_mask_second][nan_mask_third],
                                Ne_over_H_log[nan_mask][nan_mask_second][nan_mask_third], Ne_depletion_log[nan_mask][nan_mask_second][nan_mask_third],
                                S_over_H_log[nan_mask][nan_mask_second][nan_mask_third], S_depletion_log[nan_mask][nan_mask_second][nan_mask_third],
                                star_metal[nan_mask][nan_mask_second][nan_mask_third], star_ion_lums[nan_mask][nan_mask_second][nan_mask_third],
                            )

                            all_emission_lines_tmp[idx_nan_third] = all_emission_lines_tmp_third

                            nan_mask_fourth = np.isnan(all_emission_lines_tmp).any(axis=1)
                            if nan_mask_fourth.any():
                                idx_nan_fourth = np.where(nan_mask_fourth)[0]

                                # Final iteration, updating the subset which are still NaN with a lower Q (reset the age back to the original)
                                tmp_star_age_fourth = star_age[nan_mask][nan_mask_second][nan_mask_third][nan_mask_fourth]
                                tmp_star_ion_lums_fourth = star_ion_lums[nan_mask][nan_mask_second][nan_mask_third][nan_mask_fourth] / 10
                                to_interp = format_cloudy_interpolator(
                                    nH[nan_mask][nan_mask_second][nan_mask_third][nan_mask_fourth], C_over_H_log[nan_mask][nan_mask_second][nan_mask_third][nan_mask_fourth], O_over_H_log[nan_mask][nan_mask_second][nan_mask_third][nan_mask_fourth], O_depletion_log[nan_mask][nan_mask_second][nan_mask_third][nan_mask_fourth],
                                    tmp_star_age_fourth, star_metal[nan_mask][nan_mask_second][nan_mask_third][nan_mask_fourth], tmp_star_ion_lums_fourth,
                                )
                                all_emission_lines_tmp_fourth = 10.**mif_cloudy(to_interp)
                                all_emission_lines_tmp_fourth = rescaling_interpolator(
                                    line_list, cells_to_replace, all_emission_lines_tmp_fourth,
                                    O_over_H_log[nan_mask][nan_mask_second][nan_mask_third][nan_mask_fourth], O_depletion_log[nan_mask][nan_mask_second][nan_mask_third][nan_mask_fourth],
                                    C_over_H_log[nan_mask][nan_mask_second][nan_mask_third][nan_mask_fourth], C_depletion_log[nan_mask][nan_mask_second][nan_mask_third][nan_mask_fourth],
                                    N_over_H_log[nan_mask][nan_mask_second][nan_mask_third][nan_mask_fourth], N_depletion_log[nan_mask][nan_mask_second][nan_mask_third][nan_mask_fourth],
                                    Ne_over_H_log[nan_mask][nan_mask_second][nan_mask_third][nan_mask_fourth], Ne_depletion_log[nan_mask][nan_mask_second][nan_mask_third][nan_mask_fourth],
                                    S_over_H_log[nan_mask][nan_mask_second][nan_mask_third][nan_mask_fourth], S_depletion_log[nan_mask][nan_mask_second][nan_mask_third][nan_mask_fourth],
                                    star_metal[nan_mask][nan_mask_second][nan_mask_third][nan_mask_fourth], tmp_star_ion_lums_fourth,
                                )
                                all_emission_lines_tmp[idx_nan_fourth] = all_emission_lines_tmp_fourth

                    all_emission_lines[idx_nan] = all_emission_lines_tmp

                line_idx = line_index_map.get(line)
                if line_idx is not None:
                    loc_lum_cloudy = get_cloudy_line_luminosity(
                        all_emission_lines, line_idx, int(cells_to_replace.sum())
                    )

                loc_lum[cells_to_replace] = loc_lum_cloudy

            return loc_lum * u.erg / u.s / u.cm**3

        def _get_coll_line_lum(field, data):
            return data["gas", "cell_volume"] * data["gas", f"{line}_emissivity"]

        ds.add_field(
            name=("gas", f"{line}_emissivity"),
            function=_get_coll_line_emissivity,
            units="erg/s/cm**3",
            sampling_type="cell",
            display_name=f"{line} Emissivity",
        )
        ds.add_field(
            name=("gas", f"{line}_luminosity"),
            function=_get_coll_line_lum,
            units="erg/s",
            sampling_type="cell",
            display_name=f"{line} Luminosity",
        )

    # line in the form of "Lya", "Hb", "He-1640"
    def rec_line(ds, line):
        def _get_rec_line_emissivity(field, data):

            rho = data["gas", "density"].to("g/cm**3").value
            Tgas = data["gas", "temperature"].to("K").value
            ne = data["gas", "electron_number_density"].to("cm**-3").value

            # ----------------------------------------------------------

            # Find the fields for stromgren sphere corrections if needed
            if fix_unres_stromgren:

                nH = data["gas", "hydrogen_number_density"].in_units("cm**-3").d
                nO = data["gas", "oxygen_number_density"].in_units("cm**-3").d
                nC = data["gas", "carbon_number_density"].in_units("cm**-3").d
                nN = data["gas", "nitrogen_number_density"].in_units("cm**-3").d
                nNe = data["gas", "neon_number_density"].in_units("cm**-3").d
                nS = data["gas", "sulfur_number_density"].in_units("cm**-3").d

                O_depletion = data["gas", "O_dep"]
                C_depletion = data["gas", "C_dep"]
                N_depletion = data["gas", "N_dep"]
                Ne_depletion = data["gas", "Ne_dep"]
                S_depletion = data["gas", "S_dep"]

                star_age = data["deposit", "young_pop2_avg_age"].in_units("Myr").d
                star_metal = data["deposit", "young_pop2_avg_metallicity"]
                star_ion_lums = data["deposit", "young_pop2_sum_ionizing_luminosity"].d

                cells_to_replace = data["gas", "unresolved_stromgren"]

            # ----------------------------------------------------------

            el = rec_line_dict[line]["ion"].split("_")[0]
            ion_roman = rec_line_dict[line]["ion"].split("_")[1]

            nel = data["gas", f"{prim_data[el]['name']}_number_density"].to("1/cm**3").value

            xion = data["gas", f"{prim_data[el]['name']}_{fromRoman(ion_roman):02d}"].value
            xion_col = data["gas", f"{prim_data[el]['name']}_{fromRoman(ion_roman) - 1:02d}"].value

            if isinstance(data, FieldDetector):
                return np.zeros(rho.shape) * u.erg / u.s

            # Set up the arrays to interpolate
            to_interp = np.zeros((len(rho), 2))
            to_interp[:, 0] = np.log10(Tgas)
            to_interp[:, 1] = np.log10(ne)

            min_temp = 10.0

            to_interp_ch = np.array(Tgas)
            to_interp_ch[to_interp_ch < min_temp] = min_temp
            to_interp_ch[to_interp_ch > 1e9] = 1e9

            loc_rec_emis = rec_line_dict[line]["emis_grid"](to_interp)
            loc_rec_emis *= ne * nel * xion  # NOTE * df_gas[f"{el}_dep"]

            loc_col_emis = rec_line_dict[line]["emis_grid_col"](to_interp_ch)
            loc_col_emis *= ne * nel * xion_col  # NOTE * df_gas[f"{el}_dep"]

            loc_rec_lum = loc_rec_emis  # erg/s/cm^3
            loc_col_lum = loc_col_emis  # erg/s/cm^3

            loc_lum = loc_rec_lum + loc_col_lum

            if fix_unres_stromgren:
                loc_lum_cloudy = np.zeros(int(cells_to_replace.sum()))

                nH = nH[cells_to_replace]
                nO = nO[cells_to_replace]
                nC = nC[cells_to_replace]
                nN = nN[cells_to_replace]
                nNe = nNe[cells_to_replace]
                nS = nS[cells_to_replace]

                O_depletion_log = np.log10(O_depletion[cells_to_replace])
                C_depletion_log = np.log10(C_depletion[cells_to_replace])
                N_depletion_log = np.log10(N_depletion[cells_to_replace])
                Ne_depletion_log = np.log10(Ne_depletion[cells_to_replace])
                S_depletion_log = np.log10(S_depletion[cells_to_replace])

                star_age = star_age[cells_to_replace]
                star_metal = star_metal[cells_to_replace]
                star_ion_lums = star_ion_lums[cells_to_replace]

                O_over_H_log = np.log10(nO) - np.log10(nH) - np.log10(4.90E-04)     # depletion
                C_over_H_log = np.log10(nC) - np.log10(nH) - np.log10(2.69E-04)     # depletion
                N_over_H_log = np.log10(nN) - np.log10(nH) - np.log10(6.76E-05)     # depletion
                Ne_over_H_log = np.log10(nNe) - np.log10(nH) - np.log10(8.51E-05)   # depletion
                S_over_H_log = np.log10(nS) - np.log10(nH) - np.log10(1.32E-05)     # depletion

                to_interp = format_cloudy_interpolator(
                    nH, C_over_H_log, O_over_H_log, O_depletion_log,
                    star_age, star_metal, star_ion_lums,
                )

                # TODO: This line does the interpolation for all lines, but we redo it for every line.
                # Can be sped up by only doing it once and storing the results.
                all_emission_lines = 10.**mif_cloudy(to_interp)

                all_emission_lines = rescaling_interpolator(
                    line_list, cells_to_replace, all_emission_lines,
                    O_over_H_log, O_depletion_log,
                    C_over_H_log, C_depletion_log,
                    N_over_H_log, N_depletion_log,
                    Ne_over_H_log, Ne_depletion_log,
                    S_over_H_log, S_depletion_log,
                    star_metal, star_ion_lums,
                )

                line_idx = line_index_map.get(line)
                if line_idx is not None:
                    loc_lum_cloudy = get_cloudy_line_luminosity(
                        all_emission_lines, line_idx, int(cells_to_replace.sum())
                    )

                loc_lum[cells_to_replace] = loc_lum_cloudy

            return loc_lum * u.erg / u.s / u.cm**3

        def _get_rec_line_lum(field, data):
            return data["gas", "cell_volume"] * data["gas", f"{line}_emissivity"]

        ds.add_field(
            name=("gas", f"{line}_emissivity"),
            function=_get_rec_line_emissivity,
            units="erg/s/cm**3",
            sampling_type="cell",
            display_name=f"{line} Emissivity",
        )

        ds.add_field(
            name=("gas", f"{line}_luminosity"),
            function=_get_rec_line_lum,
            units="erg/s",
            sampling_type="cell",
            display_name=f"{line} Luminosity",
        )

    if all_lines:
        for line in get_coll_line_dict().keys():
            coll_line(ds, line)
        for line in get_rec_line_dict().keys():
            rec_line(ds, line)
    else:
        if coll_lines:
            for line in coll_lines:
                coll_line(ds, line)
        if rec_lines:
            for line in rec_lines:
                rec_line(ds, line)
