# Generating derived fields for the emission lines module
# NOTE: for the MEGATRON simulations ran with RAMSES-RTZ

# This module will follow closely the approach of
# Harley Katz for computing spectra and emission lines

# Everything normally needs to be in cgs units

# TODO: 

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
    get_cloudy_el_interpolator,
)

from yt_derived_fields.spectral_utils.stromgren_sphere_corrector import (
    apply_stromgren_correction,
    get_line_list_map_dict,
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

            #
            # Calculates the expected dust depletion
            #
            # ! Get the depletion factors
            # ! RR14 is based on the BARE-GR-S model of Zubko et al 2004
            # ! See Table 5 which is where we generatred the fractional contributions 
            # ! of each element

            nH = data["gas", "hydrogen_number_density"].in_units("cm**-3").d
            nO = data["gas", "oxygen_number_density"].in_units("cm**-3").d
            T = data["gas", "temperature"].in_units("K").d

            if isinstance(data, FieldDetector):
                return np.zeros(nH.shape)

            depletion_table = np.ones(len(nH))
            filt = np.log10(T) < 6.0

            x = 12.0 + (np.log10(nO) - np.log10(nH))

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
        mylog.info("Setting up interpolation for unresolved Stromgren sphere corrections.")
        # Get the interpolation matrix
        # (metal, O/H, nH, ages, ionLum, C/O) --> list of line luminosities (erg/s)
        mif_cloudy, line_list = get_cloudy_el_interpolator()
        # Mapper from the cloudy line names to the field names we want to use in yt
        line_list_map_dict = get_line_list_map_dict()

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

                volume = data["gas", "cell_volume"].to("cm**3").value

                nH = data["gas", "hydrogen_number_density"].in_units("cm**-3").d
                nO = data["gas", "oxygen_number_density"].in_units("cm**-3").d
                nC = data["gas", "carbon_number_density"].in_units("cm**-3").d
                nN = data["gas", "nitrogen_number_density"].in_units("cm**-3").d
                nNe = data["gas", "neon_number_density"].in_units("cm**-3").d
                nS = data["gas", "sulfur_number_density"].in_units("cm**-3").d

                O_depletion = np.array(data["gas", "O_dep"])
                C_depletion = np.array(data["gas", "C_dep"])
                N_depletion = np.array(data["gas", "N_dep"])
                Ne_depletion = np.array(data["gas", "Ne_dep"])
                S_depletion = np.array(data["gas", "S_dep"])

                star_age = data["deposit", "young_pop2_avg_age"].in_units("Myr").d
                star_metal = np.array(data["deposit", "young_pop2_avg_metallicity"])
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
            loc_emis *= ne * nel * xion

            if fix_unres_stromgren and cells_to_replace.sum() > 0:

                O_over_H = np.log10(nO) - np.log10(nH) - np.log10(4.90E-04)
                C_over_H = np.log10(nC) - np.log10(nH) - np.log10(2.69E-04)
                N_over_H = np.log10(nN) - np.log10(nH) - np.log10(6.76E-05)
                Ne_over_H = np.log10(nNe) - np.log10(nH) - np.log10(8.51E-05)
                S_over_H = np.log10(nS) - np.log10(nH) - np.log10(1.32E-05)

                all_emission_lines = apply_stromgren_correction(
                    mif_cloudy, line_list,
                    cells_to_replace,
                    nH, nO, nC, nN, nNe, nS,
                    O_over_H, C_over_H, N_over_H, Ne_over_H, S_over_H,
                    O_depletion, C_depletion, N_depletion, Ne_depletion, S_depletion,
                    star_age, star_metal, star_ion_lums,
                )

                line_idx = line_index_map.get(line)
                if line_idx is not None:
                    loc_lum_cloudy = get_cloudy_line_luminosity(
                        all_emission_lines, line_idx, int(cells_to_replace.sum())
                    )
                else:
                    loc_lum_cloudy = np.zeros(int(cells_to_replace.sum()))

                loc_emis_cloudy = loc_lum_cloudy / volume[cells_to_replace]
                loc_emis[cells_to_replace] = loc_emis_cloudy

            return loc_emis * u.erg / u.s / u.cm**3

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

                volume = data["gas", "cell_volume"].to("cm**3").value

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
            loc_rec_emis *= ne * nel * xion  # NOTE * df_gas[f"{el}_dep"] # erg/s/cm^3

            loc_col_emis = rec_line_dict[line]["emis_grid_col"](to_interp_ch)
            loc_col_emis *= ne * nel * xion_col  # NOTE * df_gas[f"{el}_dep"] # erg/s/cm^3

            loc_emis = loc_rec_emis + loc_col_emis

            if fix_unres_stromgren and cells_to_replace.sum() > 0:

                O_over_H = np.log10(nO) - np.log10(nH) - np.log10(4.90E-04)
                C_over_H = np.log10(nC) - np.log10(nH) - np.log10(2.69E-04)
                N_over_H = np.log10(nN) - np.log10(nH) - np.log10(6.76E-05)
                Ne_over_H = np.log10(nNe) - np.log10(nH) - np.log10(8.51E-05)
                S_over_H = np.log10(nS) - np.log10(nH) - np.log10(1.32E-05)

                all_emission_lines = apply_stromgren_correction(
                    mif_cloudy, line_list,
                    cells_to_replace,
                    nH, nO, nC, nN, nNe, nS,
                    O_over_H, C_over_H, N_over_H, Ne_over_H, S_over_H,
                    O_depletion, C_depletion, N_depletion, Ne_depletion, S_depletion,
                    star_age, star_metal, star_ion_lums,
                )

                line_idx = line_index_map.get(line)
                if line_idx is not None:
                    loc_lum_cloudy = get_cloudy_line_luminosity(
                        all_emission_lines, line_idx, int(cells_to_replace.sum())
                    )
                else:
                    loc_lum_cloudy = np.zeros(int(cells_to_replace.sum()))

                loc_emis_cloudy = loc_lum_cloudy / volume[cells_to_replace]

                print(f"Replacing {line} emissitivity from \n{loc_emis[cells_to_replace]} to \n{loc_emis_cloudy}")
                loc_emis[cells_to_replace] = loc_emis_cloudy

            return loc_emis * u.erg / u.s / u.cm**3

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
