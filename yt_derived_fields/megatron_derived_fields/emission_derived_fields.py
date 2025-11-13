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

import chemistry_data as chem_data
import numpy as np
from roman import fromRoman

from yt import units as u
from yt.fields.field_detector import FieldDetector
from yt.funcs import mylog

from yt_derived_fields.megatron_derived_fields import (
    chemistry_derived_fields as chem_fields,
    chemistry_data as chem_data,
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


def get_emission_lines(ds, coll_lines=None, rec_lines=None, all_lines=False):
    """
    Add emission line luminosity fields to the dataset, given line emissivity grids have been generated on disk.

    Args:
        ds (yt.Dataset): The dataset object.
        coll_lines (list, optional): A list of collision lines to consider. Defaults to None.
                                     Example: ["O3-5007", "S2-6731"]
        rec_lines (list, optional): A list of recombination lines to consider. Defaults to None.
                                     Example: ["Lya", "Hb", "He-1640"]
    """

    # generate chemistry fields if not already present
    if not ("gas", "electron_number_density") in ds.derived_field_list:
        chem_fields.create_chemistry_derived_fields(ds)

    # These dictionaries are used to store the emission line metadata, along with interpolation grids
    # Can be generating using the generate_atomic_grids.py script (To contain more lines or finer interpolation)
    if coll_lines is not None or all_lines:
        coll_line_dict = get_coll_line_dict()
    if rec_lines is not None or all_lines:
        rec_line_dict = get_rec_line_dict()

    # line in the form of, for example, "O3-5007"
    def coll_line(ds, line):
        def _get_coll_line_emissivity(field, data):
            nCells = len(data["gas", "density"].to("g/cm**3").value)
            rho = data["gas", "density"].to("g/cm**3").value

            Tgas = np.log10(data["gas", "temperature"].to("K").value)

            ne = data["gas", "electron_number_density"].to("cm**-3").value

            # ----------------------------------------------------------

            el = coll_line_dict[line]["ion"].split("_")[0]  # C,    O,      Fe
            ion_roman = coll_line_dict[line]["ion"].split("_")[1]  # II,   III,    VII

            nel = data["gas", f"{met_data[el]['name']}_number_density"].to("cm**-3").value
            xion = data["gas", f"{met_data[el]['name']}_{fromRoman(ion_roman):02d}"].value

            if isinstance(data, FieldDetector):
                return np.zeros(rho.shape) * u.erg / u.s

            # Set up the arrays to interpolate
            to_interp = np.zeros((nCells, 2))
            to_interp[:, 0] = Tgas
            to_interp[:, 1] = np.log10(ne)

            # get emisitivity of cells based on T and ne
            loc_emis = coll_line_dict[line]["emis_grid"](to_interp)

            # Multiply by the electron density and ion density
            # n{el} = nO, ion = O_IV
            loc_emis *= ne * nel * xion

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
            nCells = len(data["gas", "density"].to("g/cm**3").value)
            rho = data["gas", "density"].to("g/cm**3").value

            Tgas = np.log10(data["gas", "temperature"].to("K").value)

            ne = data["gas", "electron_number_density"].to("cm**-3").value

            # ----------------------------------------------------------

            el = rec_line_dict[line]["ion"].split("_")[0]
            ion_roman = rec_line_dict[line]["ion"].split("_")[1]

            nel = data["gas", f"{prim_data[el]['name']}_number_density"].to("1/cm**3").value

            xion = data["gas", f"{prim_data[el]['name']}_{fromRoman(ion_roman):02d}"].value
            xion_col = data["gas", f"{prim_data[el]['name']}_{fromRoman(ion_roman) - 1:02d}"].value

            if isinstance(data, FieldDetector):
                return np.zeros(rho.shape) * u.erg / u.s

            # Set up the arrays to interpolate
            to_interp = np.zeros((nCells, 2))
            to_interp[:, 0] = Tgas
            to_interp[:, 1] = np.log10(ne)

            min_temp = 10.0

            to_interp_ch = 10.0 ** np.array(Tgas)
            to_interp_ch[to_interp_ch < min_temp] = min_temp
            to_interp_ch[to_interp_ch > 1e9] = 1e9

            loc_rec_emis = rec_line_dict[line]["emis_grid"](to_interp)
            loc_rec_emis *= ne * nel * xion  # NOTE * df_gas[f"{el}_dep"]

            loc_col_emis = rec_line_dict[line]["emis_grid_col"](to_interp_ch)
            loc_col_emis *= ne * nel * xion_col  # NOTE * df_gas[f"{el}_dep"]

            loc_rec_lum = loc_rec_emis  # erg/s/cm^3
            loc_col_lum = loc_col_emis  # erg/s/cm^3

            return (loc_rec_lum + loc_col_lum) * u.erg / u.s / u.cm**3

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
