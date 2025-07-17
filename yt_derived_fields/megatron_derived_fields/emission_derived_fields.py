# Generating derived fields for the emission lines module
# NOTE: for the MEGATRON simulations ran with RAMSES-RTZ

# This module will follow closely the approach of
# Harley Katz for computing spectra and emission lines

# Everything normally needs to be in cgs units

# TODO: Add corections for unresolved stromgren spheres

# Author: Anatole Storck

from yt import units as u
from yt.fields.field_detector import FieldDetector

from pathlib import Path
from roman import fromRoman
import numpy as np

from yt_derived_fields.megatron_derived_fields import (
    chemistry_derived_fields as chem_fields,
)
from yt_derived_fields.megatron_derived_fields import (
    chemistry_derived_fields as chem_fields,
)

met_data = {
    "O": {"name": "oxygen", "mass": 15.9994 * u.amu, "Nion": 8},
    "Ne": {"name": "neon", "mass": 20.1797 * u.amu, "Nion": 10},
    "C": {"name": "carbon", "mass": 12.0107 * u.amu, "Nion": 6},
    "S": {"name": "sulfur", "mass": 32.065 * u.amu, "Nion": 11},
    "Mg": {"name": "magnesium", "mass": 24.305 * u.amu, "Nion": 10},
    "Si": {"name": "silicon", "mass": 28.0855 * u.amu, "Nion": 11},
    "N": {"name": "nitrogen", "mass": 14.0067 * u.amu, "Nion": 7},
    "Fe": {"name": "iron", "mass": 55.854 * u.amu, "Nion": 11},
}
prim_data = {
    "H": {"name": "hydrogen", "mass": 1.00784 * u.amu, "massFrac": 0.76},
    "He": {"name": "helium", "mass": 4.002602 * u.amu, "massFrac": 0.24},
}

_spectral_data = Path(__file__).parent.parent / "spectral_utils"


def get_emission_lines(ds, coll_lines=None, rec_lines=None, all_lines=False):
    """
    Add emission line luminosity fields to the dataset.

    Args:
        ds (yt.Dataset): The dataset object.
        coll_lines (list, optional): A list of collision lines to consider. Defaults to None.
                                     Example: ["O3-5007", "S2-6731"]
        rec_lines (list, optional): A list of recombination lines to consider. Defaults to None.
                                     Example: ["Lya", "Hb", "He-1640"]
    """

    # Need to generate the chemistry derived fields first, as we need to calculate the electron number density
    chem_fields.create_chemistry_derived_fields(ds, molecules=False, mean_molecular_weight=False)

    # These dictionaries are used to store the emission line metadata, along with interpolation grids
    # Can be generating using the generate_atomic_grids.py script (To contain more lines or finer interpolation)
    if coll_lines is not None or all_lines:
        coll_line_dict = np.load(f"{_spectral_data}/coll_line_dict.npy", allow_pickle=True).item()
    if rec_lines is not None or all_lines:
        rec_line_dict = np.load(f"{_spectral_data}/rec_line_dict.npy", allow_pickle=True).item()

    # line in the form of, for example, "O3-5007"
    def coll_line(ds, line):
        def _get_coll_line_lum(field, data):
            nCells = len(data["gas", "density"].to("g/cm**3").value)
            rho = data["gas", "density"].to("g/cm**3").value

            Tgas = np.log10(data["gas", "temperature"].to("K").value)

            ne = data["gas", "electron_number_density"].to("cm**-3").value
            cell_vol = data["gas", "volume"].to("cm**3").value

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
            loc_lum = loc_emis * cell_vol  # erg/s

            return loc_lum * u.erg / u.s

        ds.add_field(
            name=("gas", f"{line}_luminosity"),
            function=_get_coll_line_lum,
            units="erg/s",
            sampling_type="cell",
            display_name=f"{line} Luminosity",
        )

    # line in the form of "Lya", "Hb", "He-1640"
    def rec_line(ds, line):
        def _get_rec_line_lum(field, data):
            nCells = len(data["gas", "density"].to("g/cm**3").value)
            rho = data["gas", "density"].to("g/cm**3").value

            Tgas = np.log10(data["gas", "temperature"].to("K").value)

            ne = data["gas", "electron_number_density"].to("cm**-3").value
            cell_vol = data["gas", "volume"].to("cm**3").value

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

            loc_rec_lum = loc_rec_emis * cell_vol  # erg/s
            loc_col_lum = loc_col_emis * cell_vol  # erg/s

            return (loc_rec_lum + loc_col_lum) * u.erg / u.s

        ds.add_field(
            name=("gas", f"{line}_luminosity"),
            function=_get_rec_line_lum,
            units="erg/s",
            sampling_type="cell",
            display_name=f"{line} Luminosity",
        )

    if all_lines:
        [coll_line(ds, line) for line in coll_line_dict.keys()]
        [rec_line(ds, line) for line in rec_line_dict.keys()]
    else:
        for line in coll_lines:
            coll_line(ds, line)
        for line in rec_lines:
            rec_line(ds, line)
