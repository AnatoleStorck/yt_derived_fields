# An attempt to generate a spectra per cell. The idea is to check the stars in
# each cell and assign a stellar spectra to each. In practice, one would want
# to use this on bins of cells, such as to create an IFU with a regular grid.

# TODO: Add the ability to generate spectra for a given cell
# (set everything to zero then add the spectra of the stars in that cell)

import numpy as np

from yt_derived_fields.spectral_utils import pop2_stellar_spectra
from yt_derived_fields.spectral_utils import pop3_stellar_spectra
import yt_derived_fields.megatron_derived_fields.stars_derived_fields as stars_derived_fields



def _initialize_pop2_spectra(ds):
    def pop2_spectra(field, data):
        Npop2 = np.sum(data["pop2", "particle_ones"])

        gas_cell_ids = data["gas", "cell_index"]
        gas_pop2_spectra = np.zeros_like(gas_cell_ids)

        if Npop2 > 0:
            pop2_spec = pop2_stellar_spectra.get_pop_2_spectrum(data, combined=False)

            stars_cell_ids = data["pop2", "cell_index"]

            unique_star_cell_ids = np.unique(stars_cell_ids)

            for unique_star_cell_id in unique_star_cell_ids:
                gas_pop2_spectra += np.sum(
                    [
                        pop2_spec[index]
                        for index in stars_cell_ids[
                            stars_cell_ids == unique_star_cell_id
                        ]
                    ]
                )

        return gas_pop2_spectra

    ds.add_field(
        name=("gas", "pop2_spectra"),
        function=_initialize_pop2_spectra,
        units="erg/s",  # NOTE: check if this or "erg/s/A"
        sampling_type="cell",
        display_name="Population 2 Spectra",
    )


def _initialize_pop3_spectra(ds):
    def pop3_spectra(field, data):
        Npop3_alive = np.sum(data["pop3", "isAlive"])

        gas_cell_ids = data["gas", "cell_index"]
        gas_pop3_spectra = np.zeros_like(gas_cell_ids)

        if Npop3_alive > 0:
            pop3_spec = pop3_stellar_spectra.get_pop_3_spectrum(data, combined=False)

            stars_cell_ids = data["pop3", "cell_index"]

            unique_star_cell_ids = np.unique(stars_cell_ids)

            for unique_star_cell_id in unique_star_cell_ids:
                gas_pop3_spectra += np.sum(
                    [
                        pop3_spec[index]
                        for index in stars_cell_ids[
                            stars_cell_ids == unique_star_cell_id
                        ]
                    ]
                )

        return gas_pop3_spectra

    ds.add_field(
        name=("gas", "pop3_spectra"),
        function=_initialize_pop3_spectra,
        units="erg/s",  # NOTE: check if this or "erg/s/A"
        sampling_type="cell",
        display_name="Population 3 Spectra",
    )


def get_stellar_continuum(ds):
    stars_derived_fields.create_star_derived_fields(ds)
    _initialize_pop2_spectra(ds)
    _initialize_pop3_spectra(ds)
