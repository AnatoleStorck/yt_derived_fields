# An attempt to generate a spectra per cell. The idea is to check the stars in
# each cell and assign a stellar spectra to each. In practice, one would want
# to use this on bins of cells, such as to create an IFU with a regular grid.

# TODO: Add the ability to generate spectra for a given cell
# (set everything to zero then add the spectra of the stars in that cell)

import yt_derived_fields.megatron_derived_fields.stars_derived_fields as stars_derived_fields

def _initialize_pop2_spectra(ds) -> None:

    fname = ds.add_deposited_particle_field(
        ("pop2", "spectra"), method="sum", vector_field=True
    )
    print(f"Pop. II spectra field added: {fname}")


def _initialize_pop3_spectra(ds):
    if ("pop3", "spectra") not in ds.derived_field_list:
        return
    fname = ds.add_deposited_particle_field(
        ("pop3", "spectra"), method="sum", vector_field=True
    )
    print(f"Pop. III spectra field added: {fname}")


def get_stellar_continuum(ds):
    stars_derived_fields.create_star_derived_fields(ds)
    _initialize_pop2_spectra(ds)
    _initialize_pop3_spectra(ds)
