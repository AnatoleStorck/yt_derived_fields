# Generating derived fields for the nebular continuum lines module
# NOTE: for the MEGATRON simulations ran with RAMSES-RTZ

# Author: Anatole Storck

from yt_derived_fields.megatron_derived_fields import chemistry_derived_fields as chem_fields
from yt_derived_fields.spectral_utils.nebular_continuum import (
    get_nebular_continuum_recombination,
    get_nebular_continuum_two_photon
)

# Add nebular continuum derived fields to the dataset
# This includes:
# - Recombination continuum
# - Two-photon continuum
# - Total nebular continuum (recombination + two-photon)
# For each of these, both cell-by-cell or combined spectra can be obtained.
def create_nebular_continuum_derived_fields(ds, parallel=True):
    # generate chemistry fields if not already present
    if not ("gas", "electron_number_density") in ds.derived_field_list:
        chem_fields.create_chemistry_derived_fields(ds)
    _initialize_nebular_continuum(ds, parallel=parallel)



def _initialize_nebular_continuum(ds, parallel=True):

    def nebc_recomb(ds):
        def _get_nebc_recomb(field, data):

            nebc_spec = get_nebular_continuum_recombination(data, parallel=parallel)

            return nebc_spec

        ds.add_field(
            name=("gas", "nebc_recomb"),
            function=_get_nebc_recomb,
            units="erg/s",  # NOTE: check if this or "erg/s/A"
            sampling_type="cell",
            display_name="Nebular Continuum Resolved Recombination",
        )

        def _get_nebc_recomb_combined(field, data):

            nebc_spec = get_nebular_continuum_recombination(data, combined=True, parallel=parallel)

            return nebc_spec
        
        ds.add_field(
            name=("gas", "nebc_recomb_combined"),
            function=_get_nebc_recomb_combined,
            units="erg/s",  # NOTE: check if this or "erg/s/A"
            sampling_type="cell",
            display_name="Nebular Continuum Resolved Recombination Combined",
        )

    def nebc_two_photon(ds):
        def _get_nebc_two_photon(field, data):

            nebc_spec = get_nebular_continuum_two_photon(data)

            return nebc_spec

        ds.add_field(
            name=("gas", "nebc_two_photon"),
            function=_get_nebc_two_photon,
            units="erg/s",  # NOTE: check if this or "erg/s/A"
            sampling_type="cell",
            display_name="Nebular Continuum Resolved Two Photon",
        )

        def _get_nebc_two_photon_combined(field, data):

            nebc_spec = get_nebular_continuum_two_photon(data, combined=True)

            return nebc_spec

        ds.add_field(
            name=("gas", "nebc_two_photon_combined"),
            function=_get_nebc_two_photon_combined,
            units="erg/s",  # NOTE: check if this or "erg/s/A"
            sampling_type="cell",
            display_name="Nebular Continuum Resolved Two Photon Combined",
        )

    def nebc_total(ds):

        def _get_nebc_total(field, data):

            nebc_spec = get_nebular_continuum_recombination(data, parallel=parallel) + get_nebular_continuum_two_photon(data)

            return nebc_spec

        ds.add_field(
            name=("gas", "nebc_total"),
            function=_get_nebc_total,
            units="erg/s",  # NOTE: check if this or "erg/s/A"
            sampling_type="cell",
            display_name="Nebular Continuum Resolved Total",
        )
        def _get_nebc_total_combined(field, data):

            nebc_spec = get_nebular_continuum_recombination(data, combined=True, parallel=parallel) + get_nebular_continuum_two_photon(data, combined=True)

            return nebc_spec

        ds.add_field(
            name=("gas", "nebc_total_combined"),
            function=_get_nebc_total_combined,
            units="erg/s",  # NOTE: check if this or "erg/s/A"
            sampling_type="cell",
            display_name="Nebular Continuum Resolved Total Combined",
        )

    # Add the fields to the dataset
    nebc_recomb(ds)
    nebc_two_photon(ds)
    nebc_total(ds)
