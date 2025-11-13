# Generating derived fields for the stars module
# NOTE: for the MEGATRON simulations ran with RAMSES-RTZ

# Author: Anatole Storck

import yt
from yt import units as u
from yt_derived_fields.cosmology import conformal_time as ct

from yt_derived_fields.spectral_utils import pop2_stellar_spectra
from yt_derived_fields.spectral_utils import pop3_stellar_spectra


def create_star_derived_fields(ds, pop3=False, parallel=True):
    _initialize_star_age(ds)

    _initialize_pop2_star_filter(ds)
    _initialize_pop2_spectral_fields(ds, parallel=parallel)

    if pop3:
        _initialize_pop3_star_filter(ds)
        _initialize_pop3_aliveStatus(ds)
        _initialize_pop3_spectral_fields(ds)


def _initialize_star_age(ds):
    # Usually, the birth time of a star in a cosmological simulation is stored in conformal
    # lookback time. For RT simulations, the birth time is instead in proper lookback time.

    # Initialize the conformal time class
    cosmology = ct.ConformalTime()

    # NOTE: conformal_time.f90 generates a grid of values relating aexp, t_frw, and tau_frw,
    #       so only need to do it once then save the grid to file.

    # Initialize the cosmology object with the cosmological parameters
    cosmology.ct_init_cosmo(
        omega_m=ds.cosmology.omega_matter,  # 0.313899993896484E+00,
        omega_l=ds.cosmology.omega_lambda,  # 0.686094999313354E+00,
        omega_k=1.0 - (ds.cosmology.omega_matter + ds.cosmology.omega_lambda),  # 0.000005006690140E+00, (ds.cosmology.omega_curvature is wrong),
        h0=ds.hubble_constant * 100,  # h0 should be the H0 (not reduced)
    )

    # Get the expansion factor for the current redshift
    a = 1 / (1 + ds.current_redshift)

    # Get the lookback time of the current redshift in years
    stimeYr = cosmology.ct_aexp2time(a) * u.yr

    def _star_age(field, data):
        # Birth time of the star particle
        # NOTE: while the field mentions conformal time, it is actually the proper time for RT sims (At least in MEGATRON)
        star_birth_time_proper = data["star", "conformal_birth_time"].value

        # Convert from proper time to time
        star_birth_time = cosmology.ct_proptime2time(tau=star_birth_time_proper, h0=ds.hubble_constant * 100) * u.yr

        # To get the age, subtract the birth time from the current time
        star_age = (stimeYr - star_birth_time).to("Myr")

        return star_age

    ds.add_field(
        name=("star", "age"),
        function=_star_age,
        force_override=True,
        units="Myr",
        sampling_type="particle",
        display_name="Star age",
    )


def _initialize_pop2_star_filter(ds):
    """
    Initialize the Pop. II star filter.
    This filter is True if the star is a Pop. II star, False otherwise.
    """

    def _pop2_star_filter(filter, data):
        # Metallicity criteria for Pop. II stars in MEGATRON
        met_O = data["star", "particle_metallicity_002"].value
        met_Fe = data["star", "particle_metallicity_001"].value

        return (met_O * 2.09 + 1.06 * met_Fe) >= 2e-8

    yt.add_particle_filter(
        "pop2",
        function=_pop2_star_filter,
        requires=["particle_metallicity_002", "particle_metallicity_001"],
        filtered_type="star",
    )

    if "pop2" not in ds.filtered_particle_types:
        ds.add_particle_filter("pop2")


def _initialize_pop3_star_filter(ds):
    def _pop3_star_filter(filter, data):
        # Metallicity criteria for Pop. III stars in MEGATRON
        met_O = data["star", "particle_metallicity_002"].value
        met_Fe = data["star", "particle_metallicity_001"].value

        return (met_O * 2.09 + 1.06 * met_Fe) < 2e-8

    yt.add_particle_filter(
        "pop3",
        function=_pop3_star_filter,
        requires=["particle_metallicity_002", "particle_metallicity_001"],
        filtered_type="star",
    )

    if "pop3" not in ds.filtered_particle_types:
        ds.add_particle_filter("pop3")


def _initialize_pop3_aliveStatus(ds):
    """
    Initialize the Pop. III star alive status field.
    This field is True if the star is still alive, False otherwise.
    """

    import numpy as np

    def pop3_MSlifetime(mass):
        """
        Calculates the main-sequence lifetime of popIII stars of a given mass
        returns the age in Myr
        """
        # Schaerer (2002) interpolation of age
        a_fit = [0.7595398e00, -3.7303953e00, 1.4031973e00, -1.7896967e-01]

        age_gyr = 0.0
        logm = np.log10(mass)

        for i in range(len(a_fit)):
            age_gyr += a_fit[i] * (logm**i)

        age_gyr = 10.0**age_gyr
        return age_gyr * 1000.0  # returns Myr

    def isAlive(field, data):
        # Get the age of the star
        age = data["pop3", "age"].to("Myr").flatten()
        particle_mass = data["pop3", "particle_initial_mass"].to("Msun").value.flatten()

        # Get the main-sequence lifetime of the pop3
        ms_lifetime = pop3_MSlifetime(particle_mass) * u.Myr

        # A star is still alive if its age is less than its main-sequence lifetime
        return age < ms_lifetime

    ds.add_field(
        name=("pop3", "isAlive"),
        function=isAlive,
        force_override=True,
        sampling_type="particle",
        display_name="Pop. III Star Alive Status",
    )


def _initialize_pop2_spectral_fields(ds, parallel=True):
    def _pop2_spectra(field, data):

        pop2_spec = pop2_stellar_spectra.get_pop_2_spectrum(data, combined=False, parallel=parallel)

        return pop2_spec

    ds.add_field(
        name=("pop2", "spectra"),
        function=_pop2_spectra,
        #force_override=True,
        units="erg/s",
        sampling_type="particle",
        vector_field=True,
        display_name="Pop. II Star Spectra",
    )

    def _pop2_spectra_combined(field, data):

        pop2_spec_combined = pop2_stellar_spectra.get_pop_2_spectrum(data, combined=True, parallel=parallel)

        return pop2_spec_combined

    ds.add_field(
        name=("pop2", "spectra_combined"),
        function=_pop2_spectra_combined,
        #force_override=True,
        units="erg/s",
        sampling_type="particle",
        vector_field=True,
        display_name="Pop. II Combined Star Spectra",
    )


    # An attempt to generate a spectra per cell. The idea is to check the stars in
    # each cell and assign them the summed stellar spectra. In practice, one would
    # want to use this on bins of cells, such as to create an IFU with a regular grid.
    # This would also reduce the memory overhead, as assigning a spectra per cell
    # over millions of cells is quite memory intensive.

    # TODO: Currently the deposition method works on the fluid grid. I need to check
    #       if it is possible to deposit on a different grid, such as a regular (coarser) grid,
    #       and if it is possible to project the regular grid into an IFU datacube,
    #       both on and off axis.
    def _deposit_pop2_spectra(ds) -> None:

        fname = ds.add_deposited_particle_field(
            ("pop2", "spectra"), method="sum", vector_field=True
        )


def _initialize_pop3_spectral_fields(ds):
    def _pop3_spectra(field, data):

        pop3_spec = pop3_stellar_spectra.get_pop_3_spectrum(data, combined=False)

        return pop3_spec

    ds.add_field(
        name=("pop3", "spectra"),
        function=_pop3_spectra,
        #force_override=True,
        units="erg/s",
        sampling_type="particle",
        vector_field=True,
        display_name="Pop. III Star Spectra",
    )

    def _pop3_spectra_combined(field, data):

        pop3_spec_combined = pop3_stellar_spectra.get_pop_3_spectrum(data, combined=True)

        return pop3_spec_combined

    ds.add_field(
        name=("pop3", "spectra_combined"),
        function=_pop3_spectra_combined,
        #force_override=True,
        units="erg/s",
        sampling_type="particle",
        vector_field=True,
        display_name="Pop. III Combined Star Spectra",
    )

    def _deposit_pop3_spectra(ds) -> None:

        fname = ds.add_deposited_particle_field(
            ("pop3", "spectra"), method="sum", vector_field=True
        )
