# Initial file to initialize all derived fields depending on a few inputs

from yt_derived_fields.megatron_derived_fields import (
    chemistry_derived_fields,
    cooling_derived_fields,
    emission_derived_fields,
    nebular_continuum_fields,
    radiative_derived_fields,
    stars_derived_fields,
)

def create_derived_fields(
    ds,
    simple_ne=False,
    H2_cooling="moseley",
    pop3_stars=False
    ):
    """
    Create all derived fields for MEGATRON datasets.

    Parameters:
    - ds: yt dataset
    - simple_ne: whether to use simple electron number density calculation (no metal electrons)
    - H2_cooling: method for H2 cooling ('moseley', or 'H2GP')
    - pop3_stars: whether to initialize Pop. III star filters and fields
    - kwargs: additional arguments for specific field initializations
    """

    # Create chemistry derived fields
    chemistry_derived_fields.create_chemistry_derived_fields(ds, simple_ne=simple_ne)

    # Create cooling derived fields
    cooling_derived_fields.create_cooling_derived_fields(ds, H2_cooling=H2_cooling)

    # Create emission derived fields
    emission_derived_fields.get_emission_lines(ds, all_lines=True)

    # Create nebular continuum derived fields
    nebular_continuum_fields.create_nebular_continuum_derived_fields(ds)

    # Create radiative derived fields
    radiative_derived_fields.create_rt_derived_fields(ds)

    # Create star derived fields
    stars_derived_fields.create_star_derived_fields(ds, pop3_stars=pop3_stars)