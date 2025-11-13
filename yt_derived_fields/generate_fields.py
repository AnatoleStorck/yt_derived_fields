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
    pop3_stars=False,
    parallel=True,
    ):
    """
    Create all derived fields for MEGATRON datasets. A few things to note: Some runs don't include Population III star
    so the creation of pop3 fields is disabled by default. Most simulations were run if the moseley H2 cooling model, but
    some were run with H2GP (not sure about which sims used which model.)
    
    Parameters:
    - ds: yt dataset
    - simple_ne: whether to use simple electron number density calculation (no metal electrons)
                 will skip the reading of many variables (all metal fields) at the cost of accuracy
    - H2_cooling: method for H2 cooling ('moseley', or 'H2GP')
    - pop3_stars: whether to initialize Pop. III star filters and fields
    - parallel: whether to parallelize some of the field calculations if applicable, using joblib.
    """

    # Create chemistry derived fields
    chemistry_derived_fields.create_chemistry_derived_fields(ds, ne_use_primordial_only=simple_ne)

    # Create cooling derived fields
    cooling_derived_fields.create_cooling_derived_fields(ds, H2_cooling=H2_cooling)

    # Create emission derived fields
    emission_derived_fields.get_emission_lines(ds, all_lines=True)

    # Create nebular continuum derived fields
    nebular_continuum_fields.create_nebular_continuum_derived_fields(ds, parallel=parallel)

    # Create radiative derived fields
    radiative_derived_fields.create_rt_derived_fields(ds)

    # Create star derived fields
    stars_derived_fields.create_star_derived_fields(ds, pop3=pop3_stars, parallel=parallel)