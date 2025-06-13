# Generating derived fields for the stars module
# NOTE: for the MEGATRON simulations ran with RAMSES-RTZ

# Author: Anatole Storck

import yt
from yt import units as u
from yt_derived_fields.cosmology import conformal_time as ct

def create_star_derived_fields(ds):
    
    _initialize_star_age(ds)
    _initialize_pop3_star_filter(ds)


def _initialize_star_age(ds):

    # Usually, the birth time of a star in a cosmological simulation is stored in conformal
    # lookback time. For RT simulations, the birth time is instead in proper lookback time.

    # Initialize the conformal time class
    cosmology = ct.ConformalTime()

    # NOTE: conformal_time.f90 generates a grid of values relating aexp, t_frw, and tau_frw,
    #       so only need to do it once then save the grid to file.
    
    # Initialize the cosmology object with the cosmological parameters
    cosmology.ct_init_cosmo(
        omega_m =   ds.cosmology.omega_matter,   # 0.313899993896484E+00,
        omega_l =   ds.cosmology.omega_lambda,   # 0.686094999313354E+00,
        omega_k =   0.500679016113281E-05,       # ds.cosmology.omega_curvature (wrong)
        h0      =   ds.hubble_constant*100,      # h0 should be the H0 (not reduced)
    )

    # Get the expansion factor for the current redshift
    a = 1/(1 + ds.current_redshift)

    # Get the lookback time of the current redshift in years
    stimeYr = cosmology.ct_aexp2time(a) * u.yr
    
    
    def _star_age(field, data):
        
        # Birth time of the star particle
        # NOTE: while the field mentions conformal time, it is actually the proper time for RT sims
        star_birth_time_proper = data["star", "conformal_birth_time"]

        # Convert from proper time to time
        star_birth_time = cosmology.ct_proptime2time(tau=star_birth_time_proper,
                                                     h0=ds.hubble_constant*100) * u.yr

        # To get the age, subtract the birth time from the current time
        star_age = (stimeYr - star_birth_time).to("Myr")
        
        return star_age
    
    ds.add_field(name=("star", "age"),
                function=_star_age,
                force_override=True,
                units="Myr",
                sampling_type="particle",
                display_name="Star age",)
    
def _initialize_pop3_star_filter(ds):
    
    def _pop3_star_filter(filter, data):

        # Metallicity criteria for Pop. III stars in MEGATRON
        met_O = data[filter.filtered_type, "particle_metallicity_002"]
        met_Fe = data[filter.filtered_type, "particle_metallicity_001"]

        return (met_O * 2.09 + 1.06 * met_Fe) < 2e-8

    yt.add_particle_filter("pop3",
                          function=_pop3_star_filter,
                          requires=["particle_metallicity_002",
                                    "particle_metallicity_001"],
                          filtered_type="star")
    
    ds.add_particle_filter("pop3")
