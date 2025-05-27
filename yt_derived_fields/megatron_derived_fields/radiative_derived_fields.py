# Generating derived fields for the radiation transfer module
# NOTE: for the MEGATRON simulations ran with RAMSES-RTZ

# Author: Anatole Storck

from yt import units as u
import yt

ev_to_erg = 1.60218e-12     # eV/erg

# The eight energy groups in MEGATRON
energy_bands = {"IR": 1, "Opt.": 2, "FUV": 3, "LW": 4, "EUV1": 5, "EUV2": 6, "EUV3": 7, "EUV4": 8}

def create_rt_derived_fields(ds, bands="all"):
    """Create derived fields for the radiation transfer module.

    Args:
        ds (yt.Dataset): The dataset object.
        bands (str, optional): The radiation bands to initialize. The bands are IR, Opt., FUV, LW, EUV1, EUV2, EUV3, EUV4. Defaults to "all".
    """

    if bands == "all":
        for band in energy_bands.keys():
            _initialize_radiation_energy_density(ds, band)
    else:
        _initialize_radiation_energy_density(ds, bands)


def _initialize_radiation_energy_density(ds, band: str):
    
    rt_params = yt.frontends.ramses.fields.RTFieldFileHandler.get_rt_parameters(ds)
    energy_conversions = {band: rt_params[f"Group {i} egy      [eV]"][0] for band, i in energy_bands.items()}
    
    def _radiation_energy_density(field, data):
        
        photon_field = data["ramses-rt", f"Photon_density_{energy_bands[band]}"]
        photon_density_field = photon_field * rt_params["unit_pf"] * ev_to_erg * energy_conversions[band]

        return photon_density_field * u.erg/u.s/u.cm**2

    ds.add_field(name=("gas", f"radiation_energy_density_{band}"),
                 function=_radiation_energy_density,
                 units="erg/cm**3",
                 sampling_type="cell",
                 display_name=f"Radiation energy density ({band})")