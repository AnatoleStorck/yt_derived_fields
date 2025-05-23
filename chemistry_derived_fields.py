# Generating derived fields for the chemistry module
# NOTE: for the MEGATRON simulations ran with RAMSES-RTZ

# TODO: Introduce the YT TOML file for MEGATRON, grabbing the fields from "gas" instead of "ramses"

# Author: Anatole Storck

from yt import units as u
import numpy as np

# NOTE: Nion is the number of ionization
#       states tracked in MEGATRON
metal_data = {
    "O":    {"name" : "oxygen",     "mass": 15.9994 * u.amu,     "Nion": 8 },
    "Ne":   {"name" : "neon",       "mass": 20.1797 * u.amu,     "Nion": 10},
    "C":    {"name" : "carbon",     "mass": 12.0107 * u.amu,     "Nion": 6 },
    "S":    {"name" : "sulfur",     "mass": 32.065  * u.amu,     "Nion": 11},
    "Mg":   {"name" : "magnesium",  "mass": 24.305  * u.amu,     "Nion": 10},
    "Si":   {"name" : "silicon",    "mass": 28.0855 * u.amu,     "Nion": 11},
    "N":    {"name" : "nitrogen",   "mass": 14.0067 * u.amu,     "Nion": 7 },
    "Fe":   {"name" : "iron",       "mass": 55.854  * u.amu,     "Nion": 11},
    "Ca":   {"name" : "calcium",    "mass": 40.078  * u.amu,     "Nion": 0 }
}
prim_data = {
    "H":    {"name" : "hydrogen",    "mass": 1.00794  * u.amu,   "Nion": 2,     "massFrac": 0.76},
    "He":   {"name" : "helium",      "mass": 4.002602 * u.amu,   "Nion": 3,     "massFrac": 0.24}
}
molec_data = {
    "H2":   {"name" : "molecular hydrogen", "mass": 2.01588 * u.amu},
    "CO":   {"name" : "carbon monoxide",    "mass": (15.9994 + 12.0107) * u.amu},
}

def _initialize_metal_density(ds, element: str):
    metal_name = metal_data[element]["name"]
    def _metal_density(field, data):
        
        # # IF NO TOML
        # if element == "Fe":
        #     metal_density = data["gas", "density"] * data["ramses", "Metallicity"]
        # else:
        #     metal_density = (data["gas", "density"] *
        #                      data["ramses", f"hydro_{metal_name}_fraction"])
        
        if element == "Fe":
            metal_density = data["gas", "density"] * data["ramses", "Metallicity"]
        else:
            metal_density = (data["gas", "density"] *
                            data["gas", f"{metal_name}_fraction"])

        return metal_density
    
    ds.add_field(name=("gas", f"{metal_name}_density"),
                function=_metal_density,
                units="amu/cm**3",
                sampling_type="cell",
                display_name=f"{metal_name} density",)

def _initialize_metallicity(ds):
    
    def _metallicity(field, data):
        rho = data["gas", "density"]
        Z = np.zeros_like(rho)
        for element in metal_data:
            Z += data["gas", f"{element}_density"]
        Z /= rho
        return Z

    ds.add_field(name=("gas", "metallicity"),
                function=_metallicity,
                units='1',
                sampling_type="cell",
                display_name="Metallicity",)

def _initialize_primordial_density(ds, element):
    prim_name = prim_data[element]["name"]
    def _primordial_density(field, data):
        return (data["gas", "density"] *
                prim_data[element]["massFrac"] *
                (1 - data["gas", "metallicity"]))
    
    ds.add_field(name=("gas", f"{prim_name}_density"),
                function=_primordial_density,
                units="amu/cm**3",
                sampling_type="cell",
                display_name=f"{prim_name} density",)
    
def _initialize_primordial_number_density(ds, element):
    prim_name = prim_data[element]["name"]
    def _primordial_number_density(field, data):
        return data["gas", f"{prim_name}_density"] / prim_data[element]["mass"]
    
    ds.add_field(name=("gas", f"{prim_name}_number_density"),
                function=_primordial_number_density,
                units="cm**-3",
                sampling_type="cell",
                display_name=f"{prim_name} number density",)
    
def _initialize_H2_number_density(ds):
    
    def _H2_number_density(field, data):
        
        xHI = data["gas", "hydrogen_01"] # ["ramses", "hydro_H_01"]
        xHII = data["gas", "hydrogen_02"] # ["ramses", "hydro_H_02"]

        xH2 = 1 - xHI - xHII
        return data["gas", "hydrogen_number_density"] * xH2 / 2
    
    ds.add_field(name=("gas", "H2_number_density"),
                function=_H2_number_density,
                units="cm**-3",
                sampling_type="cell",
                display_name="H2 number density",)
    
def _initialize_CO(ds):
    
    def _CO_density(field, data):
        return data["gas", "density"] * data["gas", "CO_fraction"] # ["ramses", "hydro_CO_fraction"]
    
    def _CO_number_density(field, data):
        return data["gas", "CO_density"] / molec_data["CO"]["mass"]

    ds.add_field(name=("gas", "CO_density"),
                function=_CO_density,
                units="amu/cm**3",
                sampling_type="cell",
                display_name="CO density",)
    
    ds.add_field(name=("gas", "CO_number_density"),
                function=_CO_number_density,
                units="cm**-3",
                sampling_type="cell",
                display_name="CO number density",)

def _initialize_electron_number_density(ds):
    
    def _electron_number_density(field, data):

        # Ionized hydrogen electrons
        xHII = data["gas", "hydrogen_02"] # ["rames", "hydro_H_02"]
        nHII = data["gas", "hydrogen_number_density"] * xHII
        
        # Ionized helium electrons
        xHeII = data["gas", "helium_02"] # ["ramses", "hydro_He_02"]
        xHeIII = data["gas", "helium_03"] # ["ramses", "hydro_He_03"]

        nHeII = data["gas", "helium_number_density"] * xHeII
        nHeIII = data["gas", "helium_number_density"] * xHeIII
        
        # Ionized metals electrons
        nEl_ion = np.zeros_like(nHII)
        for element in metal_data:
            if element == "Ca": # No out of equilibrium information
                continue
            metal_name = metal_data[element]["name"]
            nEl = data["gas", f"{metal_name}_density"] / metal_data[element]["mass"]
            for ion in range(metal_data[element]["Nion"]):
                xEl_ion = data["gas", f"{metal_name}_{ion+1:02d}"]  # ["ramses", f"hydro_{metal_name}_{ion+1:02d}"]
                nEl_ion += ion * nEl * xEl_ion
                
        return nHII + nHeII + nHeIII + nEl_ion
    
    ds.add_field(name=("gas", "electron_number_density"),
                function=_electron_number_density,
                units="cm**-3",
                sampling_type="cell",
                display_name="Electron number density",)
    
def _initialize_mean_molecular_weight(ds):
    
    def _mean_molecular_weight(field, data):
        
        rhoH = data["gas", "hydrogen_density"]
        rhoHe = data["gas", "helium_density"]
        
        rhoMet = np.zeros_like(rhoH) 
        for element in metal_data:
            metal_name = metal_data[element]["name"]
            rhoMet += data["gas", f"{metal_name}_density"]
        
        rhoCO = data["gas", "CO_density"]
        
        sum_density = (rhoH + rhoHe + rhoMet + rhoCO) / u.amu
        
        xHI = data["gas", "hydrogen_01"] # ["ramses", "hydro_H_01"]
        xHII = data["gas", "hydrogen_02"] # ["ramses", "hydro_H_02"]
        
        nH = data["gas", "hydrogen_number_density"] * (xHI + xHII) + data["gas", "H2_number_density"]
        
        nHe = data["gas", "helium_number_density"]
        nCO = data["gas", "CO_number_density"]
        
        nMet = np.zeros_like(nH)
        
        for element in metal_data:
            metal_name = metal_data[element]["name"]
            nMet += data["gas", f"{metal_name}_density"] / metal_data[element]["mass"]

        ne = data["gas", "electron_number_density"]

        sum_number_density = (nH + nHe + nCO + nMet) + ne
        
        return sum_density / sum_number_density
    
    ds.add_field(name=("gas", "mu"),
                function=_mean_molecular_weight,
                units="1",
                sampling_type="cell",
                display_name="Mean molecular weight",)

def create_chemistry_derived_fields(ds,
                                   molecules=True,
                                   electron_number_density=True,
                                   mean_molecular_weight=True):
    """
    Initialize the derived fields for the chemistry module.
    
    Parameters
    ----------
    ds : yt.Dataset
        The dataset object.
    electron_number_density : bool, optional
        If True, include the electron number density field, needing the metal ion fractions. Default is True.
    """
    
    # Add fields for metal densities
    for element in metal_data:
        _initialize_metal_density(ds, element)
    
    _initialize_metallicity(ds)
    
    for element in prim_data:
        _initialize_primordial_density(ds, element)
        _initialize_primordial_number_density(ds, element)

    if electron_number_density:
        _initialize_electron_number_density(ds)
        
    if molecules:
        _initialize_H2_number_density(ds)
        _initialize_CO(ds)
        
        # Can't calculate mean molecular weight without CO
        if mean_molecular_weight:
            _initialize_mean_molecular_weight(ds)

