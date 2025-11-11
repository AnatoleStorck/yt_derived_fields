# This file contains the specific chemistry metadata for the MEGATRON simulations

from yt import units as u

# NOTE: Nion is the number of ionization
#       states tracked in MEGATRON
_metal_data = {
    "O": {"name": "oxygen", "mass": 15.9994 * u.amu, "Nion": 8},
    "Ne": {"name": "neon", "mass": 20.1797 * u.amu, "Nion": 10},
    "C": {"name": "carbon", "mass": 12.0107 * u.amu, "Nion": 6},
    "S": {"name": "sulfur", "mass": 32.065 * u.amu, "Nion": 11},
    "Mg": {"name": "magnesium", "mass": 24.305 * u.amu, "Nion": 10},
    "Si": {"name": "silicon", "mass": 28.0855 * u.amu, "Nion": 11},
    "N": {"name": "nitrogen", "mass": 14.0067 * u.amu, "Nion": 7},
    "Fe": {"name": "iron", "mass": 55.854 * u.amu, "Nion": 11},
    "Ca": {"name": "calcium", "mass": 40.078 * u.amu, "Nion": 0},
}
_prim_data = {
    "H": {"name": "hydrogen", "mass": 1.007947 * u.amu, "Nion": 2, "massFrac": 0.76},
    "He": {"name": "helium", "mass": 4.002602 * u.amu, "Nion": 3, "massFrac": 0.24},
}
_molec_data = {
    "H2": {"name": "molecular hydrogen", "mass": 2.01588 * u.amu},
    "CO": {"name": "carbon monoxide", "mass": (15.9994 + 12.0107) * u.amu},
}

def get_metal_data():
    return _metal_data
def get_prim_data():
    return _prim_data
def get_molec_data():
    return _molec_data
