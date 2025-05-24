# This code is used to calculate the emissivities of various lines across (rho, T) using pyneb and chianti.
# NOTE: Requires the CHIANTI database to be installed and the XUVTOP environment variable to be correctly set.

# The database is saved as coll_line_dict.npy and rec_line_dict.npy for collisional and recombination lines respectively.
# the database contains metadata about the lines, including the ion, atom, upper and lower levels, as well
# as the emissivity grid, which is a RegularGridInterpolator object taking in rho [g/cm**3] and T [K].

# Written by Harley Katz and modified by Anatole Storck

import os
# Set the XUVTOP environment variable to the location of the CHIANTI database
os.environ["XUVTOP"] = "/mnt/glacier/chianti/"

import numpy as np
import pyneb as pn
import ChiantiPy.core as ch
from scipy.interpolate import RegularGridInterpolator, interp1d

""" Physical Constants """
avogad  = 6.0221409e23           #[1/mol] Avogadro's number

# element weights [g]
element_weights = {
    'C': 12.011,
    'Fe': 55.847,
    'H': 1.00797,
    'He': 4.00260,
    'Mg': 24.305,
    'N': 14.0067,
    'Ne': 20.179,
    'O': 15.9994,
    'S': 32.06,
    'Si': 28.0855,
}
for key in element_weights:
    element_weights[key] = element_weights[key]/avogad


# Set everything to chianti for pyneb
for el in ["o","n","s","ne","c","mg"]:
    for i in ["i","ii","iii","iv","v"]:
        try:
            pn.atomicData.setDataFile(f'{el}_{i}_atom.chianti')
            pn.atomicData.setDataFile(f'{el}_{i}_coll.chianti')
        except:
            print(f"Could not set chianti data for {el}-{i}, using default")

# Initialize pyneb atoms
pyneb_atoms = {
    "C2": pn.Atom('C', 2, NLevels=15),
    "C3": pn.Atom('C', 3, NLevels=15),
    "C4": pn.Atom('C', 4, NLevels=15),
    "O1": pn.Atom('O', 1, NLevels=15),
    "O2": pn.Atom('O', 2, NLevels=15),
    "O3": pn.Atom('O', 3, NLevels=15),
    "S2": pn.Atom('S', 2, NLevels=15),
    "S3": pn.Atom('S', 3, NLevels=15),
    "S4": pn.Atom('S', 4, NLevels=15),
    "N2": pn.Atom('N', 2, NLevels=15),
    "N3": pn.Atom('N', 3, NLevels=15),
    "N4": pn.Atom('N', 4, NLevels=15),
    "N5": pn.Atom('N', 5, NLevels=15),
    "Ne3": pn.Atom('Ne', 3, NLevels=15),
    "Ne4": pn.Atom('Ne', 4, NLevels=15),
    "Ne5": pn.Atom('Ne', 5, NLevels=15),
    "Mg2": pn.Atom('Mg', 2, NLevels=15),
    "H1rec": pn.RecAtom('H', 1),
    "He2rec": pn.RecAtom('He', 2)
}

# Make a Te/ne grid
n_vals = 200
temperatures = np.logspace(1,9,n_vals)
e_densities  = np.logspace(-7,6,n_vals)


"""
# Get the collisional emissivities for the metalic transitions
"""

# Metalic transitions database
coll_line_dict = {
        # Carbon
        "C3-1908": {
            "ion": "C_III",
            "atom": pyneb_atoms["C3"],
            "lev_u": 3,
            "lev_d": 1,
            "e_weight": element_weights["C"],
            },
        "C3-1906": {
            "ion": "C_III",
            "atom": pyneb_atoms["C3"],
            "lev_u": 4,
            "lev_d": 1,
            "e_weight": element_weights["C"],
            },
        "C4-1548": {
            "ion": "C_IV",
            "atom": pyneb_atoms["C4"],
            "lev_u": 3,
            "lev_d": 1,
            "e_weight": element_weights["C"],
            },
        "C4-1550": {
            "ion": "C_IV",
            "atom": pyneb_atoms["C4"],
            "lev_u": 2,
            "lev_d": 1,
            "e_weight": element_weights["C"],
            },
        # Oxygen
        "O1-6300": {
            "ion": "O_I",
            "atom": pyneb_atoms["O1"],
            "lev_u": 4,
            "lev_d": 1,
            "e_weight": element_weights["O"],
            },
        "O1-6362": {
            "ion": "O_I",
            "atom": pyneb_atoms["O1"],
            "lev_u": 4,
            "lev_d": 2,
            "e_weight": element_weights["O"],
            },
        "O2-3728": {
            "ion": "O_II",
            "atom": pyneb_atoms["O2"],
            "lev_u": 2,
            "lev_d": 1,
            "e_weight": element_weights["O"],
            },
        "O2-3726": {
            "ion": "O_II",
            "atom": pyneb_atoms["O2"],
            "lev_u": 3,
            "lev_d": 1,
            "e_weight": element_weights["O"],
            },
        "O2-7320": {
            "ion": "O_II",
            "atom": pyneb_atoms["O2"],
            "lev_u": 4,
            "lev_d": 2,
            "e_weight": element_weights["O"],
            },
        "O2-7331": {
            "ion": "O_II",
            "atom": pyneb_atoms["O2"],
            "lev_u": 4,
            "lev_d": 3,
            "e_weight": element_weights["O"],
            },
        "O2-7319": {
            "ion": "O_II",
            "atom": pyneb_atoms["O2"],
            "lev_u": 5,
            "lev_d": 2,
            "e_weight": element_weights["O"],
            },
        "O2-7330": {
            "ion": "O_II",
            "atom": pyneb_atoms["O2"],
            "lev_u": 5,
            "lev_d": 3,
            "e_weight": element_weights["O"],
            },
        "O3-4959": {
            "ion": "O_III",
            "atom": pyneb_atoms["O3"],
            "lev_u": 4,
            "lev_d": 2,
            "e_weight": element_weights["O"],
            },
        "O3-5007": {
            "ion": "O_III",
            "atom": pyneb_atoms["O3"],
            "lev_u": 4,
            "lev_d": 3,
            "e_weight": element_weights["O"],
            }, 
        "O3-4363": {
            "ion": "O_III",
            "atom": pyneb_atoms["O3"],
            "lev_u": 5,
            "lev_d": 4,
            "e_weight": element_weights["O"],
            },
        "O3-1661": {
            "ion": "O_III",
            "atom": pyneb_atoms["O3"],
            "lev_u": 6,
            "lev_d": 2,
            "e_weight": element_weights["O"],
            },
        "O3-1666": {
            "ion": "O_III",
            "atom": pyneb_atoms["O3"],
            "lev_u": 6,
            "lev_d": 3,
            "e_weight": element_weights["O"],
            },
        # Neon
        "Ne3-3869": {
            "ion": "Ne_III",
            "atom": pyneb_atoms["Ne3"],
            "lev_u": 4,
            "lev_d": 1,
            "e_weight": element_weights["Ne"],
            },
        "Ne3-3967": {
            "ion": "Ne_III",
            "atom": pyneb_atoms["Ne3"],
            "lev_u": 4,
            "lev_d": 2,
            "e_weight": element_weights["Ne"],
            },
        # Nitrogen 2
        "N2-6583": {
            "ion": "N_II",
            "atom": pyneb_atoms["N2"],
            "lev_u": 4,
            "lev_d": 3,
            "e_weight": element_weights["N"],
            },
        "N2-6548": {
            "ion": "N_II",
            "atom": pyneb_atoms["N2"],
            "lev_u": 4,
            "lev_d": 2,
            "e_weight": element_weights["N"],
            },
        "N2-5755": {
            "ion": "N_II",
            "atom": pyneb_atoms["N2"],
            "lev_u": 5,
            "lev_d": 4,
            "e_weight": element_weights["N"],
            },
        "N3-1749": {
            "ion": "N_III",
            "atom": pyneb_atoms["N3"],
            "lev_u": 3,
            "lev_d": 1,
            "e_weight": element_weights["N"],
            },
        "N3-1754": {
            "ion": "N_III",
            "atom": pyneb_atoms["N3"],
            "lev_u": 3,
            "lev_d": 2,
            "e_weight": element_weights["N"],
            },
        "N3-1747": {
            "ion": "N_III",
            "atom": pyneb_atoms["N3"],
            "lev_u": 4,
            "lev_d": 1,
            "e_weight": element_weights["N"],
            },
        "N3-1752": {
            "ion": "N_III",
            "atom": pyneb_atoms["N3"],
            "lev_u": 4,
            "lev_d": 2,
            "e_weight": element_weights["N"],
            },
        "N3-1750": {
            "ion": "N_III",
            "atom": pyneb_atoms["N3"],
            "lev_u": 5,
            "lev_d": 2,
            "e_weight": element_weights["N"],
            },
        "N4-1486": {
            "ion": "N_IV",
            "atom": pyneb_atoms["N4"],
            "lev_u": 3,
            "lev_d": 1,
            "e_weight": element_weights["N"],
            },
        "N4-1483": {
            "ion": "N_IV",
            "atom": pyneb_atoms["N4"],
            "lev_u": 4,
            "lev_d": 1,
            "e_weight": element_weights["N"],
            },
        "N5-1243": {
            "ion": "N_V",
            "atom": pyneb_atoms["N5"],
            "lev_u": 2,
            "lev_d": 1,
            "e_weight": element_weights["N"],
            },
        "N5-1239": {
            "ion": "N_V",
            "atom": pyneb_atoms["N5"],
            "lev_u": 3,
            "lev_d": 1,
            "e_weight": element_weights["N"],
            },
        # Sulfur
        "S2-6731": {
            "ion": "S_II",
            "atom": pyneb_atoms["S2"],
            "lev_u": 2,
            "lev_d": 1,
            "e_weight": element_weights["S"],
            },
        "S2-6716": {
            "ion": "S_II",
            "atom": pyneb_atoms["S2"],
            "lev_u": 3,
            "lev_d": 1,
            "e_weight": element_weights["S"],
            },
        "S2-4069": {
            "ion": "S_II",
            "atom": pyneb_atoms["S2"],
            "lev_u": 5,
            "lev_d": 1,
            "e_weight": element_weights["S"],
            },
        "S2-4076": {
            "ion": "S_II",
            "atom": pyneb_atoms["S2"],
            "lev_u": 4,
            "lev_d": 1,
            "e_weight": element_weights["S"],
            },
        # Magnesium
        #"Mg2-2796": {
        #    "ion": "Mg_II",
        #    "atom": pyneb_atoms["Mg2"],
        #    "lev_u": 3,
        #    "lev_d": 1,
        #    "e_weight": element_weights["Mg"],
        #    },
        #"Mg2-2803": {
        #    "ion": "Mg_II",
        #    "atom": pyneb_atoms["Mg2"],
        #    "lev_u": 2,
        #    "lev_d": 1,
        #    "e_weight": element_weights["Mg"],
        #    },
}

# Initialize the emissivity grids (emission at each temperature and electron density)
for l in coll_line_dict.keys():
    print(f"initializing {l}")
    atom = coll_line_dict[l]["atom"]
    lev_u = coll_line_dict[l]["lev_u"]
    lev_d = coll_line_dict[l]["lev_d"]
    em_grid = atom.getEmissivity(tem=temperatures, den=e_densities, lev_i=lev_u, lev_j=lev_d, product=True)
    interp = RegularGridInterpolator((np.log10(temperatures), np.log10(e_densities)), em_grid, bounds_error=False, fill_value=0.0)
    coll_line_dict[l]["emis_grid"] = interp



"""
# Get the recombination emissivities for the hydrogen and helium lines
"""

# Hydrogen and Helium recombination lines
rec_line_dict = {
    "Lya": {
        "ion": "H_II",
        "atom": pyneb_atoms["H1rec"],
        "lev_u": 2,
        "lev_d": 1,
        "e_weight": 1.0,
    },
    "Ha": {
        "ion": "H_II",
        "atom": pyneb_atoms["H1rec"],
        "lev_u": 3,
        "lev_d": 2,
        "e_weight": 1.0,
    },
    "Hb": {
        "ion": "H_II",
        "atom": pyneb_atoms["H1rec"],
        "lev_u": 4,
        "lev_d": 2,
        "e_weight": 1.0,
    },
    "Hg": {
        "ion": "H_II",
        "atom": pyneb_atoms["H1rec"],
        "lev_u": 5,
        "lev_d": 2,
        "e_weight": 1.0,
    },
    "Hd": {
        "ion": "H_II",
        "atom": pyneb_atoms["H1rec"],
        "lev_u": 6,
        "lev_d": 2,
        "e_weight": 1.0,
    },
    "He2-1640": {
        "ion": "He_III",
        "atom": pyneb_atoms["He2rec"],
        "lev_u": 3,
        "lev_d": 2,
        "e_weight": 4.002602,
    },
    "He2-4686": {
        "ion": "He_III",
        "atom": pyneb_atoms["He2rec"],
        "lev_u": 4,
        "lev_d": 3,
        "e_weight": 4.002602,
    },
}

# Initialize the emissivity grids (emission at each temperature and electron density)
for key in rec_line_dict:
    print(f"initializing {key}")
    lev_u = rec_line_dict[key]["lev_u"]
    lev_d = rec_line_dict[key]["lev_d"]
    em_grid = np.zeros((len(temperatures),len(e_densities)))
    for i,ne in enumerate(e_densities):
        em = rec_line_dict[key]["atom"].getEmissivity(temperatures, ne, lev_i=lev_u, lev_j=lev_d)
        filt_loc = np.isnan(em)
        mf = interp1d(np.log10(temperatures[~filt_loc]),np.log10(em[~filt_loc]),bounds_error=False,fill_value="extrapolate")
        em_grid[:,i] = 10.0**mf(np.log10(temperatures))

    interp = RegularGridInterpolator((np.log10(temperatures), np.log10(e_densities)), em_grid, bounds_error=False, fill_value=0.0)
    rec_line_dict[key]["emis_grid"] = interp
    
"""
# Get the collisional emissivities for the hydrogen and helium lines
"""

# Set the emissivity grids (emission at each temperature)
#New collisional code - uses data directly from chianti and interpolates
#Note, we assume dens = 1, as it was found that the emissivity fluctuates by ~1% across a wide range of densities
temps_chianti = np.logspace(1, 9, 1000)
dens_chianti = 1.0

h1_ion = ch.ion("h_1",temps_chianti,dens_chianti)
h1_ion.emiss()
h1_res_dict = h1_ion.Emiss

he2_ion = ch.ion("he_2",temps_chianti,dens_chianti)
he2_ion.emiss()
he2_res_dict = he2_ion.Emiss

for key in rec_line_dict:
    lev_u = rec_line_dict[key]["lev_u"]
    lev_d = rec_line_dict[key]["lev_d"]

    if rec_line_dict[key]["ion"][:-1] == "H_I":
        rd = h1_res_dict
    elif rec_line_dict[key]["ion"][:-1] == "He_II":
        rd = he2_res_dict
    else:
        print(f"Coll ion {rec_line_dict[key]['ion']} unavailable")
        continue

    col_emissivity = np.zeros(len(temps_chianti))

    for i in range(len(rd["pretty2"])):
        if (str(rd["pretty2"][i][0]) == str(lev_u)) & (str(rd["pretty1"][i][0]) == str(lev_d)):
            col_emissivity += np.array(rd["emiss"][i])*4.0*np.pi #Note: Chianti emissivities are per steradian

    interp = interp1d(temps_chianti,col_emissivity)
    rec_line_dict[key]["emis_grid_col"] = interp


# Save the emissivity dictionaries to a file

np.save("coll_line_dict.npy", coll_line_dict, allow_pickle=True)
np.save("rec_line_dict.npy", rec_line_dict, allow_pickle=True)