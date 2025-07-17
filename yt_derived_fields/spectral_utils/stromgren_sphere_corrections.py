import numpy as np
from scipy.spatial import cKDTree


age_bins, metal_bins, mif = emission.load_SED_data(top_dir="/mnt/glacier/DATA/SEDtables", ngroups=8, SED_isEgy=True)
data = ytdatacontainer

verbose = True
ionizing_group=4,

""" Physical Constants """
clight  = 2.99792458e10          #[cm/s] light speed
pc_to_cm = 3.086e+18             #parsecs to cm



stellar_metal = 2.09 * data["young_pop2_stars", "particle_metallicity_002"] + 1.06 * data["young_pop2_stars", "particle_metallicity_001"]
star_filt = "young_pop2_stars"

N_youngpop2_stars = np.sum(data[star_filt, "particle_ones"])

if N_youngpop2_stars == 0:
    raise ValueError("LOL")

pp = np.zeros((N_youngpop2_stars, 2))
loc_ages = data["young_pop2_stars", "age"].to("Myr")
loc_ages[loc_ages <1.e-3] = 1e-3
pp[:,0] = np.log10(loc_ages) # Note that this prevents nans
pp[:,1] = np.log10(stellar_metal + 1.e-40)  # eqn in rt_spectra 

# Enforce bounds
pp[:,0][pp[:,0] < np.log10(age_bins)[0]]  = np.log10(age_bins)[0]
pp[:,0][pp[:,0] > np.log10(age_bins)[-1]] = np.log10(age_bins)[-1]
pp[:,1][pp[:,1] < metal_bins[0]]  = metal_bins[0]
pp[:,1][pp[:,1] > metal_bins[-1]] = metal_bins[-1]
star_ion_lums = (10.0**mif(pp)) * np.array(data["young_pop2_stars", "particle_initial_mass"])[:, np.newaxis]

# Make the tree for the gas cells
if verbose: print(f"Making kd tree for {len(df_gas)} cells")
point_tree = cKDTree(data["gas", "particle_position"].to("Mpccm/h"))

# Find the closest cells (only young star particles)
spos = np.zeros((star_filt.sum(),3))
spos[:,0] = data["young_pop2_stars", "particle_position_x"].to("Mpccm/h")
spos[:,1] = data["young_pop2_stars", "particle_position_y"].to("Mpccm/h")
spos[:,2] = data["young_pop2_stars", "particle_position_z"].to("Mpccm/h")

if verbose: print(f"Querying closest cells for {len(spos)} stars")
cell_dist, cell_ids = point_tree.query(spos)
if verbose: print(f"Finished querying position")

# Join lums that are in the same cell
cell_cell_ids = np.unique(cell_ids)
if verbose: print(f"There are {len(cell_cell_ids)} unique cell ids")
cell_lums = np.zeros((len(cell_cell_ids),star_ion_lums.shape[1]))
cell_n_stars = np.zeros(len(cell_cell_ids))
cell_star_mean_ages = np.zeros(len(cell_cell_ids))

loc_ages = np.array(data["young_pop2_stars", "age"])
if verbose: print(f"Looping over all {len(cell_ids)} cell_ids")



for i in range(len(cell_ids)):
    cell_lums[cell_cell_ids == cell_ids[i],:] += star_ion_lums[i,:]
    cell_n_stars[cell_cell_ids == cell_ids[i]] += 1
    cell_star_mean_ages[cell_cell_ids == cell_ids[i]] += loc_ages[i]

cell_star_mean_ages /= cell_n_stars

if verbose: print("Done joining luminosities to cells")

# Recombination rate
HII_temp = 1e4
T = HII_temp
lam_HI = 315614.0/T
alphab = 1.269e-13 * (lam_HI**1.503) / (1. + (lam_HI/0.522)**0.47)**1.923 #cm^3 s^-1

rhos = data["gas", "hydrogen_number_density"].to("cm**-3") #10.0**df_gas["nH"] # h/cc
HI   = data["gas", "hydrogen_01"]
rhos *= HI

# Stromgren radius
r_strom = (3.0 * cell_lums[:,ionizing_group:].sum(axis=1)) / (4.0 * np.pi * rhos[cell_cell_ids] * rhos[cell_cell_ids] * alphab)
r_strom = (r_strom**(1./3.)) / pc_to_cm
if verbose: print("Done calculating stromgren radii")

dx_pc = data["gas", "dx"].to("pc")[cell_cell_ids]
fff = r_strom < (dx_pc/2.)

if fff.sum() == 0:
    raise ValueError("LOL")

unresolved_cell_ids = cell_cell_ids[fff]
if verbose: print(f"There are {len(unresolved_cell_ids)} cells with unresolved stromgren spheres")

# Now that we have the ids, go back and get a list of star particles
# This may need to be optimized...
bool_list = np.array([0 if i not in unresolved_cell_ids else 1 for i in cell_ids]).astype(bool)

# Finally, make a pandas array with all of the properties of the star particles 
# Where the stromgren sphere is unresolved

# Start with the stellar properties
star_header = ["initial_mass","age","metallicity","ionizing_luminosity"]
df_strom = pd.DataFrame(np.zeros((bool_list.sum(),len(star_header))),columns=star_header)
df_strom["initial_mass"] = np.array(data["young_pop2_stars", "particle_initial_mass"])[bool_list]
df_strom["age"] = np.array(data["young_pop2_stars", "age"])[bool_list]
df_strom["metallicity"] = np.array(stellar_metal)[bool_list]
df_strom["ionizing_luminosity"] = np.log10(star_ion_lums[bool_list][:,ionizing_group:].sum(axis=1))
df_strom["x"] = np.array(data["young_pop2_stars", "particle_position_x"])[bool_list]
df_strom["y"] = np.array(data["young_pop2_stars", "particle_position_y"])[bool_list]
df_strom["z"] = np.array(data["young_pop2_stars", "particle_position_z"])[bool_list]

# Continue with the gas properties
gas_header  = ["nH","O/H","C/H"]
tmp_gas = df_gas.loc[cell_ids[bool_list]]
df_strom["nH"] = np.array(tmp_gas["nH"])
df_strom["[O/H]"]  = np.array(((tmp_gas["nO"]-tmp_gas["nH"])-np.log10(4.90E-04)))
df_strom["[C/H]"]  = np.array(((tmp_gas["nC"]-tmp_gas["nH"])-np.log10(2.69E-04)))
df_strom["[N/H]"]  = np.array(((tmp_gas["nN"]-tmp_gas["nH"])-np.log10(6.76E-05)))
df_strom["[Mg/H]"] = np.array(((tmp_gas["nMg"]-tmp_gas["nH"])-np.log10(3.98E-05)))
df_strom["[Ne/H]"] = np.array(((tmp_gas["nNe"]-tmp_gas["nH"])-np.log10(8.51E-05)))
df_strom["[Fe/H]"] = np.array(((tmp_gas["nFe"]-tmp_gas["nH"])-np.log10(3.16E-05)))
df_strom["[Si/H]"] = np.array(((tmp_gas["nSi"]-tmp_gas["nH"])-np.log10(3.24E-05)))
df_strom["[S/H]"]  = np.array(((tmp_gas["nS"]-tmp_gas["nH"])-np.log10(1.32E-05)))
df_strom["cell_idxs"] = cell_ids[bool_list]