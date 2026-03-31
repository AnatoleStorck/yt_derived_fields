import numpy as np
import pandas as pd
from scipy import spatial
from collections import defaultdict

pc_to_cm = 3.086e18


def get_unresolved_stromgren_stars(ds, age_bins, metal_bins, mif, ionizing_group=4, verbose=False):
    """
    Finds all of the cells that need a stromgren sphere correction and returns this
    in the data frame.

    Ionizing group is the index of the ionizing bands (starting at 0).

    This is different from the function below in that it returns the properties of the 
    stars rather than the cell properties. This is useful if we apply the stromgren
    correction to individual star particles rather than what we did for sphinx.
    The main difference is that this assumes that star particles have smaller separate
    stromgren spheres in the same cell rather than one giant stromgren sphere with all
    of the cells.
    """

    ds = None

    # stars
    Nstars =  ds.object("young_pop2_stars", "ones").sum()

    if Nstars < 1:
        raise ValueError("No young stars found in the dataset")

    star_pos = ds.object("young_pop2_stars", "position").in_units("Mpccm/h").d
    star_ages = ds.object("young_pop2_stars", "age").in_units("Myr").d
    star_masses = ds.object("young_pop2_stars", "initial_mass").in_units("Msun").d
    star_metal = (2.09 * ds.object("young_pop2_stars", "met_O") +
                  1.06 * ds.object("young_pop2_stars", "met_Fe")).d
    
    # gas
    Ncells = ds.object("gas", "ones").sum()
    cell_pos = ds.object("gas", "position").in_units("Mpccm/h").d
    cell_dx = ds.object("gas", "dx").in_units("pc").d

    pp = np.zeros((Nstars, 2))
    loc_ages = star_ages
    loc_ages[loc_ages < 1.e-3] = 1e-3           # floor the ages
    pp[:,0] = np.log10(loc_ages)                # Note that this prevents nans
    pp[:,1] = np.log10(star_metal + 1.e-40)     # eqn in rt_spectra 

    # Enforce bounds
    pp[:,0][pp[:,0] < np.log10(age_bins)[0]]  = np.log10(age_bins)[0]
    pp[:,0][pp[:,0] > np.log10(age_bins)[-1]] = np.log10(age_bins)[-1]
    pp[:,1][pp[:,1] < metal_bins[0]]  = metal_bins[0]
    pp[:,1][pp[:,1] > metal_bins[-1]] = metal_bins[-1]

    star_ion_lums = 10.0**mif(pp) * star_masses[:, np.newaxis]

    # Make the tree for the gas cells
    if verbose: print(f"Making kd tree for {Ncells} cells")
    point_tree = spatial.cKDTree(cell_pos)

    # Find the closest cells (only young star particles)
    spos = np.zeros((Nstars, 3))
    spos[:, 0] = star_pos[:, 0]
    spos[:, 1] = star_pos[:, 1]
    spos[:, 2] = star_pos[:, 2]
    if verbose: print(f"Querying closest cells for {len(spos)} stars")
    cell_dist, cell_ids = point_tree.query(spos)
    if verbose: print(f"Finished querying position")

    # Join lums that are in the same cell
    cell_cell_ids = np.unique(cell_ids)
    if verbose: print(f"There are {len(cell_cell_ids)} unique cell ids")
    cell_lums = np.zeros((len(cell_cell_ids),star_ion_lums.shape[1]))
    cell_n_stars = np.zeros(len(cell_cell_ids))
    cell_star_mean_ages = np.zeros(len(cell_cell_ids))

    if verbose: print(f"Looping over all {len(cell_ids)} cell_ids")


    for i in range(len(cell_ids)):
        cell_lums[cell_cell_ids == cell_ids[i],:] += star_ion_lums[i,:]
        cell_n_stars[cell_cell_ids == cell_ids[i]] += 1
        cell_star_mean_ages[cell_cell_ids == cell_ids[i]] += star_ages[i]

    cell_star_mean_ages /= cell_n_stars

    if verbose: print("Done joining luminosities to cells")

    # Recombination rate
    HII_temp = 1e4
    T = HII_temp
    lam_HI = 315614.0/T
    alphab = 1.269e-13 * (lam_HI**1.503) / (1. + (lam_HI/0.522)**0.47)**1.923 #cm^3 s^-1

    nH = ds.object("gas", "hydrogen_number_density").in_units("cm**-3").d
    xHI = ds.object("gas", "hydrogen_01").d

    rhos = nH * xHI
    
    # Stromgren radius
    r_strom = (3.0 * cell_lums[:,ionizing_group:].sum(axis=1)) / (4.0 * np.pi * rhos[cell_cell_ids] * rhos[cell_cell_ids] * alphab)
    r_strom = (r_strom**(1./3.)) / pc_to_cm
    if verbose: print("Done calculating stromgren radii")

    dx_pc = cell_dx[cell_cell_ids]
    fff = r_strom < (dx_pc/2.)

    if fff.sum() == 0:
        return []

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
    df_strom["initial_mass"] = star_masses[bool_list]
    df_strom["age"] = star_ages[bool_list]
    df_strom["metallicity"] = star_metal[bool_list]
    df_strom["ionizing_luminosity"] = np.log10(star_ion_lums[bool_list][:,ionizing_group:].sum(axis=1))
    df_strom["x"] = star_pos[:, 0][bool_list]
    df_strom["y"] = star_pos[:, 1][bool_list]
    df_strom["z"] = star_pos[:, 2][bool_list]

    # Continue with the gas properties
    gas_header  = ["nH","O/H","C/H"]
    tmp_gas = df_gas.loc[cell_ids[bool_list]]
    df_strom["nH"] = np.array(tmp_gas["nH"])
    # HK note: depletions are needed here
    df_strom["[O/H]"]  = np.array(((tmp_gas["nO"]-tmp_gas["nH"])-np.log10(4.90E-04)))
    df_strom["[C/H]"]  = np.array(((tmp_gas["nC"]-tmp_gas["nH"])-np.log10(2.69E-04)))
    df_strom["[N/H]"]  = np.array(((tmp_gas["nN"]-tmp_gas["nH"])-np.log10(6.76E-05)))
    df_strom["[Mg/H]"] = np.array(((tmp_gas["nMg"]-tmp_gas["nH"])-np.log10(3.98E-05)))
    df_strom["[Ne/H]"] = np.array(((tmp_gas["nNe"]-tmp_gas["nH"])-np.log10(8.51E-05)))
    df_strom["[Fe/H]"] = np.array(((tmp_gas["nFe"]-tmp_gas["nH"])-np.log10(3.16E-05)))
    df_strom["[Si/H]"] = np.array(((tmp_gas["nSi"]-tmp_gas["nH"])-np.log10(3.24E-05)))
    df_strom["[S/H]"]  = np.array(((tmp_gas["nS"]-tmp_gas["nH"])-np.log10(1.32E-05)))
    # depletions
    df_strom["O_dep"]  = np.array(np.log10(tmp_gas["O_dep"]))
    df_strom["C_dep"]  = np.array(np.log10(tmp_gas["C_dep"]))
    df_strom["N_dep"]  = np.array(np.log10(tmp_gas["N_dep"]))
    df_strom["Mg_dep"] = np.array(np.log10(tmp_gas["Mg_dep"]))
    df_strom["Ne_dep"] = np.array(np.log10(tmp_gas["Ne_dep"]))
    df_strom["Fe_dep"] = np.array(np.log10(tmp_gas["Fe_dep"]))
    df_strom["Si_dep"] = np.array(np.log10(tmp_gas["Si_dep"]))
    df_strom["S_dep"]  = np.array(np.log10(tmp_gas["S_dep"]))
    df_strom["cell_idxs"] = cell_ids[bool_list]

    return df_strom




def stromgren_radius(Q, nH, T=1e4):
    """
    Returns the stromgren radius in pc
    """
    lam_HI = 315614.0/T
    alphab = 1.269e-13 * (lam_HI**1.503) / (1. + (lam_HI/0.522)**0.47)**1.923 #cm^3 s^-1

    rhos = nH

    # Stromgren radius
    r_strom = (3.0 * Q) / (4.0 * np.pi * rhos * rhos * alphab)
    r_strom = (r_strom**(1./3.)) / pc_to_cm

    return r_strom

def find_connected_components(graph):
    """
    Traverses the graph and finds all of the connections
    """
    visited = set()
    components = []

    def dfs(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, component)

    for node in graph:
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(sorted(component))

    return components

def group_unresolved_strom_stars(df_strom_stars,redshift,boxsize=74.32734130387914,rad_multiplier=1.0):
    """
    Finds the stars that have overlapping stromgren spheres and groups them into one super particle
    --> this will increase the ionization parameter
    """

    # First get a list of unique cells IDs
    unique_cells = df_strom_stars["cell_idxs"].unique()

    # Create an empty DataFrame with the same headers
    df_new = pd.DataFrame(columns=df_strom_stars.columns)

    # Loop over the cells
    #for uc in tqdm(unique_cells):
    for uc in unique_cells:
        filt = df_strom_stars["cell_idxs"] == uc

        # If there is only one star in the cell, simply append it to the new array
        if filt.sum() == 1:
            df_new = pd.concat([df_new, df_strom_stars[filt]], ignore_index=True)

        # Otherwise we need to see if we can group the stars
        else:
            # Get the ionizing luminosity
            Q = 10.**df_strom_stars[filt]["ionizing_luminosity"]

            # Get the cell density --> will be the same for all cells
            nH = 10.**df_strom_stars[filt]["nH"]

            # Get the stromgren radius in pc
            r_strom = stromgren_radius(Q,nH)

            # Convert stromgren radius to box units
            r_strom /= (boxsize * 1000. * 1000. / (1. + redshift))
            r_strom = np.array(r_strom)

            # Particle positions
            part_pos = np.array(df_strom_stars[filt][["x","y","z"]])

            # Check to see which stars overlap
            overlaps = []
            """
            for i in range(filt.sum()-1):
                for j in range(i+1,filt.sum()):
                    # Sum of radii
                    rad_sum = r_strom[i]+r_strom[j]

                    # Particle distances
                    part_dist = np.sqrt(((part_pos[i] - part_pos[j])**2).sum())

                    # Do the radii overlap within --> rad multiplier is a free param
                    # that groups further away particles
                    if part_dist < rad_multiplier*rad_sum:
                        overlaps.append([i,j])
            """
            # Optimization --> Harley's code is much faster than ChatGPT
            # Even after i asked chat GPT for optimization
            for i in range(filt.sum()-1):
                j = i+1
                rad_sum = r_strom[i] + r_strom[j:]
                
                part_dist = np.sqrt(((part_pos[i] - part_pos[j:])**2).sum(axis=1))

                filt_overlap = part_dist < rad_multiplier*rad_sum

                overlaps_loc = [[i,j+k] for k in np.where(filt_overlap)[0]]

                overlaps += overlaps_loc

            # Build an adjacency list
            graph = defaultdict(set)
            for a, b in overlaps:
                graph[a].add(b)
                graph[b].add(a)

            # Find connected components
            connected_stars = find_connected_components(graph)

            # Get a flattend list of all stars in groups
            flattened = [item for sublist in connected_stars for item in sublist]

            # Get a list of isolated stromgren spheres in the cells
            isolated_stars = []
            for i in range(filt.sum()):
                if i not in flattened:
                    isolated_stars.append(i)

            # Now add the isolated stars to the new dataframe
            if len(isolated_stars) > 0:
                for i in isolated_stars:
                    new_entry = df_strom_stars[filt].iloc[i].to_dict()
                    df_new = pd.concat([df_new, pd.DataFrame([new_entry])], ignore_index=True)

            # Now we loop over the connected stars and make a super particles
            for groups in connected_stars:
                my_group = df_strom_stars[filt].iloc[groups]

                ion_lum = 10.**my_group["ionizing_luminosity"]

                # Note that some gas cell properties are the same so
                # We grab the one in index 0
                new_entry = {
                        "ionizing_luminosity": np.log10(ion_lum.sum()),
                        "initial_mass": my_group["initial_mass"].sum(),
                        "nH": my_group["nH"].iloc[0],
                        "cell_idxs": my_group["cell_idxs"].iloc[0],
                        "[O/H]": my_group["[O/H]"].iloc[0],
                        "[C/H]": my_group["[C/H]"].iloc[0],
                        "[N/H]": my_group["[N/H]"].iloc[0],
                        "[Mg/H]": my_group["[Mg/H]"].iloc[0],
                        "[Ne/H]": my_group["[Ne/H]"].iloc[0],
                        "[Fe/H]": my_group["[Fe/H]"].iloc[0],
                        "[Si/H]": my_group["[Si/H]"].iloc[0],
                        "[S/H]": my_group["[S/H]"].iloc[0],
                        "O_dep": my_group["O_dep"].iloc[0],
                        "C_dep": my_group["C_dep"].iloc[0],
                        "N_dep": my_group["N_dep"].iloc[0],
                        "Mg_dep": my_group["Mg_dep"].iloc[0],
                        "Ne_dep": my_group["Ne_dep"].iloc[0],
                        "Fe_dep": my_group["Fe_dep"].iloc[0],
                        "Si_dep": my_group["Si_dep"].iloc[0],
                        "S_dep": my_group["S_dep"].iloc[0],
                }

                # Average the relevant quantities
                for key in my_group.keys():
                    # Linear
                    if key in ["age", "metallicity", "x", "y", "z", "imf_slope"]:
                        new_entry[key] = np.average(my_group[key],weights=ion_lum)

                # Now append the super particle
                df_new = pd.concat([df_new, pd.DataFrame([new_entry])], ignore_index=True)

    for key in df_new.keys():
        if key not in ["cell_idxs"]:
            df_new[key] = df_new[key].astype(float)

    df_new["cell_idxs"] = df_new["cell_idxs"].astype(int)
    return df_new