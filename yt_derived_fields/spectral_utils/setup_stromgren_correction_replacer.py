import numpy as np

from yt_derived_fields.spectral_utils.setup_stromgren_correction_interpolators import (
    format_cloudy_interpolator,
)


def get_all_dust_depletions(metal):

    elements = [
        "Iron","Oxygen","Nitrogen","Carbon","Magnesium","Neon","Silicon","Sulphur","Calcium"
    ]
    dep_solar = [
        0.01, 0.728, 0.603, 0.501, 0.163, 1.0, 0.1, 1.0, 0.003, 1.0
    ]

    # Broken powerlaw model from RR14 (consistent with Taysun's Lya feedback)
    # See Table 1 of: https://www.aanda.org/articles/aa/pdf/2014/03/aa22803-13.pdf
    # We use the XCO,Z case
    a  = 2.21
    aH = 1.00
    b  = 0.96
    aL = 3.10
    xt = 8.10
    xs = 8.69

    x = xs + metal
    x = min(x,xs+np.log10(3.0)) # Limit metallicity to 3x solar for the dust computation

    y = a + (aH * (xs - x))
    if (x <= xt):
        y = b + (aL * (xs - x))

    y = 10.0**y # This is the Gas to Dust mass ratio

    depletion_dict = {
            elements[i]: dep_solar[i] for i in range(len(elements))
            }

    for el in depletion_dict.keys():
        d = depletion_dict[el]
        depletion = 1.0 - ( (1.0 - d) * min(1.0,162.0/y) )
        depletion_dict[el] = depletion

    return depletion_dict


def get_cloudy_emission_line_luminosities(df, mif_cloudy, line_list):
    """
    Gets the emission line luminosities for a given cloudy interpolator
    """
    # First get the array we need to interpolate accounting for the bounds
    to_interpolate = format_cloudy_interpolator(df)

    # Now get the first estimate of the emission line luminosities
    all_emission_lines = 10.**mif_cloudy(to_interpolate)

    # Check if there are any nans in the array
    has_nans = np.isnan(all_emission_lines).any()
    #if has_nans:
    #    print("!!!!WARNING!!!! THIS HALO HAS NANS")

    # Rescale in case we are above or below the gas metallicity bounds
    tmp = np.array(df["[O/H]"] + df["O_dep"] - np.array(np.log10(df["metallicity"]/0.014)))
    idx_oxygen = [i for i,j in enumerate(line_list) if j[0]=="O"]
    rescale = np.ones(len(df))
    rescale[tmp < -3.0] = (10.**tmp[tmp < -3.0]) / (10.**-3.0)
    rescale[tmp > 4.0] = (10.**tmp[tmp > 4.0]) / (10.**4.0)
    all_emission_lines[:,idx_oxygen] *= rescale[:, np.newaxis]

    # Rescale the carbon lines in case of bounds errors
    idx_carbon = [i for i,j in enumerate(line_list) if j[0]=="C"]
    C_over_H_cloudy = np.log10(np.array([get_all_dust_depletions(kk)["Carbon"] for kk in df["[O/H]"]+df["O_dep"]])) # Dust depletion (Carbon)
    tmp1 = np.array(df["[C/H]"] - df["[O/H]"])
    tmp1[tmp1 < -3.0] = -3.0
    tmp1[tmp1 > 1.0] = 1.0
    C_over_H_cloudy += np.log10(2.69E-04) + df["[O/H]"] + df["O_dep"] + tmp1 
    tmp = np.array(df["[C/H]"] + df["C_dep"] + np.log10(2.69E-04))
    rescale = np.array(tmp / C_over_H_cloudy)
    all_emission_lines[:,idx_carbon] *= rescale[:, np.newaxis]

    # Rescale nitrogen by deviation from solar
    idx_nitrogen = [i for i,j in enumerate(line_list) if j[0]=="N"]
    N_over_H_cloudy = np.log10(np.array([get_all_dust_depletions(kk)["Nitrogen"] for kk in df["[O/H]"]+df["O_dep"]])) # Dust depletion (Carbon)
    N_over_H_cloudy += np.log10(6.76E-05) + df["[O/H]"] + df["O_dep"]
    tmp = np.array(df["[N/H]"] + df["N_dep"] + np.log10(6.76E-05))
    rescale = np.array(10.**(tmp - N_over_H_cloudy))
    all_emission_lines[:,idx_nitrogen] *= rescale[:, np.newaxis]

    # Rescale neon by deviation from solar
    idx_neon = [i for i,j in enumerate(line_list) if j[:2]=="Ne"]
    Ne_over_H_cloudy = np.log10(np.array([get_all_dust_depletions(kk)["Neon"] for kk in df["[O/H]"]+df["O_dep"]])) # Dust depletion (Carbon)
    Ne_over_H_cloudy += np.log10(8.51E-05) + df["[O/H]"] + df["O_dep"]
    tmp = np.array(df["[Ne/H]"] + df["Ne_dep"] + np.log10(8.51E-05))
    rescale = np.array(10.**(tmp - Ne_over_H_cloudy))
    all_emission_lines[:,idx_neon] *= rescale[:, np.newaxis]

    # Rescale sulfur by deviation from solar
    idx_sulfur = [i for i,j in enumerate(line_list) if j[:2]=="S "]
    S_over_H_cloudy = np.log10(np.array([get_all_dust_depletions(kk)["Sulphur"] for kk in df["[O/H]"]+df["O_dep"]])) # Dust depletion (Carbon)
    S_over_H_cloudy += np.log10(1.32E-05) + df["[O/H]"] + df["O_dep"]
    tmp = np.array(df["[S/H]"] + df["S_dep"] + np.log10(1.32E-05))
    rescale = np.array(10.**(tmp - S_over_H_cloudy))
    all_emission_lines[:,idx_sulfur] *= rescale[:, np.newaxis]

    # Rescale by Q if out of bounds
    # Note that this isn't exactly correct but I feel uncomfortable with too high Q
    rescale = np.ones(len(df))
    tmp = df["ionizing_luminosity"] 
    rescale[tmp < 46.5] = 10.**(tmp[tmp < 46.5] - 46.5)
    rescale[tmp > 54.5] = 10.**(tmp[tmp > 54.5] - 54.5)
    all_emission_lines *= rescale[:, np.newaxis]

    return np.log10(all_emission_lines)





def replace_emission(df_emis, df_strom, replace_lines, line_list):
    """
    Function to replace the naive emission with the stromgren corrected one
       'C3-1908', 'C3-1906', 'C4-1548', 'C4-1550', 'O1-6300', 'O1-6362',
       'O2-3728', 'O2-3726', 'O2-7320', 'O2-7331', 'O2-7319', 'O2-7330',
       'O3-4959', 'O3-5007', 'O3-4363', 'O3-1661', 'O3-1666', 'Ne3-3869',
       'Ne3-3967', 'N2-6583', 'N2-6548', 'N2-5755', 'N3-1749', 'N3-1754',
       'N3-1747', 'N3-1752', 'N3-1750', 'N4-1486', 'N4-1483', 'N5-1243',
       'N5-1239', 'S2-6731', 'S2-6716', 'S2-4069', 'S2-4076', 'Lya', 'Ha',
       'Hb', 'Hg', 'Hd', 'He2-1640', 'He2-4686'
    """

    line_list_map_dict = {
            'H  1 1215.67A': "Lya",
            'H  1 6562.80A': "Ha",
            'H  1 4861.32A': "Hb",
            'H  1 4340.46A': "Hg",
            'H  1 4101.73A': "Hd",
            'He 2 1640.41A': "He2-1640",
            'He 2 4685.68A': "He2-4686",
            'O  1 63.1679m': None,
            'O  1 145.495m': None,
            'O  1 6300.30A': "O1-6300",
            'O  1 6363.78A': "O1-6362",
            'O  2 3726.03A': "O2-3726",
            'O  2 3728.81A': "O2-3728",
            'O 2R 3726.00A': "O2-3726",
            'O 2R 3729.00A': "O2-3728",
            'O  2 7318.92A': "O2-7319",
            'O  2 7319.99A': "O2-7320",
            'O  2 7329.67A': "O2-7330",
            'O  2 7330.73A': "O2-7331",
            'O 2R 7332.00A': "O2-7331",
            'O 2R 7323.00A': "O2-7320",
            'O  3 1660.81A': "O3-1661",
            'O  3 1666.15A': "O3-1666",
            'O  3 4363.21A': "O3-4363",
            'O 3R 4363.00A': "O3-4363",
            'O 3C 4363.00A': "O3-4363",
            'O  3 4958.91A': "O3-4959",
            'O  3 5006.84A': "O3-5007",
            'O  3 51.8004m': None,
            'O  3 88.3323m': None,
            'O  4 25.8863m': None,
            'Ne 3 3868.76A': "Ne3-3869",
            'Ne 3 3967.47A': "Ne3-3967",
            'C  2 157.636m': None,
            'C  3 1906.68A': "C3-1906",
            'C  3 1908.73A': "C3-1908",
            'C  4 1548.19A': "C4-1548",
            'C  4 1550.77A': "C4-1550",
            'N  2 5754.59A': "N2-5755",
            'N 2R 5755.00A': "N2-5755",
            'N  2 6548.05A': "N2-6548",
            'N  2 6583.45A': "N2-6583",
            'N 2R 6584.00A': "N2-6583",
            'N  2 205.283m': None,
            'N  2 121.769m': None,
            'N  3 57.3238m': None,
            'N  3 1748.65A': "N3-1749",
            'N  3 1753.99A': "N3-1754",
            'N  3 1746.82A': "N3-1747",
            'N  3 1752.16A': "N3-1752",
            'N  3 1749.67A': "N3-1750",
            'N  4 1483.32A': "N4-1483",
            'N  4 1486.50A': "N4-1486",
            'N  5 1238.82A': "N5-1239",
            'N  5 1242.80A': "N5-1243",
            'S  2 6716.44A': "S2-6716",
            'S  2 6730.82A': "S2-6731",
            'S  2 4076.35A': "S2-4076",
            'S  2 4068.60A': "S2-4069",
            'S  3 6312.06A': None,
            'S  3 9068.62A': None,
            'S  3 9530.62A': None,
            }

    label_trigger_dict = {
            key: False for key in df_emis.keys()
            }

    # Get the unique indexes of the cells that need to be replaced
    to_replace = df_strom["cell_idxs"].unique()
    to_replace_dict_map = {
            to_replace[i]: i for i in range(len(to_replace))
            }

    # Group the emission by cell
    #print("Grouping emission")
    updated_values = np.zeros((len(to_replace),replace_lines.shape[1]))

    counter = 0
    #for _,row in tqdm(df_strom.iterrows()):
    for _,row in df_strom.iterrows():
        updated_values[to_replace_dict_map[row["cell_idxs"]],:] += 10.**replace_lines[counter,:]
        counter += 1

    # Loop over the line list
    for key in line_list_map_dict.keys():
         
        # get the index of the line in the replacement array
        idx_in_replace_array = line_list.index(key)

        # Check if there is a mapping otherwise continue
        if line_list_map_dict[key] is None:
            continue

        # If there is a mapping we need to replace the luminosity in the emission array
        # with the new value

        # Start by setting the luminosity of that line to 0 if we haven't already
        if not label_trigger_dict[line_list_map_dict[key]]:
            df_emis.loc[to_replace,[line_list_map_dict[key]]] = 0.0
            # Update the trigger so we don't do this again...
            label_trigger_dict[line_list_map_dict[key]] = True

        # Now add the emission
        df_emis.loc[to_replace,[line_list_map_dict[key]]] += np.array([updated_values[:,idx_in_replace_array]]).T

    return df_emis


