# Generating derived fields for the chemistry module
# NOTE: for the MEGATRON simulations ran with RAMSES-RTZ

# TODO: Fix iron_fraction ("ramses", "Metallicity") in YT TOML file for MEGATRON

# Author: Anatole Storck

from yt import units as u
import numpy as np

import chemistry_data as chem_data

metal_data = chem_data.get_metal_data()
prim_data = chem_data.get_prim_data()
molec_data = chem_data.get_molec_data()

def _initialize_metal_density(ds, element: str):
    metal_name = metal_data[element]["name"]

    def _metal_density(field, data):
        if element == "Fe":  # NOTE: iron_fraction is "Metallicity"
            metal_density = data["gas", "density"] * data["gas", "iron_fraction"]
        else:
            metal_density = data["gas", "density"] * data["gas", f"{metal_name}_fraction"]

        return metal_density

    ds.add_field(
        name=("gas", f"{metal_name}_density"),
        function=_metal_density,
        units="amu/cm**3",
        sampling_type="cell",
        display_name=f"{metal_name.capitalize()} density",
    )

    def _metal_number_density(field, data):
        return data["gas", f"{metal_name}_density"] / metal_data[element]["mass"]

    ds.add_field(
        name=("gas", f"{metal_name}_number_density"),
        function=_metal_number_density,
        units="cm**-3",
        sampling_type="cell",
        display_name=f"{metal_name.capitalize()} number density",
    )


def _initialize_metallicity(ds):
    def _metallicity(field, data):
        rho = data["gas", "density"]
        Z = np.zeros_like(rho)
        for element in metal_data:
            Z += data["gas", f"{metal_data[element]['name']}_density"]
        Z /= rho
        return Z

    ds.add_field(
        name=("gas", "real_metallicity"),
        function=_metallicity,
        units="1",
        sampling_type="cell",
        display_name="Metallicity",
    )
    # Attempt to override old field if exists
    if ("gas", "metallicity") in ds.field_list:
        ds.add_field(
            name=("gas", "metallicity"),
            function=_metallicity,
            units="1",
            sampling_type="cell",
            display_name="Metallicity",
            force_override=True,
        )


def _initialize_primordial_density(ds, element):
    prim_name = prim_data[element]["name"]

    def _primordial_density(field, data):
        return data["gas", "density"] * prim_data[element]["massFrac"] * (1 - data["gas", "real_metallicity"])

    ds.add_field(
        name=("gas", f"{prim_name}_density"),
        function=_primordial_density,
        units="amu/cm**3",
        sampling_type="cell",
        display_name=f"{prim_name.capitalize()} density",
    )

    def _primordial_number_density(field, data):
        return data["gas", f"{prim_name}_density"] / prim_data[element]["mass"]

    ds.add_field(
        name=("gas", f"{prim_name}_number_density"),
        function=_primordial_number_density,
        units="cm**-3",
        sampling_type="cell",
        display_name=f"{prim_name.capitalize()} number density",
    )


def _initialize_H2(ds):
    def _H2_number_density(field, data):
        xHI = data["gas", "hydrogen_01"]  # ["ramses", "hydro_H_01"]
        xHII = data["gas", "hydrogen_02"]  # ["ramses", "hydro_H_02"]

        xH2 = np.clip(1 - xHI - xHII, 0, 1)
        return data["gas", "hydrogen_number_density"] * xH2 / 2

    ds.add_field(
        name=("gas", "H2_number_density"),
        function=_H2_number_density,
        units="cm**-3",
        sampling_type="cell",
        display_name="H2 number density",
    )

    def _H2_density(field, data):
        return data["gas", "H2_number_density"] * molec_data["H2"]["mass"]

    ds.add_field(
        name=("gas", "H2_density"),
        function=_H2_density,
        units="amu/cm**3",
        sampling_type="cell",
        display_name="H2 density",
    )


def _initialize_CO(ds):
    def _CO_density(field, data):
        return data["gas", "density"] * data["gas", "CO_fraction"]  # ["ramses", "hydro_CO_fraction"]

    ds.add_field(
        name=("gas", "CO_density"),
        function=_CO_density,
        units="amu/cm**3",
        sampling_type="cell",
        display_name="CO density",
    )

    def _CO_number_density(field, data):
        return data["gas", "CO_density"] / molec_data["CO"]["mass"]

    ds.add_field(
        name=("gas", "CO_number_density"),
        function=_CO_number_density,
        units="cm**-3",
        sampling_type="cell",
        display_name="CO number density",
    )


def _initialize_electron_number_density(ds):
    def _electron_number_density(field, data):
        # NOTE: Old fields without yt.toml
        # Ionized hydrogen electrons
        xHII = data["gas", "hydrogen_02"]  # ["ramses", "hydro_H_02"]
        nHII = data["gas", "hydrogen_number_density"] * xHII

        # Ionized helium electrons
        xHeII = data["gas", "helium_02"]  # ["ramses", "hydro_He_02"]
        xHeIII = data["gas", "helium_03"]  # ["ramses", "hydro_He_03"]

        nHeII = data["gas", "helium_number_density"] * xHeII
        nHeIII = data["gas", "helium_number_density"] * xHeIII * 2

        # Ionized metals electrons
        nEl_ion = np.zeros_like(nHII)
        for element in metal_data:
            if element == "Ca":  # No out of equilibrium information
                continue
            metal_name = metal_data[element]["name"]
            nEl = data["gas", f"{metal_name}_density"] / metal_data[element]["mass"]
            for ion in range(metal_data[element]["Nion"]):
                xEl_ion = data["gas", f"{metal_name}_{ion + 1:02d}"]  # ["ramses", f"hydro_{metal_name}_{ion+1:02d}"]
                nEl_ion += ion * nEl * xEl_ion

        return nHII + nHeII + nHeIII + nEl_ion

    ds.add_field(
        name=("gas", "electron_number_density"),
        function=_electron_number_density,
        units="cm**-3",
        sampling_type="cell",
        display_name="Electron number density",
    )


def _initialize_mean_molecular_weight(ds):
    def _mean_molecular_weight(field, data):
        # Compute the total density in the cell
        # NOTE: electron don't matter

        rhoH = data["gas", "hydrogen_density"]
        rhoHe = data["gas", "helium_density"]

        rhoMet = np.zeros_like(rhoH)
        for element in metal_data:
            metal_name = metal_data[element]["name"]
            rhoMet += data["gas", f"{metal_name}_density"]

        rhoCO = data["gas", "CO_density"]

        sum_density = (rhoH + rhoHe + rhoMet + rhoCO) / u.amu

        # Compute the number density of each bound structure in the cell
        # NOTE: electrons matter

        xHI = data["gas", "hydrogen_01"]  # ["ramses", "hydro_H_01"]
        xHII = data["gas", "hydrogen_02"]  # ["ramses", "hydro_H_02"]

        # NOTE: Not really the hydrogen number density, but the number of bound objects containing hydrogen
        #       Real nH is the total number density of hydrogen atoms (nH = nHI + nHII + 2*nH2)
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

    ds.add_field(
        name=("gas", "mu"),
        function=_mean_molecular_weight,
        units="1",
        sampling_type="cell",
        display_name="Mean molecular weight",
    )


def create_chemistry_derived_fields(ds, molecules=True, electron_number_density=True, mean_molecular_weight=False):
    """
    Initialize the derived fields for the chemistry module.

    Parameters
    ----------
    ds : yt.Dataset
        The dataset object.
    molecules : bool, optional
        If True, include the molecular hydrogen and carbon monoxide fields. Default is True.
    electron_number_density : bool, optional
        If True, include the electron number density field, needing the metal ion fractions. Default is True.
    mean_molecular_weight : bool, optional
        If True, include the mean molecular weight field, needing the CO, H2, and e number densities. Default is False.
    """

    # Add fields for metal densities
    for element in metal_data:
        _initialize_metal_density(ds, element)

    _initialize_metallicity(ds)

    for element in prim_data:
        _initialize_primordial_density(ds, element)

    if electron_number_density:
        _initialize_electron_number_density(ds)

    if molecules:
        _initialize_H2(ds)
        _initialize_CO(ds)

    if mean_molecular_weight and molecules and electron_number_density:
        _initialize_mean_molecular_weight(ds)
