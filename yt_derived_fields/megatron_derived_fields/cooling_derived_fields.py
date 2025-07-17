# Generating derived fields for the cooling module
# NOTE: for the MEGATRON simulations ran with RAMSES-RTZ

# NOTE: which H2 cooling function depends on the specific run! (ask people idk lol)
#       MEGATRON_OG uses H2GP while MEGATRON_SF uses Moseley (maybe idk lol)

# TODO: Introduce the YT TOML file for MEGATRON, and set the fields to "gas" instead of "ramses"

# Author: Anatole Storck

from yt import units as u
import yt_derived_fields.megatron_derived_fields.chemistry_derived_fields as chemistry_fields

import numpy as np


# Converted from coolrates_module.f90
def cooling_H2_moseley(nH, nH2, Tgas):
    T3 = 1e-3 * Tgas

    n1 = 50.0
    n2 = 450.0
    n3 = 25.0
    n4 = 900.0

    x1 = nH + (5.0 * nH2)
    x2 = nH + (4.5 * nH2)
    x3 = nH + (0.75 * nH2)
    x4 = nH + (0.05 * nH2)

    f1 = 1.1e-25 * np.sqrt(T3) * np.exp(-0.51 / T3)
    f1 = f1 * (
        ((0.7 * x1) / (1.0 + (x1 / n1))) + ((0.3 * x1) / (1.0 + (x1 / (10.0 * n1))))
    )

    f2 = 2.0e-25 * T3 * np.exp(-1.0 / T3)
    f2 = f2 * (
        ((0.35 * x2) / (1.0 + (x2 / n2))) + ((0.65 * x2) / (1.0 + (x2 / (10.0 * n2))))
    )

    f3 = 2.4e-24 * (T3**1.5) * np.exp(-2.0 / T3)
    f3 = f3 * (x3 / (1.0 + (x3 / n3)))

    f4 = 1.7e-23 * (T3**1.5) * np.exp(-4.0 / T3)
    f4 = f4 * (
        ((0.45 * x4) / (1.0 + (x4 / n4))) + ((0.55 * x4) / (1.0 + (x4 / (10.0 * n4))))
    )

    return nH2 * (f1 + f2 + f3 + f4)


# Converted from coolrates_module.f90
def cooling_H2GP(nH, nH2, Tgas):
    # cooling from Galli&Palla98
    # taken from krome

    H2GP = np.zeros_like(Tgas)

    tm = np.clip(Tgas, 13.0, 1e5)  # no cooling below 13 Kelvin and above 1e5 K
    logT = np.log10(tm)
    T3 = tm * 1e-3

    # low density limit in erg/s
    LDL = (
        10
        ** (-103.0 + 97.59 * logT - 48.05 * logT**2 + 10.8 * logT**3 - 0.9032 * logT**4)
    ) * nH
    mask_LDL = LDL == 0

    # high density limit
    HDLR = (9.5e-22 * T3**3.76) / (1.0 + 0.12 * T3**2.1) * np.exp(
        -((0.13 / T3) ** 3)
    ) + 3.0e-24 * np.exp(-0.51 / T3)  # erg/s
    HDLV = 6.7e-19 * np.exp(-5.86 / T3) + 1.6e-18 * np.exp(-11.7 / T3)  # erg/s
    HDL = HDLR + HDLV  # erg/s
    mask_HDL = HDL == 0

    H2_cooling = nH2 / (1 / HDL + 1 / LDL)  # erg/cm^3/s

    mask = (~mask_LDL) | (~mask_HDL)
    H2GP[mask] = H2_cooling[mask]

    return H2GP


# Assuming high temperature metal cooling rate is the remaining cooling
# after subtracting primordial, low temperature, dust, and CO cooling,
# and NOTE: fine_structure (metal cooling below 10^4 K)
def _initialize_metal_cooling(ds):
    def _metal_cooling(field, data):
        return (
            data["ramses", "hydro_cooling_rate"]
            - data["ramses", "hydro_cooling_primordial"]
            - data["ramses", "hydro_cooling_dust_rec"]
            - data["ramses", "hydro_cooling_dust"]
            - data["ramses", "hydro_cooling_CO"]
        )

    ds.add_field(
        name=("gas", "cooling_metal"),
        function=_metal_cooling,
        units="erg/s/cm**3",
        sampling_type="cell",
        display_name="Metal cooling",
    )


# Assuming photoheating is the remaining heating
def _initialize_photoheating_heating(ds):
    def _photoheating_heating(field, data):
        if ("ramses", "hydro_heating_cr") not in ds.derived_field_list:
            cr_heating = 0
        else:
            cr_heating = data["ramses", "hydro_heating_cr"]
        return (
            data["ramses", "hydro_heating_rate"]
            - data["ramses", "hydro_heating_h2"]
            - data["ramses", "hydro_heating_pe"]
            - data["ramses", "hydro_heating_ct"]
            - cr_heating
        )

    ds.add_field(
        name=("gas", "photoheating_heating"),
        function=_photoheating_heating,
        units="erg/s/cm**3",
        sampling_type="cell",
        display_name="Photoheating heating",
    )


def _initialize_H2_cooling(ds, H2_cooling):
    def _H2_cooling(field, data):
        nH = data["gas", "hydrogen_number_density"].to("cm**-3")
        nH2 = data["gas", "H2_number_density"].to("cm**-3")

        Tgas = data["gas", "temperature"].to("K")

        cooling_H2 = np.zeros(nH.shape)

        # H2 cooling is only valid for Tgas < 1e4 K
        temperature_mask = Tgas < 1e4 * u.K

        cooling_H2[temperature_mask] = H2_cooling(nH.value, nH2.value, Tgas.value)[
            temperature_mask
        ]

        return (
            cooling_H2 * u.erg / (u.s * u.cm**3)
        )  # NOTE: the H2 cooling function returns in erg/s/cm^3

    ds.add_field(
        name=("gas", "cooling_H2"),
        function=_H2_cooling,
        units="erg/s/cm**3",
        sampling_type="cell",
        display_name="H2 cooling",
    )


def create_cooling_derived_fields(ds, H2_cooling="moseley"):
    """
    Generate cooling and heating data from the cooling module.

    Parameters
    ----------
    ds : yt.Dataset
        The dataset object.
    halo_sphere : yt.data_objects.YTSphere
        The YT sphere to analyze.
    H2_cooling : function or str, optional
        Function or string specifying the H2 cooling model to use.
        Can be 'moseley', or 'H2GP'. If None, H2 cooling is not added.
    """

    _initialize_metal_cooling(ds)
    _initialize_photoheating_heating(ds)

    if H2_cooling is not None:
        chemistry_fields.create_chemistry_derived_fields(
            ds, electron_number_density=False, mean_molecular_weight=False
        )

        if H2_cooling == "moseley":
            H2_cooling_func = cooling_H2_moseley
        elif H2_cooling == "H2GP":
            H2_cooling_func = cooling_H2GP
        else:
            raise ValueError("Invalid H2 cooling model specified.")
        _initialize_H2_cooling(ds, H2_cooling_func)
