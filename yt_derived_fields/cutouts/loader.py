from enum import Enum
from pathlib import Path

import numpy as np
import pooch
import unyt
import yt
from scipy.io import FortranFile
from tqdm import tqdm
from yt_experiments.octree.converter import OctTree

from yt_derived_fields.megatron_derived_fields.chemistry_derived_fields import metal_data


class Scale(Enum):
    LINEAR = 0
    LOG = 1


headers: dict[int, list[tuple[str, Scale, str, str]]] = {
    1: [
        ("redshift", Scale.LINEAR, "1", "d"),
        ("dx", Scale.LOG, "cm", "d"),
        ("x", Scale.LINEAR, "Mpccm/h", "d"),
        ("y", Scale.LINEAR, "Mpccm/h", "d"),
        ("z", Scale.LINEAR, "Mpccm/h", "d"),
        ("vx", Scale.LINEAR, "cm/s", "d"),
        ("vy", Scale.LINEAR, "cm/s", "d"),
        ("vz", Scale.LINEAR, "cm/s", "d"),
        ("density", Scale.LOG, "mp/cm**3", "d"),
        ("temperature", Scale.LOG, "K", "d"),
        ("pressure", Scale.LINEAR, "dyne/cm**2", "d"),
        ("iron_number_density", Scale.LOG, "1/cm**3", "d"),
        ("oxygen_number_density", Scale.LOG, "1/cm**3", "d"),
        ("nitrogen_number_density", Scale.LOG, "1/cm**3", "d"),
        ("magnesium_number_density", Scale.LOG, "1/cm**3", "d"),
        ("neon_number_density", Scale.LOG, "1/cm**3", "d"),
        ("silicon_number_density", Scale.LOG, "1/cm**3", "d"),
        ("calcium_number_density", Scale.LOG, "1/cm**3", "d"),
        ("carbon_number_density", Scale.LOG, "1/cm**3", "d"),
        ("sulfur_number_density", Scale.LOG, "1/cm**3", "d"),
        ("carbon_monoxide_number_density", Scale.LOG, "1/cm**3", "d"),
        ("oxygen_01", Scale.LINEAR, "1", "d"),
        ("oxygen_02", Scale.LINEAR, "1", "d"),
        ("oxygen_03", Scale.LINEAR, "1", "d"),
        ("oxygen_04", Scale.LINEAR, "1", "d"),
        ("oxygen_05", Scale.LINEAR, "1", "d"),
        ("oxygen_06", Scale.LINEAR, "1", "d"),
        ("oxygen_07", Scale.LINEAR, "1", "d"),
        ("oxygen_08", Scale.LINEAR, "1", "d"),
        ("nitrogen_01", Scale.LINEAR, "1", "d"),
        ("nitrogen_02", Scale.LINEAR, "1", "d"),
        ("nitrogen_03", Scale.LINEAR, "1", "d"),
        ("nitrogen_04", Scale.LINEAR, "1", "d"),
        ("nitrogen_05", Scale.LINEAR, "1", "d"),
        ("nitrogen_06", Scale.LINEAR, "1", "d"),
        ("nitrogen_07", Scale.LINEAR, "1", "d"),
        ("carbon_01", Scale.LINEAR, "1", "d"),
        ("carbon_02", Scale.LINEAR, "1", "d"),
        ("carbon_03", Scale.LINEAR, "1", "d"),
        ("carbon_04", Scale.LINEAR, "1", "d"),
        ("carbon_05", Scale.LINEAR, "1", "d"),
        ("carbon_06", Scale.LINEAR, "1", "d"),
        ("magnesium_01", Scale.LINEAR, "1", "d"),
        ("magnesium_02", Scale.LINEAR, "1", "d"),
        ("magnesium_03", Scale.LINEAR, "1", "d"),
        ("magnesium_04", Scale.LINEAR, "1", "d"),
        ("magnesium_05", Scale.LINEAR, "1", "d"),
        ("magnesium_06", Scale.LINEAR, "1", "d"),
        ("magnesium_07", Scale.LINEAR, "1", "d"),
        ("magnesium_08", Scale.LINEAR, "1", "d"),
        ("magnesium_09", Scale.LINEAR, "1", "d"),
        ("magnesium_10", Scale.LINEAR, "1", "d"),
        ("silicon_01", Scale.LINEAR, "1", "d"),
        ("silicon_02", Scale.LINEAR, "1", "d"),
        ("silicon_03", Scale.LINEAR, "1", "d"),
        ("silicon_04", Scale.LINEAR, "1", "d"),
        ("silicon_05", Scale.LINEAR, "1", "d"),
        ("silicon_06", Scale.LINEAR, "1", "d"),
        ("silicon_07", Scale.LINEAR, "1", "d"),
        ("silicon_08", Scale.LINEAR, "1", "d"),
        ("silicon_09", Scale.LINEAR, "1", "d"),
        ("silicon_10", Scale.LINEAR, "1", "d"),
        ("silicon_11", Scale.LINEAR, "1", "d"),
        ("sulfur_01", Scale.LINEAR, "1", "d"),
        ("sulfur_02", Scale.LINEAR, "1", "d"),
        ("sulfur_03", Scale.LINEAR, "1", "d"),
        ("sulfur_04", Scale.LINEAR, "1", "d"),
        ("sulfur_05", Scale.LINEAR, "1", "d"),
        ("sulfur_06", Scale.LINEAR, "1", "d"),
        ("sulfur_07", Scale.LINEAR, "1", "d"),
        ("sulfur_08", Scale.LINEAR, "1", "d"),
        ("sulfur_09", Scale.LINEAR, "1", "d"),
        ("sulfur_10", Scale.LINEAR, "1", "d"),
        ("sulfur_11", Scale.LINEAR, "1", "d"),
        ("iron_01", Scale.LINEAR, "1", "d"),
        ("iron_02", Scale.LINEAR, "1", "d"),
        ("iron_03", Scale.LINEAR, "1", "d"),
        ("iron_04", Scale.LINEAR, "1", "d"),
        ("iron_05", Scale.LINEAR, "1", "d"),
        ("iron_06", Scale.LINEAR, "1", "d"),
        ("iron_07", Scale.LINEAR, "1", "d"),
        ("iron_08", Scale.LINEAR, "1", "d"),
        ("iron_09", Scale.LINEAR, "1", "d"),
        ("iron_10", Scale.LINEAR, "1", "d"),
        ("iron_11", Scale.LINEAR, "1", "d"),
        ("neon_01", Scale.LINEAR, "1", "d"),
        ("neon_02", Scale.LINEAR, "1", "d"),
        ("neon_03", Scale.LINEAR, "1", "d"),
        ("neon_04", Scale.LINEAR, "1", "d"),
        ("neon_05", Scale.LINEAR, "1", "d"),
        ("neon_06", Scale.LINEAR, "1", "d"),
        ("neon_07", Scale.LINEAR, "1", "d"),
        ("neon_08", Scale.LINEAR, "1", "d"),
        ("neon_09", Scale.LINEAR, "1", "d"),
        ("neon_10", Scale.LINEAR, "1", "d"),
        ("hydrogen_01", Scale.LINEAR, "1", "d"),
        ("hydrogen_02", Scale.LINEAR, "1", "d"),
        ("helium_02", Scale.LINEAR, "1", "d"),
        ("helium_03", Scale.LINEAR, "1", "d"),
        ("Habing", Scale.LOG, "erg/s/cm**2", "d"),
        ("Lyman_Werner", Scale.LOG, "erg/s/cm**2", "d"),
        ("HI_Ionising", Scale.LOG, "erg/s/cm**2", "d"),
        ("H2_Ionising", Scale.LOG, "erg/s/cm**2", "d"),
        ("HeI_Ionising", Scale.LOG, "erg/s/cm**2", "d"),
        ("HeII_ionising", Scale.LOG, "erg/s/cm**2", "d"),
    ],
    2: [
        ("redshift", Scale.LINEAR, "1", "f"),
        ("dx", Scale.LOG, "cm", "f"),
        ("x", Scale.LINEAR, "Mpccm/h", "d"),
        ("y", Scale.LINEAR, "Mpccm/h", "d"),
        ("z", Scale.LINEAR, "Mpccm/h", "d"),
        ("vx", Scale.LINEAR, "cm/s", "f"),
        ("vy", Scale.LINEAR, "cm/s", "f"),
        ("vz", Scale.LINEAR, "cm/s", "f"),
        ("density", Scale.LOG, "mp/cm**3", "f"),
        ("hydrogen_density", Scale.LOG, "1/cm**3", "f"),
        ("temperature", Scale.LOG, "K", "f"),
        ("pressure", Scale.LOG, "dyne/cm**2", "f"),
        ("iron_number_density", Scale.LOG, "1/cm**3", "f"),
        ("oxygen_number_density", Scale.LOG, "1/cm**3", "f"),
        ("nitrogen_number_density", Scale.LOG, "1/cm**3", "f"),
        ("magnesium_number_density", Scale.LOG, "1/cm**3", "f"),
        ("neon_number_density", Scale.LOG, "1/cm**3", "f"),
        ("silicon_number_density", Scale.LOG, "1/cm**3", "f"),
        ("calcium_number_density", Scale.LOG, "1/cm**3", "f"),
        ("carbon_number_density", Scale.LOG, "1/cm**3", "f"),
        ("sulfur_number_density", Scale.LOG, "1/cm**3", "f"),
        ("carbon_monoxide_number_density", Scale.LOG, "1/cm**3", "f"),
        ("oxygen_01", Scale.LOG, "1", "f"),
        ("oxygen_02", Scale.LOG, "1", "f"),
        ("oxygen_03", Scale.LOG, "1", "f"),
        ("oxygen_04", Scale.LOG, "1", "f"),
        ("oxygen_05", Scale.LOG, "1", "f"),
        ("oxygen_06", Scale.LOG, "1", "f"),
        ("oxygen_07", Scale.LOG, "1", "f"),
        ("oxygen_08", Scale.LOG, "1", "f"),
        ("nitrogen_01", Scale.LOG, "1", "f"),
        ("nitrogen_02", Scale.LOG, "1", "f"),
        ("nitrogen_03", Scale.LOG, "1", "f"),
        ("nitrogen_04", Scale.LOG, "1", "f"),
        ("nitrogen_05", Scale.LOG, "1", "f"),
        ("nitrogen_06", Scale.LOG, "1", "f"),
        ("nitrogen_07", Scale.LOG, "1", "f"),
        ("carbon_01", Scale.LOG, "1", "f"),
        ("carbon_02", Scale.LOG, "1", "f"),
        ("carbon_03", Scale.LOG, "1", "f"),
        ("carbon_04", Scale.LOG, "1", "f"),
        ("carbon_05", Scale.LOG, "1", "f"),
        ("carbon_06", Scale.LOG, "1", "f"),
        ("magnesium_01", Scale.LOG, "1", "f"),
        ("magnesium_02", Scale.LOG, "1", "f"),
        ("magnesium_03", Scale.LOG, "1", "f"),
        ("magnesium_04", Scale.LOG, "1", "f"),
        ("magnesium_05", Scale.LOG, "1", "f"),
        ("magnesium_06", Scale.LOG, "1", "f"),
        ("magnesium_07", Scale.LOG, "1", "f"),
        ("magnesium_08", Scale.LOG, "1", "f"),
        ("magnesium_09", Scale.LOG, "1", "f"),
        ("magnesium_10", Scale.LOG, "1", "f"),
        ("silicon_01", Scale.LOG, "1", "f"),
        ("silicon_02", Scale.LOG, "1", "f"),
        ("silicon_03", Scale.LOG, "1", "f"),
        ("silicon_04", Scale.LOG, "1", "f"),
        ("silicon_05", Scale.LOG, "1", "f"),
        ("silicon_06", Scale.LOG, "1", "f"),
        ("silicon_07", Scale.LOG, "1", "f"),
        ("silicon_08", Scale.LOG, "1", "f"),
        ("silicon_09", Scale.LOG, "1", "f"),
        ("silicon_10", Scale.LOG, "1", "f"),
        ("silicon_11", Scale.LOG, "1", "f"),
        ("sulfur_01", Scale.LOG, "1", "f"),
        ("sulfur_02", Scale.LOG, "1", "f"),
        ("sulfur_03", Scale.LOG, "1", "f"),
        ("sulfur_04", Scale.LOG, "1", "f"),
        ("sulfur_05", Scale.LOG, "1", "f"),
        ("sulfur_06", Scale.LOG, "1", "f"),
        ("sulfur_07", Scale.LOG, "1", "f"),
        ("sulfur_08", Scale.LOG, "1", "f"),
        ("sulfur_09", Scale.LOG, "1", "f"),
        ("sulfur_10", Scale.LOG, "1", "f"),
        ("sulfur_11", Scale.LOG, "1", "f"),
        ("iron_01", Scale.LOG, "1", "f"),
        ("iron_02", Scale.LOG, "1", "f"),
        ("iron_03", Scale.LOG, "1", "f"),
        ("iron_04", Scale.LOG, "1", "f"),
        ("iron_05", Scale.LOG, "1", "f"),
        ("iron_06", Scale.LOG, "1", "f"),
        ("iron_07", Scale.LOG, "1", "f"),
        ("iron_08", Scale.LOG, "1", "f"),
        ("iron_09", Scale.LOG, "1", "f"),
        ("iron_10", Scale.LOG, "1", "f"),
        ("iron_11", Scale.LOG, "1", "f"),
        ("neon_01", Scale.LOG, "1", "f"),
        ("neon_02", Scale.LOG, "1", "f"),
        ("neon_03", Scale.LOG, "1", "f"),
        ("neon_04", Scale.LOG, "1", "f"),
        ("neon_05", Scale.LOG, "1", "f"),
        ("neon_06", Scale.LOG, "1", "f"),
        ("neon_07", Scale.LOG, "1", "f"),
        ("neon_08", Scale.LOG, "1", "f"),
        ("neon_09", Scale.LOG, "1", "f"),
        ("neon_10", Scale.LOG, "1", "f"),
        ("hydrogen_01", Scale.LOG, "1", "f"),
        ("hydrogen_02", Scale.LOG, "1", "f"),
        ("helium_02", Scale.LOG, "1", "f"),
        ("helium_03", Scale.LOG, "1", "f"),
        ("Habing", Scale.LOG, "erg/s/cm**2", "f"),
        ("Lyman_Werner", Scale.LOG, "erg/s/cm**2", "f"),
        ("HI_Ionising", Scale.LOG, "erg/s/cm**2", "f"),
        ("H2_Ionising", Scale.LOG, "erg/s/cm**2", "f"),
        ("HeI_Ionising", Scale.LOG, "erg/s/cm**2", "f"),
        ("HeII_ionising", Scale.LOG, "erg/s/cm**2", "f"),
        ("heating_rate", Scale.LOG, "erg/s", "f"),
        ("cooling_rate", Scale.LOG, "erg/s", "f"),
    ],
}


def load_cutout(
    filename: str | Path,
    boxsize: float = 50,
    h0: float = 0.6727,
    verbose: bool = True,
    version: int | list[tuple[str, Scale, str, str]] = 1,
):
    """Load a Megatron cutout file as a yt dataset.

    Parameters
    ----------
    filename : str | Path | url
        Path to the cutout file. If a URL, it will be downloaded using pooch.
    boxsize : boxsize in Mpccm/h
        The boxsize of the original simulation in comoving Mpc/h. Default is 50.
    h0 : float
        The Hubble constant of the original simulation. Default is 0.6727.
    verbose : bool
        Whether to show a progress bar when loading the data. Default is True.
    version : int or list of (name, scale, unit) tuples
        The version of the cutout format to load. If an int, it must be a key
        in the `headers` dict. If a list, it should be a custom header
        specification. Default is 2.

    Returns
    -------
    yt.Dataset
        The loaded yt dataset.
    """
    original_path = path = Path(filename)
    try:
        import numexpr as ne

        use_numexpr = True
    except ImportError:
        use_numexpr = False

    if not path.exists():
        path = Path(pooch.retrieve(str(filename), known_hash=None))

    if isinstance(version, int):
        header = headers[version]
    else:
        header = version

    data = {}
    with FortranFile(path, "r") as ff:
        prog = tqdm if verbose and yt.is_root() else lambda x, *args, **kwargs: x
        for name, scale, _unit, dtype in prog(header, desc="Loading cutout"):
            # Read in the quantity
            raw_data = ff.read_reals(dtype)
            if scale == Scale.LOG:
                if use_numexpr:
                    ne.evaluate("10 ** raw_data", out=raw_data)
                else:
                    raw_data = 10**raw_data

            if name == "density":
                if use_numexpr:
                    ne.evaluate("raw_data / 0.76", out=raw_data)  # Convert from nH to rho
                else:
                    raw_data = raw_data / 0.76
            data[name] = raw_data

    redshift = data.pop("redshift")[0]
    aexp = 1 / (1 + redshift)

    # Create a unyt registry
    boxsize_physical = boxsize * unyt.Mpc * aexp / h0
    registry = unyt.UnitRegistry()

    # Get xc (no need for unit conversion thus)
    xc = np.stack([data.pop(_) for _ in "xyz"], axis=-1)

    center = (xc.max(axis=0) + xc.min(axis=0)) / 2

    # Special case for dx (needs precise conversion from pc)
    dx = data.pop("dx") / 3.08e18 * unyt.pc / boxsize_physical

    # Convert everything else
    for name, _, unit in header:
        if name not in data:
            continue
        data[name] = unyt.unyt_array(data[name], unit, registry=registry)

    # Get level
    level = np.round(np.log2(1 / dx)).astype(int)

    left_edge = [0, 0, 0]
    right_edge = [1, 1, 1]

    # Compute left edge at coarsest level
    lcoarse = level.min()
    left_edge = np.round(xc.min(axis=0) * 2**lcoarse - 1) / (2**lcoarse)
    right_edge = np.round(xc.max(axis=0) * 2**lcoarse + 1) / (2**lcoarse)

    # Span should be a multiple of two
    span = (right_edge - left_edge).max()
    span_lvl = int(np.floor(np.log2(1 / span)))
    span_pow2 = 1 / 2**span_lvl

    xc -= left_edge
    xc /= span_pow2
    dx /= span_pow2

    right_edge = left_edge + span_pow2

    # Update level
    level -= span_lvl

    yt.mylog.debug("Building octree")
    octree = OctTree.from_list(xc, level)

    yt.mylog.debug("Depth-first traversal")
    ref_mask, leaf_order = octree.get_refmask()
    del octree

    nan_mask = np.where(leaf_order < 0, np.nan, 1)

    def reorder(dt):
        tmp = dt[leaf_order] * nan_mask
        return tmp[:, None]

    yt.mylog.debug("Reordering data according to octree leaf order")
    data = {("gas", k): reorder(v) for k, v in data.items()}

    params = {
        "cosmological_simulation": True,
        "current_redshift": redshift,
        "hubble_constant": h0,
    }

    yt.mylog.debug("Loading octree dataset")
    ds = yt.load_octree(
        octree_mask=ref_mask,
        data=data,
        bbox=np.array([left_edge, right_edge]).T,
        num_zones=1,
        dataset_name=original_path.name,
        parameters=params,
        length_unit=boxsize_physical,
    )
    ds.domain_center = ds.arr(center, "code_length")

    yt.mylog.debug("---------------------------------------------")

    # Add metal density fields
    def create_density(element: str):
        elem_name = metal_data[element]["name"]
        elem_mass = metal_data[element]["mass"]

        def _metal_density(field, data):
            return data["gas", f"{elem_name}_number_density"] * elem_mass

        ds.add_field(
            ("gas", f"{elem_name}_density"),
            function=_metal_density,
            units="amu/cm**3",
            display_name=f"{element} Mass Density",
            sampling_type="cell",
        )

        def _metal_mass_fraction(field, data):
            return data["gas", f"{elem_name}_density"] / data["gas", "density"]

        ds.add_field(
            ("gas", f"{elem_name}_fraction"),
            function=_metal_mass_fraction,
            units="1",
            display_name=f"{element} Mass Fraction",
            sampling_type="cell",
        )

    for element in metal_data.keys():
        create_density(element)

    return ds
