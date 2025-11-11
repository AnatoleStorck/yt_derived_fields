from pathlib import Path

import numpy as np
import unyt
import yt
from scipy.io import FortranFile
from yt_experiments.octree.converter import OctTree
from enum import Enum
from tqdm import tqdm
import numexpr as ne
from yt_derived_fields.megatron_derived_fields.chemistry_derived_fields import metal_data


class Scale(Enum):
    LINEAR = 0
    LOG = 1


header: list[tuple[str, Scale, str]] = [
    ("redshift", Scale.LINEAR, "1"),
    ("dx", Scale.LOG, "cm"),
    ("x", Scale.LINEAR, "Mpccm/h"),
    ("y", Scale.LINEAR, "Mpccm/h"),
    ("z", Scale.LINEAR, "Mpccm/h"),
    ("vx", Scale.LINEAR, "cm/s"),
    ("vy", Scale.LINEAR, "cm/s"),
    ("vz", Scale.LINEAR, "cm/s"),
    ("density", Scale.LOG, "mp/cm**3"),
    ("temperature", Scale.LOG, "K"),
    ("pressure", Scale.LINEAR, "dyne/cm**2"),
    ("iron_number_density", Scale.LOG, "1/cm**3"),
    ("oxygen_number_density", Scale.LOG, "1/cm**3"),
    ("nitrogen_number_density", Scale.LOG, "1/cm**3"),
    ("magnesium_number_density", Scale.LOG, "1/cm**3"),
    ("neon_number_density", Scale.LOG, "1/cm**3"),
    ("silicon_number_density", Scale.LOG, "1/cm**3"),
    ("calcium_number_density", Scale.LOG, "1/cm**3"),
    ("carbon_number_density", Scale.LOG, "1/cm**3"),
    ("sulfur_number_density", Scale.LOG, "1/cm**3"),
    ("carbon_monoxide_number_density", Scale.LOG, "1/cm**3"),
    ("oxygen_01", Scale.LINEAR, "1"),
    ("oxygen_02", Scale.LINEAR, "1"),
    ("oxygen_03", Scale.LINEAR, "1"),
    ("oxygen_04", Scale.LINEAR, "1"),
    ("oxygen_05", Scale.LINEAR, "1"),
    ("oxygen_06", Scale.LINEAR, "1"),
    ("oxygen_07", Scale.LINEAR, "1"),
    ("oxygen_08", Scale.LINEAR, "1"),
    ("nitrogen_01", Scale.LINEAR, "1"),
    ("nitrogen_02", Scale.LINEAR, "1"),
    ("nitrogen_03", Scale.LINEAR, "1"),
    ("nitrogen_04", Scale.LINEAR, "1"),
    ("nitrogen_05", Scale.LINEAR, "1"),
    ("nitrogen_06", Scale.LINEAR, "1"),
    ("nitrogen_07", Scale.LINEAR, "1"),
    ("carbon_01", Scale.LINEAR, "1"),
    ("carbon_02", Scale.LINEAR, "1"),
    ("carbon_03", Scale.LINEAR, "1"),
    ("carbon_04", Scale.LINEAR, "1"),
    ("carbon_05", Scale.LINEAR, "1"),
    ("carbon_06", Scale.LINEAR, "1"),
    ("magnesium_01", Scale.LINEAR, "1"),
    ("magnesium_02", Scale.LINEAR, "1"),
    ("magnesium_03", Scale.LINEAR, "1"),
    ("magnesium_04", Scale.LINEAR, "1"),
    ("magnesium_05", Scale.LINEAR, "1"),
    ("magnesium_06", Scale.LINEAR, "1"),
    ("magnesium_07", Scale.LINEAR, "1"),
    ("magnesium_08", Scale.LINEAR, "1"),
    ("magnesium_09", Scale.LINEAR, "1"),
    ("magnesium_10", Scale.LINEAR, "1"),
    ("silicon_01", Scale.LINEAR, "1"),
    ("silicon_02", Scale.LINEAR, "1"),
    ("silicon_03", Scale.LINEAR, "1"),
    ("silicon_04", Scale.LINEAR, "1"),
    ("silicon_05", Scale.LINEAR, "1"),
    ("silicon_06", Scale.LINEAR, "1"),
    ("silicon_07", Scale.LINEAR, "1"),
    ("silicon_08", Scale.LINEAR, "1"),
    ("silicon_09", Scale.LINEAR, "1"),
    ("silicon_10", Scale.LINEAR, "1"),
    ("silicon_11", Scale.LINEAR, "1"),
    ("sulfur_01", Scale.LINEAR, "1"),
    ("sulfur_02", Scale.LINEAR, "1"),
    ("sulfur_03", Scale.LINEAR, "1"),
    ("sulfur_04", Scale.LINEAR, "1"),
    ("sulfur_05", Scale.LINEAR, "1"),
    ("sulfur_06", Scale.LINEAR, "1"),
    ("sulfur_07", Scale.LINEAR, "1"),
    ("sulfur_08", Scale.LINEAR, "1"),
    ("sulfur_09", Scale.LINEAR, "1"),
    ("sulfur_10", Scale.LINEAR, "1"),
    ("sulfur_11", Scale.LINEAR, "1"),
    ("iron_01", Scale.LINEAR, "1"),
    ("iron_02", Scale.LINEAR, "1"),
    ("iron_03", Scale.LINEAR, "1"),
    ("iron_04", Scale.LINEAR, "1"),
    ("iron_05", Scale.LINEAR, "1"),
    ("iron_06", Scale.LINEAR, "1"),
    ("iron_07", Scale.LINEAR, "1"),
    ("iron_08", Scale.LINEAR, "1"),
    ("iron_09", Scale.LINEAR, "1"),
    ("iron_10", Scale.LINEAR, "1"),
    ("iron_11", Scale.LINEAR, "1"),
    ("neon_01", Scale.LINEAR, "1"),
    ("neon_02", Scale.LINEAR, "1"),
    ("neon_03", Scale.LINEAR, "1"),
    ("neon_04", Scale.LINEAR, "1"),
    ("neon_05", Scale.LINEAR, "1"),
    ("neon_06", Scale.LINEAR, "1"),
    ("neon_07", Scale.LINEAR, "1"),
    ("neon_08", Scale.LINEAR, "1"),
    ("neon_09", Scale.LINEAR, "1"),
    ("neon_10", Scale.LINEAR, "1"),
    ("hydrogen_01", Scale.LINEAR, "1"),
    ("hydrogen_02", Scale.LINEAR, "1"),
    ("helium_02", Scale.LINEAR, "1"),
    ("helium_03", Scale.LINEAR, "1"),
    ("Habing", Scale.LOG, "erg/s/cm**2"),
    ("Lyman_Werner", Scale.LOG, "erg/s/cm**2"),
    ("HI_Ionising", Scale.LOG, "erg/s/cm**2"),
    ("H2_Ionising", Scale.LOG, "erg/s/cm**2"),
    ("HeI_Ionising", Scale.LOG, "erg/s/cm**2"),
    ("HeII_ionising", Scale.LOG, "erg/s/cm**2"),
]


def load_cutout(filename: str | Path, boxsize: unyt.unyt_quantity, h0: float = 0.6727, verbose: bool = True):
    filename = Path(filename)

    data = {}
    with FortranFile(filename, "r") as ff:
        prog = tqdm if verbose and yt.is_root() else lambda x, *args, **kwargs: x
        for name, scale, _unit in prog(header, desc="Loading cutout"):
            # Read in the quantity
            raw_data = ff.read_reals("float64")
            if scale == Scale.LOG:
                ne.evaluate("10 ** raw_data", out=raw_data)

            if name == "density":
                ne.evaluate("raw_data / 0.76", out=raw_data)  # Convert from nH to rho
            data[name] = raw_data

    redshift = data.pop("redshift")[0]
    aexp = 1 / (1 + redshift)

    # Create a unyt registry
    boxsize_physical = boxsize * aexp / h0
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

    yt.mylog.debug("Building octree")
    oct = OctTree.from_list(xc, level)

    yt.mylog.debug("Depth-first traversal")
    ref_mask, leaf_order = oct.get_refmask()

    nan_mask = np.where(leaf_order < 0, np.nan, 1)

    def reorder(dt):
        tmp = dt[leaf_order] * nan_mask
        return tmp[:, None]

    yt.mylog.debug("Reordering data according to octree leaf order")
    data = {("gas", k): reorder(v) for k, v in data.items()}

    left_edge = [0, 0, 0]
    right_edge = [1, 1, 1]

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
        dataset_name=f"Cutout/{filename.name}",
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
