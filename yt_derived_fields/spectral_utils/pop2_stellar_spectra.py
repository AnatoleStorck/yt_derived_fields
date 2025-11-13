import numpy as np
import pandas as pd
import unyt as u

from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator

from pathlib import Path
from typing import Optional
from functools import cache

from yt.fields.field_detector import FieldDetector
from yt_derived_fields.spectral_utils.wavelength import wavelength_space, _block_mean_last_axis


# BPASS (v2.2.1) spectra (interpolated over metallicity and age)

# Grids for BPASS spectra
METAL_NAMES = [
    "zem5",
    "zem4",
    "z001",
    "z002",
    "z003",
    "z004",
    "z006",
    "z008",
    "z010",
    "z014",
    "z020",
    "z030",
    "z040",
]
METAL_VALS = np.array([1e-5, 1e-4, 1e-3, 2e-3, 3e-3, 4e-3, 6e-3, 8e-3, 1e-2, 1.4e-2, 2e-2, 3e-2, 4e-2])
AGES = 10.0 ** (6.0 + 0.1 * np.arange(51))

@cache
def _resolve_bpass_dir(data_dir: Optional[str]) -> Path:
    """
    Resolve directory containing the BPASS reduced spectra .npy files.
    Tries:
      - explicit data_dir if provided
      - BPASS_SPECTRA_DIR env var
      - known fallbacks
    """
    candidates: list[Path] = []
    if data_dir:
        candidates.append(Path(data_dir))
    # Fallback onto known paths (glamdring, infinity)
    candidates.append(Path("/mnt/glacier/DATA/bpass_v2.2.1_imf_chab300"))
    candidates.append(Path("/data100/cadiou/Megatron/DATA/bpass_v2.2.1_imf_chab300"))

    for base in candidates:
        test_file = base / f"reduced_spectra-bin-imf_chab300.{METAL_NAMES[0]}.dat.npy"
        if test_file.exists():
            return base

    raise FileNotFoundError(
        "Could not locate BPASS spectra directory. "
        "Pass data_dir=..., set BPASS_SPECTRA_DIR, or place files under one of the known paths."
    )


@cache
def _load_pop2_cube(data_dir: Optional[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all metallicities into a single cube once and cache.

    Returns:
      metal_vals: (n_metals,)
      ages:       (n_ages,)
      spectra:    (n_metals, n_ages, n_wvl_native) with units erg/s/Å per 1 Å bin
    """
    base = _resolve_bpass_dir(data_dir)

    # Load first file to determine native (n_ages, n_wvl) shape; files are stored as (n_wvl, n_ages) or vice versa, so we keep .T
    first = np.load((base / f"reduced_spectra-bin-imf_chab300.{METAL_NAMES[0]}.dat.npy"), mmap_mode="r").T
    if first.shape[0] != AGES.shape[0]:
        # Basic sanity check that the file layout matches the expected ages axis
        raise ValueError(
            f"Unexpected shape for BPASS data: got {first.shape}, expected first axis to match ages ({AGES.shape[0]})."
        )

    n_ages, n_wvl = first.shape
    n_metals = len(METAL_NAMES)

    all_spec = np.empty((n_metals, n_ages, n_wvl), dtype=first.dtype)
    all_spec[0] = first

    for i, m in enumerate(METAL_NAMES[1:], start=1):
        dat = np.load((base / f"reduced_spectra-bin-imf_chab300.{m}.dat.npy"), mmap_mode="r").T
        if dat.shape != (n_ages, n_wvl):
            raise ValueError(
                f"BPASS file for {m} has shape {dat.shape}, expected {(n_ages, n_wvl)}; "
                "ensure you’re using a consistent reduced grid."
            )
        all_spec[i] = dat

    return METAL_VALS, AGES, all_spec


@cache
def generate_pop_II_spec_interp(
    lmin: int,
    lmax: int,
    downsample: bool,
    ds_nwv: int,
    data_dir: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, RegularGridInterpolator]:
    """
    Build and cache an interpolator over (metallicity, age) for Pop II spectra.

    Returns:
      metal_vals: (n_metals,)
      ages:       (n_ages,)
      interpolator: RegularGridInterpolator((metal_vals, ages), spectra_view)
        - When called with points of shape (..., 2) returns (..., n_wvl or n_wvl_ds)
    Notes:
      - Assumes the reduced BPASS arrays are already on a native 1 Å grid covering [lmin, lmax].
    """

    metal_vals, ages, all_spec = _load_pop2_cube(data_dir)  # (M, A, W_native)
    # Validate the wavelength extent against the requested window length
    wvls_native = wavelength_space(lmin, lmax, downsample=False, ds_nwv=ds_nwv)
    n_wvl_expected = wvls_native.shape[0]
    if all_spec.shape[-1] != n_wvl_expected:
        raise ValueError(
            f"BPASS reduced spectra last axis length {all_spec.shape[-1]} does not match requested "
            f"wavelength window length {n_wvl_expected} for [{lmin}, {lmax}]. "
            "Adjust lmin/lmax to match the reduced grid or regenerate reduced arrays."
        )

    spectra_view = all_spec
    if downsample and ds_nwv > 1:
        spectra_view = _block_mean_last_axis(spectra_view, ds_nwv)

    interp = RegularGridInterpolator(
        (metal_vals, ages),
        spectra_view,
        bounds_error=False,
        fill_value=None,
    )
    return metal_vals, ages, interp


def get_pop_2_spectrum(
    data,
    combined: bool = False,
    lmin: int = 1150,
    lmax: int = 10000,
    downsample: bool = True,
    ds_nwv: int = 5,
    n_batch: int = 5000,
    ncpu_max: int = 10,
    data_dir: Optional[str] = None,
):
    """
    Calculates the Population II spectrum (BPASS v2.2.1).
    Units: erg/s per native bin (per 1 Å if using the standard reduced arrays).

    Params:
      data: mapping with required fields:
            - data["pop2", "particle_ones"]
            - data["pop2", "particle_metallicity_002"] (O)
            - data["pop2", "particle_metallicity_001"] (Fe)
            - data["pop2", "age"]
            - data["pop2", "particle_initial_mass"]
      combined: sum spectra over all particles if True
      lmin, lmax: wavelength range in Å
      downsample: block-mean spectra and wavelengths by ds_nwv
      ds_nwv: integer downsampling factor
      n_batch: batch size when parallelizing very large inputs
      ncpu_max: max CPUs for joblib parallelization
      data_dir: directory containing reduced_spectra-bin-imf_chab300.*.dat.npy files
      progress: if True, show a tqdm progress bar when parallelizing

    Returns:
      - If combined=True: 1D unyt array (n_wvl or n_wvl_ds,) with erg/s
      - If combined=False: 2D array (N_pop2, n_wvl or n_wvl_ds) with erg/s
    """

    N_pop2 = int(np.sum(data["pop2", "particle_ones"]))
    if N_pop2 == 0:
        # No Pop II stars; return zeroed spectrum (no contribution)
        return np.zeros_like(wavelength_space(lmin, lmax, downsample, ds_nwv)) * u.erg / u.s


    # Build interpolation inputs
    to_interp = np.empty((N_pop2, 2), dtype=float)
    met_O = np.asarray(data["pop2", "particle_metallicity_002"].value, dtype=float)
    met_Fe = np.asarray(data["pop2", "particle_metallicity_001"].value, dtype=float)
    Z_mix = 2.09 * met_O + 1.06 * met_Fe  # same mixture as before
    age_yr = data["pop2", "age"].to("yr").value

    # YT passes through an array of shape (8, 8, 8, 8) when initially detecting fields.
    # Stop the function before it gets to the interpolator, which expects a flattened array of sane values
    if isinstance(data, FieldDetector):
        return np.zeros(age_yr.shape) * u.erg / u.s

    # Get the interpolator (gives a spectra based on stellar mass and age)
    metal_grid, ages_grid, spec_interp_p2 = generate_pop_II_spec_interp(
        lmin=lmin, lmax=lmax, downsample=downsample, ds_nwv=ds_nwv, data_dir=data_dir
    )

    # Enforce grid bounds
    to_interp[:, 0] = np.clip(Z_mix, metal_grid.min(), metal_grid.max())
    to_interp[:, 1] = np.clip(age_yr, ages_grid.min(), ages_grid.max())

    # Get a list of initial masses
    initial_masses = data["pop2", "particle_initial_mass"].to("Msun").value

    # Don't parallelize for small numbers of particles
    if N_pop2 <= 1e5:
        # If there are few particles, do not parallelize
        p2_spec = spec_interp_p2(to_interp) * initial_masses[:, None]
        if combined:
            p2_spec = p2_spec.sum(axis=0)
        return p2_spec * u.erg / u.s

    # Chunk the data for efficient parallelization
    all_c1 = [i * n_batch for i in range(1 + len(to_interp) // n_batch)]
    all_c2 = [c1 + n_batch for c1 in all_c1]
    all_c2[-1] = len(to_interp)
    n_cpus = min(len(all_c1), ncpu_max)

    test_shape = spec_interp_p2(to_interp[:1]).shape[-1]
    results = np.zeros((N_pop2, test_shape))  # shape: (N_cells, N_wavelengths)

    def batch_interp(c1, c2):
        return spec_interp_p2(to_interp[c1:c2, :]) * initial_masses[c1:c2, None]

    from tqdm import tqdm
    batch_results = Parallel(n_jobs=n_cpus)(delayed(batch_interp)(all_c1[i], all_c2[i]) for i in tqdm(range(len(all_c1))))

    # Fill results array
    for i, batch_result in enumerate(batch_results):
        results[all_c1[i] : all_c2[i], :] = batch_result

    p2_spec = np.array(results)
    if combined:
        p2_spec = p2_spec.sum(axis=0)

    return p2_spec * u.erg / u.s
