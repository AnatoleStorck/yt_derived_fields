import numpy as np
import pandas as pd
import unyt as u

from scipy.interpolate import RegularGridInterpolator

from pathlib import Path
from typing import Optional
from functools import cache

from yt_derived_fields.spectral_utils.wavelength import wavelength_space, _block_mean_last_axis


@cache
def _resolve_data_paths(data_dir: Optional[str]) -> tuple[Path, Path]:
    """
    Resolve the numpy spectra file and the params csv. Tries:
    - explicit data_dir if provided
    - POP3_SPECTRA_DIR environment variable
    - known fallbacks
    """
    candidates = []
    if data_dir:
        candidates.append(Path(data_dir))
    # Fallback onto known paths (glamdring, infinity)
    candidates.append(Path("/mnt/glacier/DATA/Pop_III_spectra"))
    candidates.append(Path("/data100/cadiou/Megatron/DATA/Pop_III_spectra"))

    for base in candidates:
        spec = base / "reduced_popiii_spec.npy"
        params = base / "model_params.dat"
        if spec.exists() and params.exists():
            return spec, params

    raise FileNotFoundError(
        "Could not locate Pop III spectra files. "
        "Pass data_dir=<...> or place files under one of the known paths."
    )


@cache
def _load_pop3_data(data_dir: Optional[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Load masses (Msun) and spectra (erg/s/Å) once; caches results.
    Returns:
      masses: shape (n_masses,)
      spectra: shape (n_masses, n_wvl)
    """
    spec_path, params_path = _resolve_data_paths(data_dir)
    # mmap to avoid copying large arrays on repeated calls
    spectra = np.load(spec_path, mmap_mode="r")  # (n_masses, n_wvl)
    props = pd.read_csv(params_path)
    masses = props["Mass_Msol"].to_numpy()
    return masses, spectra


# Larkin et al. (2023) spectra (interpolated over mass)
def generate_pop_III_spec_interp(
    downsample: bool,
    ds_nwv: int,
    data_dir: Optional[str] = None,
) -> RegularGridInterpolator:
    """
    Build an interpolator over mass for Pop III spectra.

    Returns:
      RegularGridInterpolator over mass -> spectrum vector (erg/s/Å)
    Notes:
      - Uses block-mean downsampling for speed and consistency.
      - Does not crop by [lmin, lmax] because the native wavelength grid is defined by the data file.
    """
    masses, spectra = _load_pop3_data(data_dir)  # spectra: (n_masses, n_wvl)

    if downsample and ds_nwv > 1:
        spectra = _block_mean_last_axis(spectra, ds_nwv)  # still (n_masses, n_wvl_ds)

    # Interpolator over mass; returns a vector of shape (n_wvl or n_wvl_ds,)
    return RegularGridInterpolator((masses,), spectra, bounds_error=False, fill_value=None)


def get_pop_3_spectrum(
    data,
    combined: bool = True,
    lmin: int = 1150,
    lmax: int = 10000,
    downsample: bool = True,
    ds_nwv: int = 5,
    data_dir: Optional[str] = None,
):
    """
    Calculates the Population III spectrum.
    Units: erg/s (per 1 Å bin if your data are per-Å). The native spectra are per Å.

    Params:
      data: a yt data object or similar providing
            data["pop3", "isAlive"] and data["pop3", "particle_initial_mass"]
      combined: if True, returns the summed spectrum for all alive pop3 stars
      lmin, lmax: wavelength range used for reporting wavelength arrays and sizing empty outputs
      downsample: whether to block-mean downsample the spectra
      ds_nwv: integer downsampling factor (e.g., 5 => averages every 5 Å bin)
      data_dir: optional directory containing reduced_popiii_spec.npy and model_params.dat
                If omitted, POP3_SPECTRA_DIR or known fallbacks are used.

    Returns:
      - If combined=True: 1D spectrum with shape (n_wvl or n_wvl_ds,) and units of erg/s
      - If combined=False: 2D array (n_alive, n_wvl or n_wvl_ds) with units of erg/s
    """
    pop3_alive_status = data["pop3", "isAlive"]

    # Fast path: no alive stars
    if not np.any(pop3_alive_status):
        return np.zeros_like(wavelength_space(lmin, lmax, downsample, ds_nwv)) * u.erg / u.s

    # Active masses in Msun; derive bounds from the data itself
    active_popIII_masses = data["pop3", "particle_initial_mass"][pop3_alive_status].to("Msun").value
    masses_all, _ = _load_pop3_data(data_dir)
    active_popIII_masses = np.clip(active_popIII_masses, masses_all.min(), masses_all.max())

    # Interpolate spectra for the active masses
    spec_interp_p3 = generate_pop_III_spec_interp(downsample, ds_nwv, data_dir)
    p3_spec = spec_interp_p3(active_popIII_masses)  # (n_alive, n_wvl[_ds])

    if combined:
        p3_spec = p3_spec.sum(axis=0)

    # Keep units consistent with prior return (erg/s); if your data are per-Å this is erg/s/Å per bin
    return p3_spec * u.erg / u.s
