import numpy as np

# Most interpolation functions are generated on a wavelength grid of 1 A
# This function is used to downsample the spectra if desired
def _block_mean_last_axis(a: np.ndarray, factor: int) -> np.ndarray:
    """
    Block-mean downsampling along the last axis.
    Trims any leftover elements so the last axis is divisible by factor.
    """
    if factor is None or factor <= 1:
        return np.asanyarray(a)
    n = a.shape[-1] - (a.shape[-1] % factor)
    if n <= 0:
        raise ValueError(f"Downsample factor {factor} is larger than array length {a.shape[-1]}.")
    a_trim = a[..., :n]
    new_shape = a_trim.shape[:-1] + (n // factor, factor)
    return a_trim.reshape(new_shape).mean(axis=-1)


def wavelength_space(lmin: int, lmax: int, downsample: bool, ds_nwv: int) -> np.ndarray:
    """
    Wavelength array in Å. When downsample=True, returns block-meaned bin centers.
    """
    wvls = np.arange(lmin, lmax + 1, dtype=float)
    if downsample and ds_nwv > 1:
        # block-mean the axis to align with block-mean spectra
        wvls = _block_mean_last_axis(wvls[None, :], ds_nwv)[0]
    return wvls