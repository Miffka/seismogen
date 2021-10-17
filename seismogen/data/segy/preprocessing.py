import numpy as np


def clip_normalize(array: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    out_arr = array.copy().astype(np.float)

    right = np.percentile(array, percentile)
    left = np.percentile(array, 100 - percentile)
    bound = np.max([np.abs(left), np.abs(right)])

    np.clip(array, -bound, bound, out_arr)
    out_arr /= bound

    return out_arr


def scale_to_uint8(array: np.ndarray) -> np.ndarray:

    return ((array - array.min()) / array.ptp() * 255).round().astype(np.uint8)
