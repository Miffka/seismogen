from typing import List, Tuple

import numpy as np
import torch


def rle2mask(mask_rle: str, shape: Tuple) -> np.ndarray:
    """
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if mask_rle != mask_rle:
        return np.zeros(shape)
    elif mask_rle == "":
        return np.zeros(shape)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def mask2rle(img: np.ndarray) -> str:
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def out2rle(outputs: torch.Tensor) -> List[str]:
    rles = []

    try:
        prediction = outputs.detach().cpu().numpy()
    except Exception:
        prediction = outputs
    for j, sample in enumerate(prediction):
        for class_slice in sample:
            rles.append(mask2rle(class_slice.T > 0.5))
    return rles
