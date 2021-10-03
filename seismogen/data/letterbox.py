from typing import Dict, Tuple, Union

import cv2
import numpy as np


def letterbox_forward(
    img: np.ndarray, size: int, mask: bool = False, fill_color: int = 0
) -> Tuple[np.ndarray, Dict[str, Union[Tuple[int], float, str]]]:
    """
    Arguments
    ---------
        img     (np.ndarray) : 2D or 3D array, shape (height, width, *num_channels)
        size           (int) : target size, (height, width)
        mask          (bool) : whether the input image is a mask (required nearest interpolation)
        fill_color     (int) : value of color to fill borders

    Returns
    -------
        img     (np.ndarray) : image with shape "size"
        (optinal) pad (dict) : dictionary with padding info, "pad" key has
                               pad size in format (xyxy)

    Convert image to specified size by letterboxing.
    """
    net_h, net_w = size, size
    im_h, im_w = img.shape[:2]
    pad = {"pad": (0, 0, 0, 0), "orig_size": (im_h, im_w), "target_size": (net_h, net_w), "scale": 1.0, "type": None}

    if im_w == net_w and im_h == net_h:
        return img, pad

    if im_w / net_w >= im_h / net_h:
        scale = net_w / im_w
        pad["type"] = "height"
    else:
        scale = net_h / im_h
        pad["type"] = "width"

    if scale != 1:
        img = img.copy().astype(np.float32)
        if mask:
            img = cv2.resize(img, (int(scale * im_w), int(scale * im_h)), interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.resize(img, (int(scale * im_w), int(scale * im_h)), interpolation=cv2.INTER_LINEAR)
        im_h, im_w = img.shape[:2]
        pad["scale"] = scale

    if im_w == net_w and im_h == net_h:
        return img, pad

    # padding
    pad_w = (net_w - im_w) / 2
    pad_h = (net_h - im_h) / 2

    pad["pad"] = (int(pad_w), int(pad_h), int(pad_w + 0.5), int(pad_h + 0.5))
    if img.ndim == 2:
        pad_ = ((pad["pad"][1], pad["pad"][3]), (pad["pad"][0], pad["pad"][2]))
    else:
        pad_ = ((pad["pad"][1], pad["pad"][3]), (pad["pad"][0], pad["pad"][2]), (0, 0))
    img = np.pad(
        img,
        pad_,
        mode="constant",
        constant_values=fill_color,
    )

    return img, pad


def letterbox_backward(image: np.ndarray, pad: Dict[str, Union[Tuple[int], float, str]]) -> np.ndarray:

    if pad["type"] is None:
        return image

    # Crop image
    h_coords = (pad["pad"][1], pad["target_size"][0] - pad["pad"][3])
    w_coords = (pad["pad"][0], pad["target_size"][1] - pad["pad"][2])
    image = image[:, h_coords[0] : h_coords[1], w_coords[0] : w_coords[1]]

    cropped_size = image.shape[:2]
    if cropped_size == pad["orig_size"]:
        return image

    # Resize image
    img_list = []
    for channel_img in image:
        img_list.append(
            cv2.resize(channel_img, (pad["orig_size"][1], pad["orig_size"][0]), interpolation=cv2.INTER_NEAREST)
        )

    return np.stack(img_list)
