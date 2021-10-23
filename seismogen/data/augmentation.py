from typing import Optional

import albumentations as A
import cv2

from seismogen.models.fix_seeds import fix_seeds


def get_augmentations(resize: int = 224, augmentation_intensity: Optional[str] = None) -> A.Compose:
    fix_seeds(24)
    crop_limits = (int(resize * 0.85), resize)

    if augmentation_intensity == "slight":
        p_augment = 0.15
        p_scale = 0.15
        p_blur = 0.05
        p_dropout = 0.05
        p_flip = 0.15
        p_noise = 0.15
        gauss_limit = 0.005

    elif augmentation_intensity == "light":
        p_augment = 0.25
        p_scale = 0.2
        p_blur = 0.1
        p_dropout = 0.05
        p_flip = 0.2
        p_noise = 0.2
        gauss_limit = 0.01

    elif augmentation_intensity == "medium":
        p_augment = 0.5
        p_scale = 0.2
        p_blur = 0.2
        p_dropout = 0.1
        p_flip = 0.2
        p_noise = 0.2
        gauss_limit = 0.015

    elif augmentation_intensity == "heavy":
        p_augment = 0.5
        p_scale = 0.35
        p_blur = 0.35
        p_dropout = 0.15
        p_flip = 0.35
        p_noise = 0.35
        gauss_limit = 0.02

    elif augmentation_intensity is None:
        return None
    else:
        raise ValueError("Improper augmentation flag: should be equal to None, light, medium, or heavraisey")

    augmentation = A.Compose(
        [
            A.OneOf(
                [A.HorizontalFlip(), A.VerticalFlip()],
                p=p_flip,
            ),
            A.OneOf(
                [A.Rotate(p=1.0, limit=30), A.RandomRotate90(p=1.0)],
                p=p_scale,
            ),
            A.OneOf(
                [
                    A.ShiftScaleRotate(p=1.0, rotate_limit=30),
                    A.RandomSizedCrop(
                        min_max_height=crop_limits,
                        height=resize,
                        width=resize,
                        w2h_ratio=1.0,
                        interpolation=cv2.INTER_CUBIC,
                    ),
                ],
                p=p_scale,
            ),
            A.Blur(blur_limit=3, p=p_blur),
            A.CoarseDropout(max_height=7, max_width=7, p=p_dropout),
            A.GaussNoise(var_limit=(0.0, gauss_limit), p=p_noise),
        ],
        p=p_augment,
    )

    return augmentation
