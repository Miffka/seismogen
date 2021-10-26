import albumentations as A
import torch


def get_transforms(normalize: bool = True, num_channels: int = 3) -> A.Compose:

    if normalize:
        mean_ = [0.5] * num_channels
        std_ = [0.13] * num_channels
    else:
        mean_ = [0.0] * num_channels
        std_ = [1.0] * num_channels

    transforms = A.Compose(
        [
            A.Normalize(mean=mean_, std=std_),
        ]
    )

    return transforms


def backward_transform(image: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    if normalized:
        ret_img = (image * 0.13 + 0.5) * 255.0
    else:
        ret_img = image * 255.0

    return ret_img
