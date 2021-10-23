import albumentations as A


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
