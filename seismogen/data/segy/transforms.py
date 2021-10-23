import albumentations as A


def get_transforms(normalize: bool = True, num_channels: int = 3) -> A.Compose:

    if normalize:
        raise NotImplementedError("Normalization is not implemented yet")
    else:
        mean_ = [0.0] * num_channels
        std_ = [1.0] * num_channels

    transforms = A.Compose(
        [
            A.Normalize(mean=mean_, std=std_),
        ]
    )

    return transforms
