import albumentations as A


def get_transforms(size: int = 224, normalize: bool = True, num_channels: int = 3, nearest: bool = False) -> A.Compose:

    if normalize:
        if nearest:
            mean_ = [0.48, 0.50, 0.17]
            std_ = [0.084, 0.045, 0.28]
        else:
            mean_ = [0.502] * num_channels
            std_ = [0.09] * num_channels
    else:
        mean_ = [0.0] * num_channels
        std_ = [1.0] * num_channels

    transforms = A.Compose(
        [
            A.PadIfNeeded(size, size),
            A.Resize(size, size),
            A.Normalize(mean=mean_, std=std_),
        ]
    )

    return transforms
