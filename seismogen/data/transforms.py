import albumentations as A


def get_transforms(size: int = 224, num_channels: int = 3) -> A.Compose:

    transforms = A.Compose(
        [
            A.PadIfNeeded(size, size),
            A.Resize(size, size),
            A.Normalize(mean=[0.502] * num_channels, std=[0.09] * num_channels),
        ]
    )

    return transforms
