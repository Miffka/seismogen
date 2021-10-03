import os
import os.path as osp
from typing import Dict, Optional, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from seismogen.data.letterbox import letterbox_forward
from seismogen.data.rle_utils import rle2mask


class SegDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_dir: str,
        train_meta: str,
        num_channels: int = 3,
        size: int = 224,
        letterbox: bool = False,
        augmentation: Optional[A.Compose] = None,
        transform: Optional[A.Compose] = None,
        split: str = "train",
        val_size: float = 0.2,
    ):
        self.image_dir = image_dir
        self.image_names = sorted(os.listdir(self.image_dir))
        self.train_meta = (
            pd.read_csv(train_meta).drop_duplicates(["ImageId", "ClassId"]) if train_meta is not None else None
        )
        assert num_channels in [1, 3], f"Num channels should be in [1, 3], got {num_channels}"
        self.num_channels = num_channels

        self.size = size
        self.letterbox = letterbox
        self.augmentation = augmentation
        self.transform = transform

        assert split in ["train", "val", "test"], f"Split value should be in ['train', 'val', 'test'], got {split}"
        self.split = split
        self.train = self.split in ["train", "val"]

        self.get_train_val_split(val_size)
        self.track_num = osp.basename(image_dir).split(".")[0].split("_")[-1]

    def get_train_val_split(self, val_size: float):
        if self.split in ["train", "val"]:
            train_imgs, val_imgs = train_test_split(self.image_names, test_size=val_size, random_state=24)
            if self.split == "train":
                self.image_names = train_imgs
            else:
                self.image_names = val_imgs

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        img_name = self.image_names[index]  # ['ImageId']
        path = osp.join(self.image_dir, img_name)

        img = cv2.imread(path)
        if self.num_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        assert img.ndim == 3, f"Image should have 3 dimensions, got {img.ndim}"

        img_shape = (img.shape[0], img.shape[1])
        ce_mask = (
            [
                (1) * rle2mask(rle, shape=(img.shape[1], img.shape[0]))[:, :, None]
                for i, rle in enumerate(self.train_meta[self.train_meta["ImageId"] == img_name]["EncodedPixels"])
            ]
            if self.train
            else None
        )
        ce_mask = np.concatenate(ce_mask, -1) if self.train else None
        # np.sum(ce_mask, axis=0, dtype=np.float32)[:, :, None]

        if self.letterbox:
            img, pad = letterbox_forward(img, size=self.size)
            if ce_mask is not None:
                ce_mask, _ = letterbox_forward(ce_mask, size=self.size, mask=True)
        else:
            pad = None

        if self.augmentation is not None:
            img_mask_dict = self.augmentation(image=img, mask=ce_mask)
            img, ce_mask = img_mask_dict["image"], img_mask_dict["mask"]

        if self.transform is not None:
            if self.train:
                img = self.transform(image=(img), mask=(ce_mask))
            else:
                img = self.transform(image=(img))

        result = {
            "image_shape": torch.tensor(img_shape),
            "image_name": img_name,
            "pad": str(pad),
            "image": torch.tensor(img["image"]).transpose(-1, 0).transpose(-1, -2).float(),
            "mask": torch.tensor(img["mask"]).transpose(-1, 0).transpose(-1, -2).float() if self.train else [],
        }

        return result
