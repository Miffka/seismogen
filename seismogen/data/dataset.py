import os
import os.path as osp
from typing import Dict, Optional, Union

import albumentation as A
import cv2
import numpy as np
import pandas as pd
import torch

from seismogen.data.rle_utils import rle2mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, train_meta: str, transform: Optional[A.Compose] = None, train: bool = True):
        self.image_dir = image_dir
        self.image_names = os.listdir(self.image_dir)
        self.train_meta = (
            pd.read_csv(train_meta).drop_duplicates(["ImageId", "ClassId"]) if train_meta is not None else None
        )
        self.transform = transform

        self.train = train

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        img_name = self.image_names[index]  # ['ImageId']
        path = osp.join(self.image_dir, img_name)

        img = cv2.imread(path)
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
        if self.transform is not None:
            if self.train:
                img = self.transform(image=(img), mask=(ce_mask))
            else:
                img = self.transform(image=(img))

        result = {
            "image_shape": torch.tensor(img_shape),
            "image_name": img_name,
            "image": torch.tensor(img["image"]).transpose(-1, 0).transpose(-1, -2).float(),
            "mask": torch.tensor(img["mask"]).transpose(-1, 0).transpose(-1, -2).float() if self.train else [],
        }

        return result
