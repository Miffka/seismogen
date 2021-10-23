import os.path as osp
from typing import Dict, Optional, Tuple, Union

import albumentations
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms

from seismogen.data.letterbox import letterbox_forward


class SEGYDataset:
    def __init__(
        self,
        markup_path: str,
        data_dir: str,
        target_type: Optional[str] = None,
        test_size: float = 0.0,
        split: str = "train",
        random_state: int = 24,
        size: int = 512,
        letterbox: bool = False,
        augmentation: Optional[albumentations.Compose] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        self.markup_path = markup_path
        self.data_dir = data_dir
        self.target_type = target_type

        self.test_size = test_size
        assert split in ["train", "val", "test"]
        self.split = split
        self.random_state = random_state

        self.size = size
        self.letterbox = letterbox
        self.augmentation = augmentation
        self.transform = transform

        self.markup = self.init_markup()
        self.dataset_name = self.markup["ds_name"].unique()[0]

    def init_markup(self):
        markup = pd.read_csv(self.markup_path)
        if self.test_size > 0:
            train_val_markup, test_markup = train_test_split(
                markup,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=markup["orient"],
            )
            val_size = self.test_size / (1 - self.test_size)
            train_markup, val_markup = train_test_split(
                train_val_markup,
                test_size=val_size,
                random_state=self.random_state,
                stratify=train_val_markup["orient"],
            )

            if self.split == "train":
                markup = train_markup
            elif self.split == "val":
                markup = val_markup
            else:
                markup = test_markup

        return markup

    def __len__(self):
        return self.markup.shape[0]

    def get_target_path(self, row):
        if self.target_type is None:
            return None

        target_col = f"{self.target_type}_path"
        if target_col in row:
            return osp.join(self.data_dir, row[target_col])

        return None

    def read_img_and_target(self, img_fpath: str, targ_fpath: Optional[str] = None) -> Tuple[Union[np.ndarray, None]]:
        img = cv2.imread(img_fpath)[:, :, :1]

        if targ_fpath is not None:
            targ = (cv2.imread(targ_fpath)[:, :, 0] > 0).astype(np.long)
        else:
            targ = None

        return img, targ

    def apply_aug_transform(
        self, img: np.ndarray, ce_mask: Optional[np.ndarray] = None
    ) -> Tuple[Union[torch.Tensor, None, np.ndarray]]:
        if self.letterbox:
            img, pad = letterbox_forward(img, size=self.size)
            if ce_mask is not None:
                ce_mask, _ = letterbox_forward(ce_mask, size=self.size, mask=True)
        else:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
            if ce_mask is not None:
                ce_mask = cv2.resize(ce_mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            pad = None

        if img.ndim == 2:
            img = np.expand_dims(img, 2)

        if self.augmentation is not None:
            if ce_mask is not None:
                img_mask_dict = self.augmentation(image=img, mask=ce_mask)
                img, ce_mask = img_mask_dict["image"], img_mask_dict["mask"]
            else:
                img = self.augmentation(image=img)["image"]

        if self.transform is not None:
            if ce_mask is not None:
                img_mask_dict = self.transform(image=(img), mask=(ce_mask))
                img, ce_mask = img_mask_dict["image"], img_mask_dict["mask"]
            else:
                img = self.transform(image=(img))["image"]

        return img, ce_mask, pad

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        row = self.markup.iloc[idx]
        img_fpath = osp.join(self.data_dir, row["volume_path"])
        target_fpath = self.get_target_path(row)

        img, targ = self.read_img_and_target(img_fpath, target_fpath)
        img_shape = img.shape[:2]
        img, targ, pad = self.apply_aug_transform(img, targ)

        result = {
            "image_shape": torch.tensor(img_shape),
            "dataset_name": row["ds_name"],
            "orient": row["orient"],
            "axis_idx": torch.tensor(row["axis_idx"]),
            "orient_idx": torch.tensor(row["orient_idx"]),
            "pad": str(pad),
            "target_type": self.target_type,
            "image": torch.tensor(img).transpose(2, 0).transpose(1, 2).float(),
            "target": torch.tensor(targ) if targ is not None else [],
        }

        return result
