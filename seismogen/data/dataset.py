import os
import os.path as osp
from typing import Dict, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from seismogen.data import nearest  # noqa F401
from seismogen.data.letterbox import letterbox_forward
from seismogen.data.rle_utils import rle2mask


class SegDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_dir: str,
        train_meta: str,
        num_channels: int = 3,
        size: int = 224,
        mode: str = "multilabel",
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
        assert mode in {"multiclass", "multilabel"}
        self.mode = mode
        self.mask_dtype = torch.float if mode == "multilabel" else torch.long
        self.letterbox = letterbox
        self.augmentation = augmentation
        self.transform = transform

        assert split in ["train", "val", "test"], f"Split value should be in ['train', 'val', 'test'], got {split}"
        self.split = split
        self.train = self.split in ["train", "val"]

        self.get_train_val_split(val_size)
        self.track_num = osp.basename(image_dir).split(".")[0].split("_")[-1]

    def get_train_val_split(self, val_size: float):
        if self.split in ["train", "val"] and val_size > 0:
            train_imgs, val_imgs = train_test_split(self.image_names, test_size=val_size, random_state=24)
            self.train_imgs = train_imgs
            if self.split == "train":
                self.image_names = train_imgs
            else:
                self.image_names = val_imgs

    def __len__(self) -> int:
        return len(self.image_names)

    def read_image(self, img_name: str, image_dir: str) -> np.ndarray:
        path = osp.join(image_dir, img_name)

        img = cv2.imread(path)
        if self.num_channels == 1:
            img = np.expand_dims(img[:, :, 0], axis=2)
        assert img.ndim == 3, f"Image should have 3 dimensions, got {img.ndim}"

        return img

    def get_gt_image_and_mask(
        self, img_name: str, meta_df: pd.DataFrame, image_dir: str, force_load_mask: bool = False
    ) -> Tuple[np.ndarray]:
        img = self.read_image(img_name, image_dir=image_dir)

        ce_mask = (
            [
                (1) * rle2mask(rle, shape=(img.shape[1], img.shape[0]))[:, :, None]
                for i, rle in enumerate(meta_df[meta_df["ImageId"] == img_name]["EncodedPixels"])
            ]
            if self.train or force_load_mask
            else None
        )
        ce_mask = np.concatenate(ce_mask, -1) if self.train or force_load_mask else None
        if self.mode == "multiclass" and ce_mask is not None:
            ce_mask = (ce_mask * np.arange(1, 8)[None, None, :]).max(axis=2).astype(np.long)

        return img, ce_mask

    def apply_aug_transform(self, img: np.ndarray, ce_mask: np.ndarray) -> Tuple[Union[torch.Tensor, None, np.ndarray]]:

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
                img_mask_dict = self.transform(image=(img), mask=(ce_mask))
                img, ce_mask = img_mask_dict["image"], img_mask_dict["mask"]
            else:
                img = self.transform(image=(img))["image"]

        return img, ce_mask, pad

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        img_name = self.image_names[index]  # ['ImageId']
        img, ce_mask = self.get_gt_image_and_mask(img_name, self.train_meta, self.image_dir)
        img_shape = (img.shape[0], img.shape[1])
        img, ce_mask, pad = self.apply_aug_transform(img, ce_mask)

        result = {
            "image_shape": torch.tensor(img_shape),
            "image_name": img_name,
            "pad": str(pad),
            "image": torch.tensor(img).transpose(-1, 0).transpose(-1, -2).float(),
            "mask": torch.tensor(ce_mask, dtype=self.mask_dtype).transpose(-1, 0).transpose(-1, -2)
            if self.train
            else [],
        }
        return result


class NearestSegDataset(SegDataset):
    def __init__(
        self, additional_meta: Optional[str] = None, additional_image_dir: Optional[str] = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        assert self.mode == "multiclass", f"NearestSegDataset requires mode 'multiclass', got mode {self.mode}"
        assert (
            self.num_channels == 1
        ), f"NearestSegDataset requires num_channels 1, got num_channels {self.num_channels}"

        self.additional_meta_df = self.prepare_additional_meta(additional_meta)
        self.additional_image_dir = self.select_additional_image_dir(additional_image_dir)
        self.gt_images_df = self.prepare_gt_images_df(self.additional_meta_df)

    def prepare_additional_meta(self, additional_meta: str) -> pd.DataFrame:
        if self.split in ["train", "val"]:
            additional_meta_df = self.train_meta[np.isin(self.train_meta["ImageId"], self.train_imgs)]
        else:
            additional_meta_df = pd.read_csv(additional_meta).drop_duplicates(["ImageId", "ClassId"])
        return additional_meta_df

    def select_additional_image_dir(self, additional_image_dir: str) -> str:
        if self.split in ["train", "val"]:
            additional_image_dir = self.image_dir
        else:
            additional_image_dir = additional_image_dir

        return additional_image_dir

    def prepare_gt_images_df(self, train_df: pd.DataFrame) -> pd.DataFrame:
        gt_images = train_df[["ImageId"]].drop_duplicates().reset_index(drop=True)
        gt_images["prefix"] = gt_images["ImageId"].map(lambda x: nearest.get_prefix_and_num(x)[0])
        gt_images["num"] = gt_images["ImageId"].map(lambda x: nearest.get_prefix_and_num(x)[1])

        return gt_images

    def construct_channels(self, curr_img: np.ndarray, gt_img: np.ndarray, gt_ce_mask: np.ndarray) -> np.ndarray:
        diff_img = curr_img.astype(np.float) - gt_img.astype(np.float)
        diff_img = ((diff_img - diff_img.min()) / diff_img.ptp() * 255).astype(np.uint8)
        gt_img = ((gt_img - gt_img.min()) / gt_img.ptp() * 255).astype(np.uint8)
        gt_ce_mask = ((gt_ce_mask - gt_ce_mask.min()) / gt_ce_mask.ptp() * 255).astype(np.uint8)
        out_img = np.concatenate((gt_img, diff_img, gt_ce_mask), axis=2)
        return out_img

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        img_name = self.image_names[index]  # ['ImageId']
        curr_img, curr_ce_mask = self.get_gt_image_and_mask(img_name, meta_df=self.train_meta, image_dir=self.image_dir)
        img_shape = (curr_img.shape[0], curr_img.shape[1])

        if self.split == "train":
            min_distance = min(nearest.sample_min_distance(), 20)
        else:
            min_distance = 0
        gt_img_name = nearest.find_nearest_gt(img_name, gt_images=self.gt_images_df, min_distance=min_distance)
        gt_img, gt_ce_mask = self.get_gt_image_and_mask(
            gt_img_name, meta_df=self.additional_meta_df, image_dir=self.additional_image_dir, force_load_mask=True
        )

        gt_img = np.expand_dims(cv2.resize(gt_img, dsize=img_shape[::-1], interpolation=cv2.INTER_CUBIC), axis=2)
        gt_ce_mask = np.expand_dims(
            cv2.resize(gt_ce_mask, dsize=gt_img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST), axis=2
        )
        img = self.construct_channels(curr_img, gt_img, gt_ce_mask)

        img, ce_mask, pad = self.apply_aug_transform(img, curr_ce_mask)

        result = {
            "image_shape": torch.tensor(img_shape),
            "image_name": img_name,
            "gt_image_name": gt_img_name,
            "pad": str(pad),
            "image": torch.tensor(img).transpose(-1, 0).transpose(-1, -2).float(),
            "mask": torch.tensor(ce_mask, dtype=self.mask_dtype).transpose(-1, 0).transpose(-1, -2)
            if self.train
            else [],
        }
        return result
