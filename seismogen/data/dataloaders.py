import argparse
import os.path as osp
from typing import Dict

import torch

from seismogen.data.augmentation import get_augmentations
from seismogen.data.dataset import NearestSegDataset, SegDataset
from seismogen.data.sampler import ImbalancedDatasetSampler, UndersampledDatasetSampler
from seismogen.data.transforms import get_transforms


def init_dataloaders(args: argparse.ArgumentParser) -> Dict[str, torch.utils.data.DataLoader]:

    dataloaders_dict = {}

    for split in ["train", "val", "test"]:
        if split in ["train", "test"]:
            dir_id = split
        else:
            dir_id = "train"

        mask_mode = "multiclass" if args.num_classes == 8 else "multilabel"
        split_datasets = []
        for track_num in range(1, args.track_num + 1):
            train_meta = (
                None
                if split == "test"
                else osp.join(args.data_dir, "Lab1", "data", f"{dir_id}_seg_track{track_num}.csv")
            )
            augmentation = (
                None
                if split == "test"
                else get_augmentations(
                    resize=args.size, augmentation_intensity=args.augmentation_intensity, gauss_limit=args.gauss_limit
                )
            )
            num_channels = 1 if args.nearest else args.num_channels

            kwargs = dict(
                image_dir=osp.join(args.data_dir, "Lab1", "data", f"{dir_id}_images_track{track_num}"),
                train_meta=train_meta,
                num_channels=num_channels,
                size=args.size,
                mode=mask_mode,
                letterbox=args.letterbox,
                augmentation=augmentation,
                transform=get_transforms(size=args.size, num_channels=args.num_channels, nearest=args.nearest),
                split=split,
                val_size=args.val_size,
            )

            if args.nearest:
                ds_class = NearestSegDataset
                kwargs.update(
                    dict(
                        additional_meta=osp.join(args.data_dir, "Lab1", "data", f"train_seg_track{track_num}.csv"),
                        additional_image_dir=osp.join(args.data_dir, "Lab1", "data", f"train_images_track{track_num}"),
                    )
                )
            else:
                ds_class = SegDataset

            dataset = ds_class(**kwargs)

            if args.track_num == 2 and track_num < args.track_num and split != "train":
                continue

            split_datasets.append(dataset)

        dataset = torch.utils.data.ConcatDataset(split_datasets)

        if split != "train":
            sampler = torch.utils.data.SequentialSampler(dataset)
        elif args.sample_type == "random":
            sampler = torch.utils.data.RandomSampler(dataset)
        elif args.sample_type == "undersampled":
            sampler = UndersampledDatasetSampler(dataset, num_samples=args.num_samples)
        elif args.sample_type == "imbalanced":
            sampler = ImbalancedDatasetSampler(dataset, args.balancing_coeff)
        else:
            raise NotImplementedError

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers
        )
        dataloaders_dict[split] = dataloader

    return dataloaders_dict
