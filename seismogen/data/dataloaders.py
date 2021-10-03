import argparse
import os.path as osp
from typing import Dict

import torch

from seismogen.data.augmentation import get_augmentations
from seismogen.data.dataset import SegDataset
from seismogen.data.transforms import get_transforms


def init_dataloaders(args: argparse.ArgumentParser) -> Dict[str, torch.utils.data.DataLoader]:

    dataloaders_dict = {}

    for split in ["train", "test"]:
        dataset = SegDataset(
            image_dir=osp.join(args.data_dir, "Lab1", "data", f"{split}_images_track{args.track_num}"),
            train_meta=None
            if split == "test"
            else osp.join(args.data_dir, "Lab1", "data", f"{split}_seg_track{args.track_num}.csv"),
            augmentation=None
            if split == "test"
            else get_augmentations(
                resize=args.size, augmentation_intensity=args.augmentation_intensity, gauss_limit=args.gauss_limit
            ),
            transform=get_transforms(size=args.size, num_channels=args.num_channels),
            train=split == "train",
        )

        if split == "test":
            sampler = torch.utils.data.SequentialSampler(dataset)
        else:
            sampler = torch.utils.data.RandomSampler(dataset)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers
        )
        dataloaders_dict[split] = dataloader

    return dataloaders_dict
