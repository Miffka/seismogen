import argparse
import json
import os.path as osp
from typing import Dict

import torch

from seismogen.data.augmentation import get_augmentations
from seismogen.data.segy.dataset import SEGYDataset
from seismogen.data.segy.transforms import get_transforms


def init_dataloaders(args: argparse.ArgumentParser) -> Dict[str, torch.utils.data.DataLoader]:

    dataloaders_dict = {}
    dataset_split_dict = {"train": args.train_datasets, "test": args.test_datasets, "view": args.view_datasets}

    json_fpath = osp.join(args.data_dir, args.json_path)
    with open(json_fpath) as fin:
        volume_json = json.load(fin)

    for split in ["train", "val", "test"]:
        split_datasets = []

        for main_split, main_split_datasets in dataset_split_dict.items():
            if main_split == "test" and split != "test":
                continue

            if main_split == "view" and split != "train":
                continue

            for dataset_name in main_split_datasets:

                markup_fpath = osp.join(args.data_dir, osp.join(volume_json[dataset_name][0]["markup"]))
                target_type = None if main_split == "view" else args.target_type
                augmentation = (
                    None
                    if split != "train"
                    else get_augmentations(resize=args.size, augmentation_intensity=args.augmentation_intensity)
                )
                transform = get_transforms(num_channels=1)

                kwargs = dict(
                    markup_path=markup_fpath,
                    data_dir=args.data_dir,
                    target_type=target_type,
                    test_size=args.test_size,
                    split=split,
                    random_state=args.random_state,
                    size=args.size,
                    letterbox=args.letterbox,
                    augmentation=augmentation,
                    transform=transform,
                )

                dataset = SEGYDataset(**kwargs)

                split_datasets.append(dataset)

        dataset = torch.utils.data.ConcatDataset(split_datasets)

        if split != "train":
            sampler = torch.utils.data.SequentialSampler(dataset)
        elif args.sample_type == "random":
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            raise NotImplementedError

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers
        )
        dataloaders_dict[split] = dataloader

    return dataloaders_dict
