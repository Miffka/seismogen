import argparse

import segmentation_models_pytorch as smp  # noqa F401
import torch


def load_net(args: argparse.Namespace) -> torch.nn.Module:

    model = eval(f"smp.{args.seg_model_arch}")(
        encoder_name=args.backbone,
        encoder_weights=args.pretrained_weights,
        in_channels=args.num_channels,
        classes=7,
    )

    return model
