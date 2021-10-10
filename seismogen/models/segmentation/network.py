import argparse

import segmentation_models_pytorch as smp  # noqa F401
import torch


def load_net(args: argparse.Namespace) -> torch.nn.Module:

    model = eval(f"smp.{args.seg_model_arch}")(
        encoder_name=args.backbone,
        encoder_weights=args.pretrained_weights,
        in_channels=args.num_channels,
        classes=args.num_classes,
    )
    state = {}

    if args.weights is not None:
        state = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        del state["state_dict"]

    return model, state
