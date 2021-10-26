import argparse
from typing import Dict, Tuple

import segmentation_models_pytorch as smp  # noqa F401
import torch
import torch.nn as nn


def replace_bns(model: nn.Module, NewNorm: nn.Module, num_groups: int = 32) -> None:
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_bns(module, NewNorm, num_groups=num_groups)

        if isinstance(module, nn.BatchNorm2d):
            ## simple module
            num_channels = module.num_features
            new_bn = NewNorm(num_groups, num_channels)
            setattr(model, name, new_bn)


def load_net(args: argparse.Namespace) -> Tuple[nn.Module, Dict]:

    if args.fake_imgs_classify:
        aux_params = dict(
            pooling="avg",
            dropout=0.5,
            activation=None,
            classes=1,
        )
    else:
        aux_params = None

    model = eval(f"smp.{args.seg_model_arch}")(
        encoder_name=args.backbone,
        encoder_weights=args.pretrained_weights,
        in_channels=args.num_channels,
        classes=args.num_classes,
        aux_params=aux_params,
    )
    state = {}
    if args.norm_layer == "GroupNorm":
        replace_bns(model, nn.GroupNorm, num_groups=args.gn_num_groups)

    if args.weights is not None:
        state = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        del state["state_dict"]

    return model, state
