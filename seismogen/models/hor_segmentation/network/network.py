import argparse
from typing import Dict, Tuple

import torch

from seismogen.models.hor_segmentation.network.simple_gan import (
    load_net as load_net_generator,
)
from seismogen.models.segmentation.network import load_net as load_net_baseline


def load_net(args: argparse.Namespace) -> Tuple[torch.nn.Module, Dict]:

    if not args.enable_gan:
        model, state = load_net_baseline(args)
        return model, state
    else:
        if not args.fake_imgs_classify:
            args.num_classes += 1
        disc, _ = load_net_baseline(args)
        gen, state = load_net_generator(args)

        return (gen, disc), state
