import argparse
from typing import Dict, Tuple

import torch

from seismogen.models.segmentation.network import load_net as load_net_baseline


def load_net(args: argparse.Namespace) -> Tuple[torch.nn.Module, Dict]:

    if not args.enable_gan:
        model, state = load_net_baseline(args)
    else:
        raise NotImplementedError("Task types other than simple 'segment' are not implemented yet")

    return model, state
