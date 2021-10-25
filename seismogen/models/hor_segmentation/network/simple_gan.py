import argparse
from typing import Dict, Tuple

import numpy as np
import torch
from mobile_stylegan.models.mapping_network import MappingNetwork
from mobile_stylegan.models.mobile_synthesis_network import MobileSynthesisNetwork
from torch import nn


class MobileStyleGAN(nn.Module):
    orig_channels = [512, 512, 512, 512, 512, 256, 128, 64]
    orig_size = 1024

    def __init__(self, style_dim, output_size):
        super().__init__()
        self.style_dim = style_dim
        self.output_size = output_size

        assert (output_size <= 1024) and (np.log2(output_size) % 1 == 0)
        self.num_c_skip = np.log2(self.orig_size / output_size).astype(int)
        self.channels = self.orig_channels[self.num_c_skip :]

        self.mapping_network = MappingNetwork(self.style_dim, n_layers=len(self.channels))
        self.generator = MobileSynthesisNetwork(style_dim=self.style_dim, channels=self.channels)

    def forward(self, var: torch.Tensor) -> torch.Tensor:
        style = self.mapping_network(var)
        img = self.generator(style)["img"].mean(1).unsqueeze(1)

        return img


def load_net(args: argparse.Namespace) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    model = MobileStyleGAN(style_dim=args.style_dim, output_size=args.size)
    state = {}

    if args.weights is not None:
        state = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(state["state_dict_gen"])
        del state["state_dict_gen"], state["state_dict"]

    return model, state
