import os
import torch


class TorchConfig(object):
    if os.getenv("FORCE_CPU") == "1":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch_config = TorchConfig()
