import argparse
from typing import Dict

import torch


def define_optimizer(
    args: argparse.Namespace,
    net: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    lr_multiplier: float = 1.0,
    new_optimizer: bool = False,
    postfix: str = "",
) -> torch.optim.Optimizer:
    params = [p for p in net.parameters() if p.requires_grad]
    if args.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr * lr_multiplier, weight_decay=args.weight_decay)
    elif args.optimizer_type == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr * lr_multiplier, weight_decay=args.weight_decay)
    elif args.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            params, lr=args.lr * lr_multiplier, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer is not implemented for type {args.optimizer_type}")

    if "optimizer" in state.keys() and not new_optimizer:
        try:
            optimizer.load_state_dict(state[f"optimizer_state{postfix}"])
        except Exception as e:
            print(
                f"\nOptimizer state has not been loaded: {e}. Continuing training with the given state dict ignored\n"
            )
    return optimizer


def define_scheduler(
    args: argparse.Namespace, optimizer: torch.optim.Optimizer, state: Dict[str, torch.Tensor], postfix: str = ""
) -> torch.optim.lr_scheduler._LRScheduler:
    lr_scheduler = args.lr_scheduler

    if lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=args.patience, factor=args.gamma_factor, verbose=True
        )
    elif lr_scheduler is None:
        pass
    else:
        raise NotImplementedError()

    if (lr_scheduler is not None) and ("scheduler" in state):
        try:
            lr_scheduler.load_state_dict(state[f"scheduler_state{postfix}"])
        except Exception as e:
            print(
                f"\nScheduler state has not been loaded: {e}. Continuing training with the given state dict ignored\n"
            )
    return lr_scheduler
