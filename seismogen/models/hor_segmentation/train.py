import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm
from pytorch_toolbelt import losses
from torch.utils.tensorboard import SummaryWriter

from seismogen.config import system_config
from seismogen.data.segy.dataloaders import init_dataloaders
from seismogen.models.fix_seeds import fix_seeds
from seismogen.models.hor_segmentation.network.network import load_net
from seismogen.models.hor_segmentation.parser import get_parser
from seismogen.models.train_utils import define_optimizer, define_scheduler
from seismogen.torch_config import torch_config


def define_losses(reduction: str = "mean") -> Tuple[torch.nn.Module]:
    loss_f = torch.nn.BCEWithLogitsLoss(reduction=reduction)
    loss_d = losses.DiceLoss(mode=losses.BINARY_MODE, ignore_index=0)

    return loss_f, loss_d


def get_all_dataset_names(dataloader: torch.utils.data.DataLoader) -> List[str]:
    ds_names = []
    for dataset in dataloader.dataset.datasets:
        ds_names.append(dataset.dataset_name)

    return ds_names


def reduce_loss_dict(loss_dict: Dict[str, Dict[str, float]], postfix: str = "val") -> Dict[str, Dict[str, float]]:
    loss_dict_out = {}

    for ds_name, ds_loss_dict in loss_dict.items():
        ds_dict = {}
        for loss_name in ["Loss BCE", "Loss DICE"]:
            ds_dict[f"{loss_name}_{postfix}"] = ds_loss_dict[loss_name] / ds_loss_dict["num_samples"]

        loss_dict_out[ds_name] = ds_dict

    return loss_dict_out


@torch.no_grad()
def eval_model(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    epoch_num: int,
    fp16: bool = False,
    postfix: str = "val",
    writer: Optional[SummaryWriter] = None,
):
    loss_f, loss_d = define_losses(reduction="mean")

    ds_names = get_all_dataset_names(val_dataloader)
    loss_dict = {ds_name: defaultdict(float) for ds_name in ds_names}
    loss_dict["all"] = defaultdict(float)

    for sample in tqdm.tqdm(val_dataloader, desc="Validate", total=len(val_dataloader)):
        with torch.cuda.amp.autocast(enabled=fp16):
            predict = model(sample["image"].to(torch_config.device))
            ds_names_sample = np.asarray(sample["dataset_name"])

            for ds_name in set(ds_names_sample):
                ds_ids = ds_names_sample == ds_name

                loss_f_value = loss_f(predict[ds_ids], sample["target"][ds_ids].to(torch_config.device)) * ds_ids.sum()
                loss_d_value = loss_d(predict[ds_ids], sample["target"][ds_ids].to(torch_config.device)) * ds_ids.sum()

                for ds_idx_name in [ds_name, "all"]:
                    loss_dict[ds_idx_name]["Loss BCE"] += loss_f_value.item()
                    loss_dict[ds_idx_name]["Loss DICE"] += loss_d_value.item()
                    loss_dict[ds_idx_name]["num_samples"] += ds_ids.sum()

    # Reduce loss dict
    loss_dict = reduce_loss_dict(loss_dict, postfix=postfix)

    if writer is not None:
        for ds_name, ds_loss_dict in loss_dict.items():
            for metric_name, metric_value in ds_loss_dict.items():
                writer.add_scalar(f"{metric_name}/{ds_name}", round(metric_value, 3), global_step=epoch_num)

    return loss_dict["all"]


def train_one_epoch(
    model: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    optimizer: torch.optim.Optimizer,
    epoch_num: int,
    fp16_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    writer: Optional[SummaryWriter] = None,
) -> torch.nn.Module:

    loss_f, loss_d = define_losses(reduction="mean")

    total_samples = len(dataloaders["train"])
    progress_bar = tqdm.tqdm(enumerate(dataloaders["train"]), desc=f"Train epoch {epoch_num}", total=total_samples)

    for i, sample in progress_bar:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=fp16_scaler is not None):
            predict = model(sample["image"].to(torch_config.device))

            loss_f_value = loss_f(predict, sample["target"].to(torch_config.device))
            loss_d_value = loss_d(predict, sample["target"][:, 0].to(torch_config.device)).mean()
            loss = loss_f_value + loss_d_value

        if fp16_scaler is not None:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        progress_bar.set_description(f"Epoch: {epoch_num} Loss: {round(loss.item(), 3)}")
        if writer is not None:
            global_step = i + total_samples * epoch_num
            writer.add_scalar("Loss", round(loss.item(), 3), global_step=global_step)
            writer.add_scalar("Loss BCE", round(loss_f_value.item(), 3), global_step=global_step)
            writer.add_scalar("Loss DICE", round(loss_d_value.item(), 3), global_step=global_step)

    val_loss_dict = eval_model(
        model,
        dataloaders["val"],
        epoch_num,
        fp16=fp16_scaler is not None,
        postfix="val",
        writer=writer,
    )

    _ = eval_model(
        model,
        dataloaders["test"],
        epoch_num,
        fp16=fp16_scaler is not None,
        postfix="test",
        writer=writer,
    )

    if scheduler is not None:
        scheduler.step(val_loss_dict["Loss DICE_val"])

    return val_loss_dict["Loss DICE_val"]


def train_model():
    parser = get_parser()
    args = parser.parse_args()

    writer = SummaryWriter(
        osp.join(system_config.log_dir + "_" + args.target_type, f"add_gan {args.enable_gan}", args.task_name)
    )

    dataloaders = init_dataloaders(args)

    fix_seeds(args.random_state)
    model, state = load_net(args)

    model.to(torch_config.device)

    optimizer = define_optimizer(args, model, state)
    scheduler = define_scheduler(args, optimizer, state)  # noqa F841

    init_epoch = state.get("epoch", 0)
    best_metric = state.get("best_metric", 100)

    save_dir = osp.join(system_config.model_dir, args.task_name)
    os.makedirs(save_dir, exist_ok=True)

    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    else:
        fp16_scaler = None

    if args.evaluate_before_training:
        _ = eval_model(
            model,
            dataloaders["val"],
            init_epoch - 1,
            fp16=fp16_scaler is not None,
            postfix="val",
            writer=writer,
        )
        _ = eval_model(
            model,
            dataloaders["test"],
            init_epoch - 1,
            fp16=fp16_scaler is not None,
            postfix="test",
            writer=writer,
        )

    fix_seeds(args.random_state)
    for epoch_num in tqdm.tqdm(range(init_epoch, init_epoch + args.epochs), desc="Epochs"):
        current_metric = train_one_epoch(
            model,
            dataloaders,
            optimizer,
            epoch_num,
            fp16_scaler=fp16_scaler,
            writer=writer,
        )
        state = {
            "state_dict": model.state_dict(),
            "epoch": epoch_num,
            "best_metric": current_metric,
        }
        torch.save(state, osp.join(save_dir, "last.pth"))

        if current_metric < best_metric:
            best_metric = current_metric
            torch.save(state, osp.join(save_dir, "best.pth"))


if __name__ == "__main__":
    train_model()
