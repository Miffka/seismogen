import os
import os.path as osp
from typing import Dict, Optional

import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from seismogen.config import system_config
from seismogen.data.segy.dataloaders import init_dataloaders
from seismogen.models.fix_seeds import fix_seeds
from seismogen.models.hor_segmentation.network.network import load_net
from seismogen.models.hor_segmentation.parser import get_parser
from seismogen.models.hor_segmentation.utils import define_losses, eval_model
from seismogen.models.train_utils import define_optimizer, define_scheduler
from seismogen.torch_config import torch_config


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

            target_types = np.asarray(sample["target_type"])
            valid_ids = target_types != 0
            if valid_ids.sum() == 0:
                continue

            loss_f_value = loss_f(predict[valid_ids], sample["target"][valid_ids].to(torch_config.device))
            loss_d_value = loss_d(predict[valid_ids], sample["target"][valid_ids].to(torch_config.device)).mean()
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
