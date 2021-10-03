import os
import os.path as osp
from typing import Dict, Optional

import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from seismogen.config import system_config
from seismogen.data.dataloaders import init_dataloaders
from seismogen.models.fix_seeds import fix_seeds
from seismogen.models.parser import get_parser
from seismogen.models.segmentation.loss import DiceLoss
from seismogen.models.segmentation.network import load_net
from seismogen.models.segmentation.predict import get_prediction
from seismogen.models.train_utils import define_optimizer, define_scheduler
from seismogen.torch_config import torch_config


@torch.no_grad()
def eval_model(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    epoch_num: int,
    writer: Optional[SummaryWriter] = None,
):
    total_samples = len(val_dataloader)
    loss_f = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(torch_config.device).reshape(1, 7, 1, 1)
    )
    loss_d = DiceLoss(torch_config.device)

    loss_f_value = 0
    loss_d_value = 0

    for sample in tqdm.tqdm(val_dataloader, desc="Validate", total=total_samples):
        predict = model(sample["image"].to(torch_config.device))

        loss_f_value += loss_f(predict, sample["mask"].to(torch_config.device))
        loss_d_value += loss_d(predict, sample["mask"].to(torch_config.device)).mean()

    loss = loss_f_value + loss_d_value

    loss_dict = {
        "Loss_val": loss.item() / total_samples,
        "Loss BCE_val": loss_f_value.item() / total_samples,
        "Loss DICE_val": loss_d_value.item() / total_samples,
    }

    if writer is not None:
        for metric_name, metric_value in loss_dict.items():
            writer.add_scalar(metric_name, round(metric_value, 3), global_step=epoch_num)

    return loss_dict


def train_one_epoch(
    model: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    optimizer: torch.optim.Optimizer,
    epoch_num: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    writer: Optional[SummaryWriter] = None,
) -> torch.nn.Module:

    total_samples = len(dataloaders["train"])
    loss_f = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(torch_config.device).reshape(1, 7, 1, 1)
    )
    loss_d = DiceLoss(torch_config.device)

    progress_bar = tqdm.tqdm(enumerate(dataloaders["train"]), desc=f"Train epoch {epoch_num}", total=total_samples)

    for i, sample in progress_bar:
        optimizer.zero_grad()
        predict = model(sample["image"].to(torch_config.device))

        loss_f_value = loss_f(predict, sample["mask"].to(torch_config.device))
        loss_d_value = loss_d(predict, sample["mask"].to(torch_config.device)).mean()
        loss = loss_f_value + loss_d_value
        loss.backward()

        optimizer.step()

        progress_bar.set_description(f"Epoch: {epoch_num} Loss: {round(loss.item(), 3)}")
        if writer is not None:
            global_step = i + total_samples * epoch_num
            writer.add_scalar("Loss", round(loss.item(), 3), global_step=global_step)
            writer.add_scalar("Loss BCE", round(loss_f_value.item(), 3), global_step=global_step)
            writer.add_scalar("Loss DICE", round(loss_d_value.item(), 3), global_step=global_step)

    val_loss_dict = eval_model(model, dataloaders["val"], epoch_num, writer=writer)

    if scheduler is not None:
        scheduler.step(val_loss_dict["Loss DICE_val"])

    return val_loss_dict["Loss DICE_val"]


def train_model():
    parser = get_parser()
    args = parser.parse_args()

    writer = SummaryWriter(osp.join(system_config.log_dir + str(args.track_num), args.task_name))

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

    fix_seeds(args.random_state)
    for epoch_num in range(init_epoch, init_epoch + args.epochs):
        current_metric = train_one_epoch(model, dataloaders, optimizer, epoch_num, writer=writer)
        state = {
            "state_dict": model.state_dict(),
            "epoch": epoch_num,
            "best_metric": best_metric,
        }
        torch.save(state, osp.join(save_dir, f"epoch_{epoch_num}.pth"))

        if current_metric < best_metric:
            best_metric = current_metric
            torch.save(state, osp.join(save_dir, "best.pth"))

    state = torch.load(osp.join(save_dir, "best.pth"))
    model.load_state_dict(state["state_dict"])

    prediction = get_prediction(model, dataloaders["test"])
    prediction.to_csv(osp.join(save_dir, f"{args.task_name}.csv"))


if __name__ == "__main__":
    train_model()
