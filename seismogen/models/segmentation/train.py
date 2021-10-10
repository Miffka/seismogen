import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import torch
import tqdm
from pytorch_toolbelt import losses
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


def define_losses(
    num_classes: int, use_focal: bool = False, loss_weights: Optional[List[float]] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
) -> Tuple[torch.nn.Module]:

    assert num_classes in {7, 8}
    if num_classes == 7:
        loss_f = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(loss_weights).to(torch_config.device).reshape(1, num_classes, 1, 1)
        )
        loss_d = DiceLoss(torch_config.device)
    else:
        if use_focal:
            loss_f = losses.FocalLoss()
        else:
            loss_f = torch.nn.CrossEntropyLoss()

        loss_d = DiceLoss(torch_config.device)

    return loss_f, loss_d


@torch.no_grad()
def eval_model(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    epoch_num: int,
    loss_weights: Optional[List[float]] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    use_focal: bool = False,
    fp16: bool = False,
    writer: Optional[SummaryWriter] = None,
):
    num_classes = model.segmentation_head[0].out_channels
    loss_f, loss_d = define_losses(num_classes, use_focal=use_focal, loss_weights=loss_weights)

    loss_f_value = 0
    loss_d_value = 0

    total_samples = len(val_dataloader)
    for sample in tqdm.tqdm(val_dataloader, desc="Validate", total=total_samples):

        with torch.cuda.amp.autocast(enabled=fp16):
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
    loss_weights: Optional[List[float]] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    use_focal: bool = False,
    fp16_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    writer: Optional[SummaryWriter] = None,
) -> torch.nn.Module:

    num_classes = model.segmentation_head[0].out_channels
    loss_f, loss_d = define_losses(num_classes, use_focal=use_focal, loss_weights=loss_weights)

    total_samples = len(dataloaders["train"])
    progress_bar = tqdm.tqdm(enumerate(dataloaders["train"]), desc=f"Train epoch {epoch_num}", total=total_samples)

    for i, sample in progress_bar:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=fp16_scaler is not None):
            predict = model(sample["image"].to(torch_config.device))

            loss_f_value = loss_f(predict, sample["mask"].to(torch_config.device))
            loss_d_value = loss_d(predict, sample["mask"].to(torch_config.device)).mean()
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
        loss_weights=loss_weights,
        use_focal=use_focal,
        fp16=fp16_scaler is not None,
        writer=writer,
    )

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

    if args.loss_weights is None:
        args.loss_weights = [1.0] * args.num_classes
    assert len(args.loss_weights) == args.num_classes

    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    else:
        fp16_scaler = None

    fix_seeds(args.random_state)
    for epoch_num in range(init_epoch, init_epoch + args.epochs):
        current_metric = train_one_epoch(
            model,
            dataloaders,
            optimizer,
            epoch_num,
            loss_weights=args.loss_weights,
            use_focal=args.use_focal,
            fp16_scaler=fp16_scaler,
            writer=writer,
        )
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

    prediction = get_prediction(
        model,
        dataloaders["test"],
        fp16=args.fp16,
        tta_type=args.tta_type,
        fill_holes=args.fill_holes,
        biggest_only=args.biggest_only,
    )
    prediction.to_csv(osp.join(save_dir, f"{args.task_name}.csv"))


if __name__ == "__main__":
    train_model()
