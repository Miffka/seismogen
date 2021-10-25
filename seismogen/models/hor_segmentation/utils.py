from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm
from pytorch_toolbelt import losses
from torch.utils.tensorboard import SummaryWriter

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
    predicted_class_idx: int = 0,
    writer: Optional[SummaryWriter] = None,
):
    loss_f, loss_d = define_losses(reduction="mean")

    ds_names = get_all_dataset_names(val_dataloader)
    loss_dict = {ds_name: defaultdict(float) for ds_name in ds_names}
    loss_dict["all"] = defaultdict(float)

    for sample in tqdm.tqdm(val_dataloader, desc="Validate", total=len(val_dataloader)):
        with torch.cuda.amp.autocast(enabled=fp16):
            predict = model(sample["image"].to(torch_config.device))[:, predicted_class_idx].unsqueeze(1)
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
