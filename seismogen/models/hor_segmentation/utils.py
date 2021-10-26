import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import tqdm
from pytorch_toolbelt import losses
from torch.utils.tensorboard import SummaryWriter

from seismogen.data.segy.transforms import backward_transform
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


def _add_masks(image: np.ndarray, mask: Union[np.ndarray, torch.Tensor], color: tuple) -> np.ndarray:

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = mask.round().astype(np.uint8)
    image[mask > 0] = image[mask > 0] * 0.7 + np.asarray(color) * 0.3
    return image


@torch.no_grad()
def visualize_masks(
    model: torch.nn.Module,
    input_data: Union[torch.utils.data.DataLoader, torch.Tensor],
    epoch_num: int = 0,
    header: str = "val",
    shown_class_idx: int = 0,
    writer: Optional[SummaryWriter] = None,
):
    if isinstance(input_data, torch.utils.data.DataLoader):
        imgs = []
        gt_masks = []
        sample_names = []

        for dataset in input_data.dataset.datasets:
            sample_idx = len(dataset) // 2
            sample = dataset[sample_idx]
            imgs.append(sample["image"])
            gt_masks.append(sample["target"].numpy()[0])
            sample_names.append(f"{dataset.dataset_name}_{sample_idx}")

        imgs = torch.stack(imgs)

    elif isinstance(input_data, torch.Tensor):
        assert input_data.ndim == 4
        imgs = input_data
        sample_names = [f"image_{idx}" for idx in range(imgs.shape[0])]
        gt_masks = None

    else:
        raise NotImplementedError("Only DataLoader and torch.Tensor input types are supported now")

    output_masks = torch.sigmoid(model(imgs.to(torch_config.device))[:, shown_class_idx]).cpu().numpy()
    imgs = backward_transform(imgs).numpy().astype(np.uint8)[:, 0]

    pred_color = (0, 0, 255)
    gt_color = (255, 0, 0)

    for idx, (img, output_mask, sample_name) in enumerate(zip(imgs, output_masks, sample_names)):
        img = np.expand_dims(img, 2)
        img = np.repeat(img, 3, axis=2)
        img = _add_masks(img, output_mask, color=pred_color)
        cv2.putText(img, "pred", (20, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, pred_color, 2)

        if gt_masks is not None:
            img = _add_masks(img, gt_masks[idx], color=gt_color)
            cv2.putText(img, "gt", (20, 45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, gt_color, 2)

        if writer is not None:
            writer.add_image(
                f"{header}/{sample_name}",
                img,
                global_step=epoch_num,
                dataformats="HWC",
            )


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
    model.eval()
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

        visualize_masks(
            model,
            val_dataloader,
            epoch_num=epoch_num,
            header=postfix,
            shown_class_idx=predicted_class_idx,
            writer=writer,
        )

    time.sleep(5)

    return loss_dict["all"]
