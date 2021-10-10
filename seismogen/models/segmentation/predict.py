from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import tqdm
from pytorch_toolbelt.inference import tta  # noqa F401
from scipy.ndimage.morphology import binary_closing, binary_fill_holes
from skimage.measure import label
from skimage.morphology import square

from seismogen.data.letterbox import letterbox_backward
from seismogen.data.rle_utils import out2rle
from seismogen.torch_config import torch_config


def make_ohe_mask(multiclass_mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    ohe_mask = torch.zeros((num_classes, *multiclass_mask.shape), dtype=multiclass_mask.dtype)
    for class_num in range(num_classes):
        ohe_mask[class_num] = multiclass_mask == class_num

    return ohe_mask


def get_filled_mlab(binary_masks: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    new_mlab_mask = []

    if isinstance(binary_masks, torch.Tensor):
        binary_masks = binary_masks.numpy()

    for mask_slice in binary_masks:
        new_mlab_mask.append(binary_fill_holes(binary_closing(mask_slice, structure=square(6))))

    new_mlab_mask = np.stack(new_mlab_mask, axis=0)

    return new_mlab_mask


def get_biggest_component(binary_mask: np.ndarray) -> np.ndarray:
    binary_mask = binary_closing(binary_mask, structure=square(6))
    labels = label(binary_mask)
    largestCC = labels == (np.argmax(np.bincount(labels.flat)[1:]) + 1)
    return largestCC


def get_biggest_component_mlab(binary_masks: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    new_mlab_mask = []

    if isinstance(binary_masks, torch.Tensor):
        binary_masks = binary_masks.numpy()

    for class_idx, mask_slice in enumerate(binary_masks):
        if mask_slice.sum() < 1500:
            new_mlab_mask.append(mask_slice)
            continue
        largest_cc = get_biggest_component(mask_slice)
        new_mlab_mask.append(largest_cc)
    new_mlab_mask = np.stack(new_mlab_mask, axis=0)

    return new_mlab_mask


@torch.no_grad()
def get_prediction(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    fp16: bool = False,
    tta_type: Optional[str] = None,
    fill_holes: bool = False,
    biggest_only: bool = False,
) -> pd.DataFrame:

    num_classes = model.segmentation_head[0].out_channels
    multiclass = num_classes == 8
    all_predicts = []

    for batch_idx, batch in tqdm.tqdm(enumerate(test_loader), desc="Predict", total=len(test_loader)):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=fp16):
                if tta_type is None:
                    predict = model(batch["image"].to(torch_config.device).detach())
                else:
                    predict = eval(f"tta.{tta_type}")(model, batch["image"].to(torch_config.device))

        predict_original_size = []
        for k, _pred in enumerate(predict):
            if batch["pad"][k] != "None":
                predict_original_size.append(letterbox_backward(_pred.cpu().numpy(), eval(batch["pad"][k])))
            else:
                predict_original_size.append(
                    torch.nn.functional.interpolate(_pred.float().unsqueeze(0).cpu(), batch["image_shape"][k].tolist())[
                        0
                    ]
                )

        if multiclass:
            new_pred = []
            for pred_ in predict_original_size:
                pred_ = torch.max(pred_, axis=0)[1]
                pred_ = make_ohe_mask(pred_, num_classes)
                new_pred.append(pred_)
            predict_original_size = new_pred

        if fill_holes:
            new_pred = []
            for pred_ in predict_original_size:
                new_pred.append(get_filled_mlab(pred_))
            predict_original_size = new_pred

        if biggest_only:
            new_pred = []
            for pred_ in predict_original_size:
                new_mlab_mask = get_biggest_component_mlab(pred_)
                new_pred.append(new_mlab_mask)
            predict_original_size = new_pred

        predict2str = out2rle(predict_original_size)
        for i, sample in enumerate(predict):
            range_start = i * num_classes + int(multiclass)
            range_end = (i + 1) * num_classes
            for j in range(range_start, range_end):
                # each predicton consists of 8 classes, we iterate over them and add to the list
                all_predicts.append(
                    {"EncodedPixels": predict2str[j], "ImageId": batch["image_name"][i], "ClassId": j % 7}
                )

    predict = pd.DataFrame(all_predicts)

    return predict
