from typing import Tuple

import numpy as np
import pandas as pd


def get_prefix_and_num(image_id: str) -> Tuple[str, int]:
    prefix, num_part = image_id.split("_")
    return prefix, int(num_part.split(".")[0])


def find_nearest_gt(image_id: str, gt_images: pd.DataFrame) -> str:
    id_prefix, id_num_part = get_prefix_and_num(image_id)
    possible_prefixes = gt_images[gt_images["prefix"] == id_prefix]
    possible_prefixes["idx_distance"] = (possible_prefixes["num"] - id_num_part).map(np.abs)

    return possible_prefixes.sort_values("idx_distance").head(1).to_dict()
