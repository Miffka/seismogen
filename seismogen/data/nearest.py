from typing import Tuple

import numpy as np
import pandas as pd
from scipy import interpolate

HIST = np.asarray(
    [
        0.12441077,
        0.08585859,
        0.0523569,
        0.02659933,
        0.02575758,
        0.01835017,
        0.01195286,
        0.00572391,
        0.00538721,
        0.00218855,
        0.00185185,
        0.0013468,
        0.00084175,
        0.0003367,
        0.0003367,
        0.0003367,
    ]
)
BIN_EDGES = np.asarray(
    [1.0, 3.75, 6.5, 9.25, 12.0, 14.75, 17.5, 20.25, 23.0, 25.75, 28.5, 31.25, 34.0, 36.75, 39.5, 42.25, 45.0]
)


def get_prefix_and_num(image_id: str) -> Tuple[str, int]:
    prefix, num_part = image_id.split("_")
    return prefix, int(num_part.split(".")[0])


def find_nearest_gt(image_id: str, gt_images: pd.DataFrame, min_distance: int = 0) -> str:
    id_prefix, id_num_part = get_prefix_and_num(image_id)
    candidates = gt_images[gt_images["prefix"] == id_prefix].copy()
    candidates["idx_distance"] = np.abs(candidates["num"].values - id_num_part)
    candidates = candidates[candidates["idx_distance"] > min_distance]

    return candidates.sort_values("idx_distance").head(1)["ImageId"].values[0]


def sample_min_distance(hist: np.ndarray = HIST, bin_edges: np.ndarray = BIN_EDGES, n_samples: int = 1) -> np.ndarray:
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist * np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = np.random.rand(n_samples)
    r = inv_cdf(r).round(0).astype(int)
    if n_samples == 1:
        r = r[0]
    return r
