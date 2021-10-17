import argparse
import json
import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import segyio
import tqdm
from skimage import morphology

from seismogen.config import system_config
from seismogen.data.segy.horizons import read_horizons_from_paths
from seismogen.data.segy.preprocessing import clip_normalize, scale_to_uint8


def save_iline_xline_segy(
    segyvol: segyio.SegyFile,
    savedir: str,
    prefix: str = "volume",
    cut_path_len: int = 0,
) -> Dict[str, List]:
    data = defaultdict(list)
    os.makedirs(savedir, exist_ok=True)

    for orient in ["iline", "xline"]:
        orient_ids = getattr(segyvol, f"{orient}s")
        for idx, orient_idx in tqdm.tqdm(enumerate(orient_ids), total=len(orient_ids), desc=f"{prefix} {orient}"):

            img = getattr(segyvol, orient)[orient_idx]
            img = clip_normalize(img)
            img = scale_to_uint8(img)
            img = img.T
            savepath = osp.join(savedir, f"{orient}_{idx}.png")

            cv2.imwrite(savepath, img)

            data["orient"].append(orient)
            data["axis_idx"].append(idx)
            data["orient_idx"].append(orient_idx)
            data[f"{prefix}_path"].append(savepath[cut_path_len:])

        print(f"Successfully converted {prefix} {orient}")

    return data


def save_iline_xline_array(
    array: np.ndarray,
    savedir: str,
    prefix: str = "volume",
    cut_path_len: int = 0,
    dilate: bool = False,
) -> Dict[str, List]:
    data = defaultdict(list)
    os.makedirs(savedir, exist_ok=True)

    orient = "iline"
    for idx in tqdm.tqdm(range(array.shape[0]), desc=f"{prefix} {orient}"):

        img = array[idx]
        if dilate:
            img = morphology.binary_dilation(img, morphology.disk(5, dtype=bool))
        img = img.astype(np.float16)
        img = scale_to_uint8(img)
        img = img.T
        savepath = osp.join(savedir, f"{orient}_{idx}.png")

        cv2.imwrite(savepath, img)

        data["orient"].append(orient)
        data["axis_idx"].append(idx)
        data[f"{prefix}_path"].append(savepath[cut_path_len:])

    print(f"Successfully converted {prefix} {orient}")

    orient = "xline"
    for idx in tqdm.tqdm(range(array.shape[1]), desc=f"{prefix} {orient}"):

        img = array[:, idx]
        if dilate:
            img = morphology.binary_dilation(img, morphology.disk(5, dtype=bool))
        img = img.astype(np.float16)
        img = scale_to_uint8(img)
        img = img.T
        savepath = osp.join(savedir, f"{orient}_{idx}.png")

        cv2.imwrite(savepath, img)

        data["orient"].append(orient)
        data["axis_idx"].append(idx)
        data[f"{prefix}_path"].append(savepath[cut_path_len:])

    print(f"Successfully converted {prefix} {orient}")

    return data


def convert_ilines_xlines_to_png(
    volume_path: str,
    ds_name: str,
    volume_idx: int = 0,
    mask_path: Optional[str] = None,
    horizons_path: Optional[str] = None,
    data_dir: str = system_config.data_dir,
) -> pd.DataFrame:

    savedir = osp.join(data_dir, "processed", ds_name)
    cut_path_len = len(data_dir) + 1

    with segyio.open(osp.join(data_dir, volume_path)) as segyvol:
        vol_savedir = osp.join(savedir, f"{volume_idx:02d}_volume")
        vol_data = save_iline_xline_segy(segyvol, vol_savedir, prefix="volume", cut_path_len=cut_path_len)

    if mask_path is not None:
        with segyio.open(osp.join(data_dir, mask_path)) as segyvol:
            mask_savedir = osp.join(savedir, f"{volume_idx:02d}_mask")
            mask_data = save_iline_xline_segy(segyvol, mask_savedir, prefix="mask", cut_path_len=cut_path_len)
            vol_data.update(mask_data)

    if horizons_path is not None:
        volume_horizons = read_horizons_from_paths(
            osp.join(data_dir, volume_path), osp.join(data_dir, horizons_path), dilate=False
        )
        horizons_savedir = osp.join(savedir, f"{volume_idx:02d}_horizons")
        hor_data = save_iline_xline_array(
            volume_horizons, horizons_savedir, prefix="horizons", cut_path_len=cut_path_len, dilate=True
        )
        vol_data.update(hor_data)

    df = pd.DataFrame(vol_data)

    df["ds_name"] = ds_name
    df["w_volume_path"] = volume_path
    df["w_mask_path"] = mask_path
    df["w_horizons_path"] = horizons_path

    return df


def convert_all_volumes(
    volume_json: Dict[str, List[Dict[str, str]]], data_dir: str = system_config.data_dir
) -> Dict[str, List[Dict[str, str]]]:

    for ds_name, ds_list in volume_json.items():
        cut_path_len = len(data_dir) + 1
        markup_dir = osp.join(data_dir, "processed", ds_name, "markup")
        os.makedirs(markup_dir, exist_ok=True)

        for vol_idx, vol_dict in enumerate(ds_list):
            volume_path = vol_dict["volume"]
            mask_path = vol_dict.get("mask")
            horizons_path = vol_dict.get("horizons")

            df_vol = convert_ilines_xlines_to_png(volume_path, ds_name, vol_idx, mask_path, horizons_path, data_dir)

            volume_fname = osp.splitext(osp.basename(volume_path))[0]
            markup_path = osp.join(markup_dir, f"{vol_idx:02d}_{volume_fname}.csv")
            df_vol.to_csv(markup_path, index=False)

            volume_json[ds_name][vol_idx]["markup"] = markup_path[cut_path_len:]

    return volume_json


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Convert segy data to png")

    parser.add_argument("--data_dir", type=str, default=system_config.data_dir)
    parser.add_argument("--path_to_json", type=str, default=osp.join("processed", "volume.json"))

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    json_path = osp.join(args.data_dir, args.path_to_json)

    with open(json_path, "r") as fin:
        volume_json = json.load(fin)

    volume_json = convert_all_volumes(volume_json, data_dir=args.data_dir)

    with open(json_path, "w") as fout:
        json.dump(volume_json, fout, indent=2)
