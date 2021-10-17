import os.path as osp
from collections import namedtuple
from typing import Tuple

import numpy as np
import segyio
from skimage import morphology

from seismogen.config import system_config

VolumeMeta = namedtuple("VolumeMeta", ["shape", "starting_inline", "starting_xline", "z_step"])


def read_horizons_meta(fpath: str) -> VolumeMeta:

    with segyio.open(fpath) as segyfile:
        shape = (len(segyfile.ilines), len(segyfile.xlines), segyfile.samples.size)
        starting_inline = segyfile.ilines[0]
        starting_xlines = segyfile.xlines[0]
        z_step = segyio.tools.dt(segyfile) / 1000

    output = VolumeMeta(shape=shape, starting_inline=starting_inline, starting_xline=starting_xlines, z_step=z_step)

    return output


def read_horizons(
    fpath: str, shape: Tuple[int], starting_inline: int, starting_xline: int, z_step: int, dilate: bool = False
) -> np.ndarray:
    # horizons_dat = [i.strip().split() for i in open(fpath).readlines()]
    # horizons_dat = [
    #     [int(i[1]) - starting_inline, int(i[2])- starting_xline, round(float(i[3])/z_step)] for i in horizons_dat if not (i[1] == '"Inline"' or i[1]=='-')
    # ]
    horizons_dat = []
    with open(fpath) as fin:
        for line in fin:
            cols = line.strip().split("\t")
            if len(cols) > 1 and "line" not in cols[1] and cols[1] != "-":
                horizons_dat.append(
                    [int(cols[1]) - starting_inline, int(cols[2]) - starting_xline, round(float(cols[3]) / z_step)]
                )

    horizons = np.zeros(shape, dtype=bool)
    for h in horizons_dat:
        horizons[h[0], h[1], h[2]] = True

    if dilate:
        horizons = morphology.binary_dilation(horizons, morphology.ball(5, dtype=np.bool))

    return horizons


def read_horizons_from_paths(
    volume_path: str, horizons_path: str, data_dir: str = system_config.data_dir, dilate: bool = False
) -> np.ndarray:

    volume_meta = read_horizons_meta(osp.join(data_dir, volume_path))
    volume_hor = read_horizons(osp.join(data_dir, horizons_path), dilate=dilate, **volume_meta._asdict())

    return volume_hor
