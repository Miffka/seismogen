import os.path as osp

CURRENT_PATH = osp.dirname(osp.realpath(__file__))


class SystemConfig(object):
    root_dir = osp.join(CURRENT_PATH, "..")
    data_dir = osp.realpath(osp.join(CURRENT_PATH, "..", "data"))
    model_dir = osp.realpath(osp.join(CURRENT_PATH, "..", "models"))
    log_dir = osp.realpath(osp.join(CURRENT_PATH, "..", "runs"))


system_config = SystemConfig()
