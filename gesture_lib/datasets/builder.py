# import sys
# sys.path.append(".")
from mmcv.utils import build_from_cfg
from .registry import DATASET


def build_dataset(data_cfg, default_args=None):
    return build_from_cfg(data_cfg, DATASET, default_args)
