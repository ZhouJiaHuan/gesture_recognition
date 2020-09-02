from gesture_lib.registry import DATASET
from gesture_lib.ops.registry import build_from_cfg


def build_dataset(data_cfg, default_args=None):
    return build_from_cfg(data_cfg, DATASET, default_args)
