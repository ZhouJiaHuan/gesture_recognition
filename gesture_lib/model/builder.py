# import sys
# sys.path.append(".")
from mmcv.utils import build_from_cfg
from .registry import MODELS


def build_model(model_cfg, default_args=None):
    return build_from_cfg(model_cfg, MODELS, default_args)
