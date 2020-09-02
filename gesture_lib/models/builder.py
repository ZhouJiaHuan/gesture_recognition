from gesture_lib.ops.registry import build_from_cfg
from gesture_lib.registry import MODELS, MATCHERS


def build_model(model_cfg, default_args=None):
    return build_from_cfg(model_cfg, MODELS, default_args)


def build_matcher(matcher_cfg, default_args=None):
    return build_from_cfg(matcher_cfg, MATCHERS, default_args)
