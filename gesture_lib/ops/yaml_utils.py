import yaml
import os.path as osp
from easydict import EasyDict as edict


def parse_yaml(yaml_file):
    assert osp.exists(yaml_file), "{} not exists!".format(yaml_file)
    with open(yaml_file, 'r', encoding='utf-8') as f:
        yaml_data = f.read()
    try:
        cfgs = edict(yaml.safe_load(yaml_data))
    except Exception as e:
        print("parse failed for {}".format(yaml_file))
        raise(e)
    return cfgs
