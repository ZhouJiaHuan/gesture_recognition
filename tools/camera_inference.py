import sys
sys.path.append(".")

import argparse
from apis import Inference


def parse_args():
    parser = argparse.ArgumentParser(description="camera inference")
    parser.add_argument('config',
                        help="yaml config file path")
    parser.add_argument('checkpoint',
                        help="path to gesture model '.pth' checkpoint")
    parser.add_argument('--op_model',
                        help="path to openpose model folder, ignored when using trt-pose")
    parser.add_argument('--trt_model',
                        help="path to trt model, ignored when using openpose")
    parser.add_argument('--trt_json',
                        help="path to trt pose json file, ignored when using openpose")
    parser.add_argument('--extractor', default='surf',
                        help="extractor for person feature, 'surf' or 'akaze'")
    parser.add_argument('--seq_len', type=int, default=30,
                        help="sequence length for recognition, default 30")
    parser.add_argument('--show', action='store_true',
                        help="show the recognition result or not")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg_path = args.config
    ckp = args.checkpoint
    extractor = args.extractor
    if cfg_path is None or ckp is None:
        print("configure path and checkpoint must be specified!")
        raise
    if cfg_path.split('.')[-1] != 'yaml':
        print("invalid .yaml configure file!")
        raise
    if ckp.split('.')[-1] != 'pth':
        print("invalid .pth checkpoint file!")
        raise
    if extractor not in ['surf', 'akaze']:
        print("invalid extractor!, got {}".format(extractor))
        raise

    camera_infer = Inference(cfg_path=cfg_path,
                             checkpoints=ckp,
                             seq_len=args.seq_len,
                             op_model=args.op_model,
                             trt_model=args.trt_model,
                             pose_json_path=args.trt_json,
                             feature=extractor,
                             )
    camera_infer.infer_pipeline(show=args.show)


if __name__ == "__main__":
    main()
